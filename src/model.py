"""
Modelo didatico inspirado no U-RNN para nowcasting espaco-temporal de inundacao urbana.

Referencia conceitual: Cao et al., "U-RNN: high-resolution spatiotemporal nowcasting of
urban flooding" (ver link no README). O artigo original combina uma estrutura U-like
(encoder/decoder convolucional com skip connections, como um U-Net) com celulas
recorrentes convolucionais (ConvGRU/ConvLSTM) no "gargalo" e/ou em cada escala, alem de
um paradigma de treinamento chamado Sliding Window-based Pre-warming: janelas de tempo
passadas sao usadas para "aquecer" (inicializar) o estado oculto da rede recorrente
antes de comecar a prever os proximos passos, reduzindo o erro de um estado inicial
arbitrario (zero) no começo da sequencia.

Por que convolucoes?
    O campo de inundacao e uma grandeza espacialmente estruturada (agua se propaga
    para vizinhos mais baixos). Convolucoes 2D capturam padroes locais de vizinhanca
    (declive, obstaculos urbanos) de forma parametricamente eficiente, exatamente como
    em uma U-Net de segmentacao de imagens.

Por que recorrencia?
    A inundacao e um processo temporal com memoria: a profundidade em t depende do
    estado acumulado em t-1, t-2, etc. Uma celula recorrente convolucional (ConvGRU)
    mantém um estado oculto espacial (um "mapa" de memoria, nao um vetor) que evolui
    passo a passo, absorvendo chuva e propagando agua ao longo do tempo.

Relacao com ConvGRU/U-RNN:
    A celula `ConvGRUCell` abaixo implementa as mesmas 3 portas de um GRU classico
    (update, reset, candidate), mas substituindo as multiplicacoes matriciais densas
    por convolucoes 2D, para que o estado oculto preserve a estrutura espacial do
    grid (H x W), assim como no bloco recorrente do U-RNN.

Diferenca em relacao ao modelo oficial:
    Este modelo eh muito mais raso e leve (poucas camadas, sem multiplas escalas de
    resolucao, sem skip connections completas de uma U-Net profunda, sem as
    features fisicas completas do artigo - ex.: DEM de alta resolucao, rede de
    drenagem, condicoes de contorno hidraulicas). O objetivo aqui eh didatico:
    demonstrar, de forma funcional e auditavel, a mesma ideia arquitetural
    (encoder convolucional -> memoria recorrente convolucional -> decoder
    convolucional) rodando em CPU em poucos minutos, nao reproduzir o desempenho
    ou a fidelidade fisica do artigo original.
"""

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """Celula GRU convolucional: estado oculto e um tensor espacial [B, hidden, H, W]."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # Uma unica convolucao produz as portas update (z) e reset (r) de uma vez,
        # a partir da concatenacao [entrada, estado_anterior] no eixo de canais.
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding
        )
        # Convolucao separada para o estado candidato (depende do reset gate).
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        update_gate, reset_gate = torch.chunk(gates, 2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)

        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))

        h_new = (1 - update_gate) * h_prev + update_gate * candidate
        return h_new


class ConvEncoder(nn.Module):
    """Encoder convolucional: extrai features espaciais de cada frame de entrada."""

    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Decoder convolucional: projeta o estado recorrente de volta para um mapa de profundidade."""

    def __init__(self, hidden_channels: int, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Sigmoid: profundidade normalizada de inundacao em [0, 1], conforme a
        # normalizacao aplicada em synthetic_data.py.
        return torch.sigmoid(self.net(h))


class URNNLite(nn.Module):
    """
    Arquitetura leve inspirada em U-RNN: Encoder conv -> ConvGRU recorrente -> Decoder conv.

    Fluxo (implementa o paradigma Sliding Window-based Pre-warming):
      1. "Pre-warming": os T_in frames de entrada (janela deslizante do passado) sao
         percorridos um a um pela ConvGRU, apenas para inicializar/aquecer o estado
         oculto — nenhuma predicao e emitida nessa fase.
      2. "Rollout": a partir do estado ja aquecido, o modelo gera T_out previsoes
         futuras de forma autoregressiva, realimentando sua propria saida (o mapa de
         profundidade previsto) como entrada do proximo passo, junto com os canais
         auxiliares do ultimo frame observado (chuva, terreno, impermeabilidade)
         mantidos constantes como aproximacao simples de forcante externo.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 16, hidden_channels: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_channels)
        self.gru = ConvGRUCell(base_channels, hidden_channels)
        self.decoder = ConvDecoder(hidden_channels, out_channels=1)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, t_out: int) -> torch.Tensor:
        """
        x: [B, T_in, C, H, W] — janela de observacao passada.
        t_out: numero de passos futuros a prever.
        retorna: [B, T_out, 1, H, W] — profundidade de inundacao prevista, em [0, 1].
        """
        b, t_in, c, h, w = x.shape
        hidden = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)

        # Fase de pre-warming: consome a janela passada para aquecer o estado oculto.
        for t in range(t_in):
            feat = self.encoder(x[:, t])
            hidden = self.gru(feat, hidden)

        # Fase de rollout autoregressivo: gera T_out mapas futuros.
        last_frame = x[:, -1].clone()  # usado para manter canais auxiliares (chuva/terreno/imperm)
        predictions = []
        for _ in range(t_out):
            feat = self.encoder(last_frame)
            hidden = self.gru(feat, hidden)
            pred = self.decoder(hidden)  # [B, 1, H, W]
            predictions.append(pred)

            # Realimenta a predicao como canal de profundidade do proximo passo,
            # mantendo os canais auxiliares (chuva/terreno/impermeabilidade) do
            # ultimo frame observado como aproximacao simples.
            last_frame = torch.cat([pred, last_frame[:, 1:]], dim=1)

        return torch.stack(predictions, dim=1)  # [B, T_out, 1, H, W]


if __name__ == "__main__":
    model = URNNLite(in_channels=4, base_channels=16, hidden_channels=32)
    dummy_x = torch.randn(2, 5, 4, 64, 64)
    out = model(dummy_x, t_out=5)
    print(f"Output shape: {out.shape}")  # esperado: [2, 5, 1, 64, 64]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Numero de parametros: {n_params:,}")
