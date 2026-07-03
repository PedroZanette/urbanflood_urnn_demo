"""
Gerador de dataset sintetico 2D para nowcasting de inundacao urbana.

Por que dados sinteticos?
-------------------------
O dataset oficial do U-RNN (Figshare, ~18GB compactado / ~116GB extraido) nao pode
ser extraido integralmente no ambiente local desta atividade por limitacao de
armazenamento (ver data/README.md e outputs/zip_inventory.txt). O enunciado da
atividade permite explicitamente o uso de dados sinteticos justificados, desde que
a solucao final execute e gere saidas verificaveis.

Este modulo simula, de forma simplificada mas fisicamente inspirada, os mesmos
tipos de canal que o problema real de nowcasting de inundacao urbana usa:
    - terreno (proxy de DSM/elevacao), com declividade e depressoes (bacias);
    - impermeabilidade/urbanizacao (fracao de area impermeavel por celula);
    - chuva variavel no tempo;
    - profundidade de inundacao evoluindo no espaco e no tempo, por um esquema
      simples de acumulo (chuva x impermeabilidade), roteamento para celulas
      vizinhas mais baixas (fluxo descendente) e perda por infiltracao.

Nao ha pretensao de reproduzir um modelo hidrodinamico (ex.: Saint-Venant 2D,
usado no artigo original). O objetivo e gerar sequencias espaco-temporais
coerentes o suficiente para treinar/avaliar um modelo de nowcasting didatico.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticDatasetConfig:
    n_samples: int = 40
    height: int = 64
    width: int = 64
    t_in: int = 5
    t_out: int = 5
    seed: int = 42


def _generate_terrain(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Terreno sintetico: rampa geral (declividade) + depressoes gaussianas (bacias)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Declividade geral: terreno mais alto num canto, mais baixo no oposto.
    slope_dir = rng.uniform(0, 2 * np.pi)
    slope = np.cos(slope_dir) * xx / w + np.sin(slope_dir) * yy / h

    terrain = slope * 10.0  # metros, escala arbitraria

    # Depressoes (bacias de acumulo de agua), simuladas como gaussianas invertidas.
    n_basins = rng.integers(2, 5)
    for _ in range(n_basins):
        cy = rng.uniform(0.15, 0.85) * h
        cx = rng.uniform(0.15, 0.85) * w
        depth = rng.uniform(3.0, 7.0)
        spread = rng.uniform(0.06, 0.15) * max(h, w)
        terrain -= depth * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * spread ** 2)))

    # Ruido de alta frequencia leve (microtopografia).
    terrain += rng.normal(0, 0.15, size=(h, w))

    return terrain.astype(np.float32)


def _generate_impermeability(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Mascara de impermeabilidade/urbanizacao em [0, 1]: 1 = totalmente impermeavel (urbano)."""
    base = rng.uniform(0.2, 0.4)
    field = np.full((h, w), base, dtype=np.float32)

    # "Quadras" urbanas retangulares com maior impermeabilidade.
    n_blocks = rng.integers(4, 9)
    for _ in range(n_blocks):
        bh = rng.integers(h // 10, h // 4)
        bw = rng.integers(w // 10, w // 4)
        y0 = rng.integers(0, max(1, h - bh))
        x0 = rng.integers(0, max(1, w - bw))
        field[y0:y0 + bh, x0:x0 + bw] += rng.uniform(0.4, 0.6)

    field += rng.normal(0, 0.03, size=(h, w))
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _generate_rain_series(t: int, rng: np.random.Generator) -> np.ndarray:
    """Serie temporal de intensidade de chuva (escalar por passo), com um pico central."""
    peak_t = rng.uniform(0.2, 0.6) * t
    width = rng.uniform(0.15, 0.3) * t
    intensity = rng.uniform(0.7, 1.3)
    steps = np.arange(t)
    rain = intensity * np.exp(-((steps - peak_t) ** 2) / (2 * width ** 2))
    rain += rng.normal(0, 0.03, size=t).clip(min=0)
    return np.clip(rain, 0, None).astype(np.float32)


def _simulate_flood_sequence(
    terrain: np.ndarray,
    imperm: np.ndarray,
    rain_series: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """
    Simula a evolucao da profundidade de inundacao ao longo do tempo.

    Esquema simplificado (nao e um solver hidrodinamico completo):
      1. Fonte de agua: chuva x impermeabilidade (areas impermeaveis geram mais
         escoamento superficial direto; areas permeaveis infiltram parte da agua).
      2. Roteamento: cada celula transfere uma fracao de sua lamina d'agua para
         o vizinho de menor elevacao efetiva (terreno + agua acumulada), imitando
         fluxo gravitacional para baixo.
      3. Perda por infiltracao residual proporcional a (1 - impermeabilidade).
    """
    h, w = terrain.shape
    depth = np.zeros((h, w), dtype=np.float32)
    sequence = np.zeros((n_steps, h, w), dtype=np.float32)

    infiltration_rate = 0.06 * (1.0 - imperm)  # areas permeaveis perdem mais agua
    runoff_gain = 0.5 + 0.5 * imperm  # areas impermeaveis acumulam mais rapido

    for t in range(n_steps):
        depth += rain_series[t] * runoff_gain
        depth *= (1.0 - infiltration_rate)
        depth = _route_flow(depth, terrain)
        depth = np.clip(depth, 0.0, None)
        sequence[t] = depth

    return sequence


def _route_flow(depth: np.ndarray, terrain: np.ndarray, flow_fraction: float = 0.25) -> np.ndarray:
    """
    Roteia uma fracao da lamina d'agua de cada celula para o vizinho (4-conectado)
    com menor "superficie efetiva" (terreno + agua), se este for mais baixo.
    Vetorizado com deslocamentos (roll) para manter o script leve em CPU.
    """
    surface = terrain + depth
    new_depth = depth.copy()

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in shifts:
        neighbor_surface = np.roll(surface, shift=(dy, dx), axis=(0, 1))

        downhill = surface > neighbor_surface
        diff = np.where(downhill, surface - neighbor_surface, 0.0)

        outflow = np.minimum(depth * flow_fraction * 0.25, diff * 0.5)
        outflow = np.where(downhill, outflow, 0.0)

        new_depth -= outflow
        # A agua que sai desta celula entra na celula vizinha correspondente
        # (deslocamento inverso), preservando o balanco de massa.
        neighbor_depth_gain = np.roll(outflow, shift=(-dy, -dx), axis=(0, 1))
        new_depth += neighbor_depth_gain

    return new_depth


def _normalize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + eps)


def generate_synthetic_dataset(config: SyntheticDatasetConfig):
    """
    Gera X, y para nowcasting espaco-temporal de inundacao urbana.

    Retorna:
        X: np.ndarray [N, T_in, C=4, H, W]
            Canais: 0=profundidade passada (normalizada), 1=chuva (broadcast espacial),
                    2=terreno normalizado, 3=impermeabilidade.
        y: np.ndarray [N, T_out, 1, H, W]
            Profundidade de inundacao futura normalizada em [0, 1].
        meta: dict com terrenos/impermeabilidade/chuva usados (para visualizacao).
    """
    rng = np.random.default_rng(config.seed)
    h, w = config.height, config.width
    t_total = config.t_in + config.t_out

    X = np.zeros((config.n_samples, config.t_in, 4, h, w), dtype=np.float32)
    y = np.zeros((config.n_samples, config.t_out, 1, h, w), dtype=np.float32)

    last_terrain = last_imperm = last_rain = last_flood_seq = None

    for n in range(config.n_samples):
        sample_seed = rng.integers(0, 2**31 - 1)
        sample_rng = np.random.default_rng(sample_seed)

        terrain = _generate_terrain(h, w, sample_rng)
        imperm = _generate_impermeability(h, w, sample_rng)
        rain_series = _generate_rain_series(t_total, sample_rng)
        flood_seq = _simulate_flood_sequence(terrain, imperm, rain_series, t_total)

        # Normalizacao por amostra para manter a saida do modelo em [0, 1].
        flood_max = flood_seq.max() + 1e-6
        flood_norm = flood_seq / flood_max
        terrain_norm = _normalize(terrain)
        rain_norm = rain_series / (rain_series.max() + 1e-6)

        for ti in range(config.t_in):
            X[n, ti, 0] = flood_norm[ti]
            X[n, ti, 1] = rain_norm[ti]  # escalar de chuva, broadcast uniforme por celula
            X[n, ti, 2] = terrain_norm
            X[n, ti, 3] = imperm

        for to in range(config.t_out):
            y[n, to, 0] = flood_norm[config.t_in + to]

        last_terrain, last_imperm, last_rain, last_flood_seq = terrain, imperm, rain_series, flood_norm

    meta = {
        "terrain": last_terrain,
        "impermeability": last_imperm,
        "rain_series": last_rain,
        "flood_sequence_normalized": last_flood_seq,
        "config": config,
    }
    return X, y, meta


if __name__ == "__main__":
    cfg = SyntheticDatasetConfig()
    X, y, meta = generate_synthetic_dataset(cfg)
    print(f"X shape: {X.shape}  (N, T_in, C, H, W)")
    print(f"y shape: {y.shape}  (N, T_out, 1, H, W)")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]  y range: [{y.min():.3f}, {y.max():.3f}]")
