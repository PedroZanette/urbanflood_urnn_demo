# Nowcasting Espaço-Temporal de Inundações Urbanas com U-RNN (demo funcional)

**Americas TechGuard | Período 7** — Implementação Funcional de Nowcasting Espaço-Temporal de
Inundações Urbanas com U-RNN

**Aluno:** Pedro Henrique Nunes Zanette

O enunciado oficial completo desta atividade está guardado em `data/` (PDF e DOCX) apenas como
referência acadêmica local — não faz parte da solução técnica e não é versionado no Git.

---

## Contexto da atividade

O Americas TechGuard desenvolve soluções de monitoramento, prevenção e resposta a eventos
climáticos extremos. Nesta etapa (Período 7), o desafio é estudar, executar e implementar uma
solução funcional inspirada no modelo **U-RNN**, publicado para nowcasting espaço-temporal de
inundações urbanas em alta resolução. Diferente do Período 6 (mapa estático de suscetibilidade),
aqui o objetivo é modelar **como a inundação evolui no tempo**.

A entrega exigida é um repositório GitHub público e/ou notebook Python funcional — **não** um
relatório técnico em PDF — com evidências claras de execução e relação direta com o U-RNN.

## Objetivo desta solução

Demonstrar, de ponta a ponta, um pipeline funcional de nowcasting espaço-temporal de inundação
urbana **inspirado na arquitetura do U-RNN** (encoder convolucional + memória recorrente
convolucional tipo ConvGRU + decoder convolucional), usando dados sintéticos tecnicamente
justificados, e gerar evidências objetivas e verificáveis (imagens, GIF, métricas, logs).

## Links obrigatórios de referência

- **Artigo científico:** [U-RNN: high-resolution spatiotemporal nowcasting of urban flooding](https://www.sciencedirect.com/science/article/pii/S002216942500455X)
- **Material suplementar / dataset (Figshare):** [Supplementary data of U-RNN](https://figshare.com/articles/dataset/Supplementary_data_of_U-RNN_high-resolution_spatiotemporal_nowcasting_of_urban_flooding_/28082549?file=51407804)
- **GitHub oficial:** [holmescao/U-RNN](https://github.com/holmescao/U-RNN)

Esses três materiais foram usados como guia técnico: o artigo para entender o problema, as
entradas/saídas e a arquitetura conceitual (Etapa 1 abaixo); o GitHub oficial para entender a
estrutura de um pipeline de nowcasting em PyTorch e o paradigma de pré-aquecimento do estado
recorrente; e o material suplementar do Figshare como fonte do dataset oficial — baixado, mas
não extraído integralmente (ver justificativa abaixo).

---

## 1. Estudo técnico: conceitos

### O que é nowcasting de inundações urbanas
É a predição de **curto prazo** da evolução espaço-temporal de uma inundação, a partir de dados
como chuva recente, terreno e estados hidrológicos observados/simulados. O foco é responder
"como o campo de inundação vai evoluir nos próximos minutos/horas", não apenas "onde inundação é
possível".

### Diferença entre nowcasting, suscetibilidade e risco
- **Suscetibilidade**: mapa estático que indica onde inundação é fisicamente mais provável,
  dado terreno, drenagem e uso do solo — não tem componente temporal explícita (foi o foco do
  Período 6).
- **Nowcasting**: prediz a evolução **temporal** de curtíssimo prazo do próprio evento de
  inundação, dado um estado inicial observado e forçantes recentes (chuva). Tem dimensão
  espaço-temporal.
- **Risco**: combina exposição, vulnerabilidade, população, infraestrutura crítica e impactos
  potenciais — vai muito além de prever profundidade de água. **Esta entrega não é, e não deve
  ser interpretada como, um modelo de risco**: ela demonstra apenas a peça de nowcasting.

### O que o U-RNN busca resolver
O artigo propõe um modelo de deep learning para gerar mapas futuros de profundidade de
inundação em alta resolução espacial, de forma muito mais rápida que simulações hidrodinâmicas
tradicionais (ex.: solvers de Saint-Venant 2D), mantendo coerência espaço-temporal — um
substituto (surrogate) aprendido para nowcasting operacional.

### Entradas e saídas esperadas de um problema U-RNN
- **Entradas**: série de chuva, terreno/DSM, impermeabilidade/uso do solo, área de drenagem, e
  estados temporais passados (mapas de profundidade de inundação observados/simulados).
- **Saídas**: mapas futuros de profundidade de inundação — uma sequência espaço-temporal, não
  apenas um mapa único.

### Ideia geral da arquitetura (U-like + ConvGRU)
Estrutura em formato "U" (como uma U-Net): um encoder convolucional reduz e extrai
características espaciais, e um decoder convolucional reconstrói o mapa de saída, com blocos
recorrentes convolucionais (ConvGRU/ConvLSTM) inseridos para capturar a evolução temporal — a
recorrência mantém um **estado espacial** (não um vetor), preservando a estrutura de grid ao
longo do tempo.

### Sliding Window-based Pre-warming
Paradigma de treinamento/inferência em que uma **janela deslizante de passos passados** é
alimentada ao modelo recorrente **apenas para inicializar (aquecer) o estado oculto**, antes de
começar a gerar as predições futuras propriamente ditas. Isso evita começar a prever a partir de
um estado zerado/arbitrário, reduzindo erro nos primeiros passos previstos.

### Materiais usados como referência
Ver seção "Links obrigatórios de referência" acima. O artigo fundamentou a escolha de arquitetura
(encoder/decoder convolucional + ConvGRU) e o paradigma de pré-aquecimento; o GitHub oficial
serviu de referência para a estrutura geral de um pipeline PyTorch de nowcasting; o material
suplementar do Figshare é o dataset oficial, baixado e mantido em `data/raw/` (ver próxima seção).

---

## Estratégia adotada e justificativa do uso de dados sintéticos/reduzidos

O arquivo oficial do material suplementar (`data/raw/urbanflood24_api.zip`) **foi baixado e é
mantido em `data/raw/`**, porém **não foi extraído integralmente** devido ao tamanho aproximado
de **116 GB após extração** (18 GB compactado). O ambiente de desenvolvimento local não possui
espaço em disco suficiente para essa extração completa.

O enunciado oficial da atividade permite explicitamente, nesses casos, o uso de "tutoriais,
quickstart, subconjuntos, exemplos reduzidos, dados sintéticos justificados ou pesos
pré-treinados [...] desde que a solução final execute e gere saídas verificáveis".

Estratégia adotada:
1. **Inspeção sem extração** (`src/inspect_zip.py`): o ZIP é auditado lendo apenas seu índice
   central (`zipfile.ZipFile(...).infolist()`), sem descompactar nenhum conteúdo — lista
   arquivos, extensões, tamanhos compactado/estimado extraído, e busca por tipos de arquivo úteis
   (`.npy`, `.npz`, `.csv`, `.json`, `.txt`, `.png`, `.tif`). O resumo é salvo em
   `outputs/zip_inventory.txt`. O script nunca extrai o ZIP inteiro; se encontrar candidatos
   pequenos, apenas **sugere** (não executa) um comando de extração pontual.
2. **Dados sintéticos justificados** (`src/synthetic_data.py`): geração determinística (com
   seed) de um dataset 2D leve (grade 64×64) que simula, de forma fisicamente inspirada mas
   simplificada, os mesmos tipos de canal do problema real — terreno com declividade e
   depressões, máscara de impermeabilidade/urbanização, chuva variável no tempo, e uma sequência
   de profundidade de inundação propagando por um esquema simples de fonte (chuva ×
   impermeabilidade) + roteamento gravitacional para células vizinhas mais baixas +
   infiltração. Isso permite treinar e avaliar o modelo (`URNNLite`) de ponta a ponta, sem
   depender do dataset de 116 GB.

Este caminho foi escolhido em vez de baixar pesos pré-treinados do repositório oficial porque o
repositório oficial (`holmescao/U-RNN`) não disponibiliza checkpoints prontos para inferência
imediata sem o ambiente/dataset original completo — tornando a geração de dados sintéticos a
via mais direta para uma entrega **funcional e verificável** dentro do prazo e da infraestrutura
disponíveis.

---

## Arquitetura implementada (`src/model.py`)

`URNNLite`: um modelo leve e didático, inspirado na estrutura do U-RNN, mas **não** uma
reprodução matemática do artigo original.

```
Entrada [B, T_in, C=4, H, W]
        │
        ▼
 Encoder convolucional (2× Conv2D + ReLU)   ← extrai features espaciais por frame
        │
        ▼
 ConvGRUCell (update/reset/candidate gates via Conv2D)  ← memória espaço-temporal
        │  (Sliding Window-based Pre-warming: T_in passos apenas aquecem o estado)
        ▼
 Rollout autoregressivo por T_out passos:
   Encoder → ConvGRU → Decoder (2× Conv2D + sigmoid) → mapa previsto [0,1]
        │
        ▼
Saída [B, T_out, 1, H, W]  (profundidade de inundação normalizada)
```

- **Por que convoluções?** O campo de inundação é espacialmente estruturado (água se propaga
  para vizinhos mais baixos); convoluções 2D capturam padrões locais de vizinhança de forma
  parametricamente eficiente — igual à lógica de uma U-Net.
- **Por que recorrência?** A profundidade de inundação em `t` depende do estado acumulado em
  `t-1, t-2, ...`. A `ConvGRUCell` mantém um estado oculto **espacial** (um mapa H×W, não um
  vetor), evoluindo passo a passo.
- **Relação com ConvGRU/U-RNN**: a célula implementa as 3 portas clássicas de um GRU (update,
  reset, candidate), substituindo multiplicações densas por convoluções 2D — a mesma ideia
  central do bloco recorrente do U-RNN.
- **Diferença para o modelo oficial**: sem múltiplas escalas de resolução, sem skip connections
  completas de uma U-Net profunda, sem DEM real de alta resolução nem rede de drenagem física.
  O objetivo é demonstrar a mesma ideia arquitetural (encoder conv → memória recorrente conv →
  decoder conv) de forma funcional e rápida em CPU, não reproduzir a fidelidade física ou o
  desempenho do artigo.

---

## Estrutura do projeto

```
urbanflood_urnn_demo/
├── README.md                          ← este arquivo
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── urnn_nowcasting_demo.ipynb      ← notebook executável, mesmo conteúdo em formato narrativo
├── src/
│   ├── synthetic_data.py               ← geração determinística do dataset sintético
│   ├── model.py                        ← URNNLite: ConvGRU-based, inspirado no U-RNN
│   ├── train_demo.py                   ← pipeline completo: treino + avaliação + outputs
│   ├── visualize.py                    ← plots, GIF, curva de loss
│   └── inspect_zip.py                  ← inspeciona o ZIP oficial sem extrair
├── data/
│   ├── raw/urbanflood24_api.zip         ← dataset oficial (Figshare), não extraído (ver acima)
│   ├── sample/                          ← reservado para amostra pontual (vazio por padrão)
│   ├── README.md
│   └── *.pdf, *.docx                    ← enunciado oficial (referência local, não versionado)
└── outputs/
    ├── input_sequence.png
    ├── target_sequence.png
    ├── prediction_sequence.png
    ├── flood_nowcasting.gif
    ├── loss_curve.png
    ├── metrics.json
    ├── run_log.txt
    └── zip_inventory.txt
```

## Instalação

Requer Python 3.10+ (testado com Python 3.14 em Ubuntu). Bibliotecas leves e padrão de
mercado: `numpy`, `matplotlib`, `torch` (CPU), `imageio` (GIF), `tqdm`, `scikit-learn`
(`train_test_split`), `jupyter`/`ipykernel` (notebook).

```bash
git clone <url-do-seu-repositorio>
cd urbanflood_urnn_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python src/train_demo.py
```

Gera dataset sintético, treina o modelo `URNNLite` por poucas épocas, avalia no conjunto de
teste e salva todas as evidências em `outputs/`. Roda em CPU (poucos minutos); usa GPU
automaticamente se disponível (`torch.cuda.is_available()`).

Para inspecionar o ZIP oficial (sem extrair):

```bash
python src/inspect_zip.py
```

Gera `outputs/zip_inventory.txt` com o inventário do dataset oficial.

## Como abrir o notebook

```bash
jupyter notebook notebooks/urnn_nowcasting_demo.ipynb
```

O notebook reproduz o mesmo pipeline de forma narrativa, célula a célula, exibindo as imagens
geradas em `outputs/` diretamente inline.

## Outputs gerados

| Arquivo | Descrição |
|---|---|
| `outputs/input_sequence.png` | Frames de profundidade de inundação passada usados como entrada |
| `outputs/target_sequence.png` | Mapas futuros reais (ground truth) da amostra de teste |
| `outputs/prediction_sequence.png` | Mapas futuros previstos pelo `URNNLite` |
| `outputs/flood_nowcasting.gif` | Animação comparando entrada / alvo / predição, quadro a quadro |
| `outputs/loss_curve.png` | Curva de MSE de treino ao longo das épocas |
| `outputs/metrics.json` | Configuração do dataset, do modelo, hiperparâmetros e métricas finais (MSE/MAE de teste) |
| `outputs/run_log.txt` | Log completo da execução (dispositivo, épocas, tempos) |
| `outputs/zip_inventory.txt` | Inventário do ZIP oficial, gerado por `src/inspect_zip.py` |

## Métricas geradas

Ver `outputs/metrics.json` para os valores exatos da última execução (MSE/MAE de teste, número
de parâmetros do modelo, tempo total de execução). O objetivo destas métricas é demonstrar que o
pipeline aprende um padrão não trivial (a perda de treino decresce de forma consistente — ver
`outputs/loss_curve.png`), não estabelecer estado da arte.

## Limitações

- Dados sintéticos não substituem validação com dados reais de campo.
- Não há calibração hidrológica local (sem dados pluviométricos ou topográficos reais de uma
  bacia urbana específica).
- Não há sensores reais de nível d'água nem validação hidrodinâmica (comparação com um solver
  físico como Saint-Venant 2D).
- O dataset oficial do U-RNN (Figshare) não foi extraído integralmente por limitação de
  armazenamento (~116 GB); apenas seu índice foi inspecionado.
- O modelo `URNNLite` é demonstrativo e inspirado no U-RNN — não é uma reprodução completa da
  arquitetura, do treinamento ou dos resultados reportados no artigo original.
- O esquema de propagação de água no gerador sintético é uma aproximação simplificada (roteamento
  gravitacional local), não um solver hidrodinâmico validado.

## Relação com o Americas TechGuard

Uma solução de nowcasting como esta poderia apoiar o Americas TechGuard em:
- **Monitoramento contínuo**: gerar previsões de curtíssimo prazo da evolução de um evento de
  inundação já em curso, a partir de observações recentes.
- **Prevenção e resposta**: antecipar áreas que devem inundar nos próximos passos de tempo,
  priorizando resposta operacional antes do pico do evento.
- **Geração de alertas**: alimentar um sistema de alerta automatizado com mapas de profundidade
  prevista, em vez de depender apenas de limiares estáticos de suscetibilidade.

Integrações futuras possíveis: dados reais de chuva (radar/pluviômetros), sensores de nível
d'água, imagens de satélite/DSM de alta resolução, geoprocessamento (drenagem real, uso do solo),
dashboards de monitoramento e disseminação automática de alertas.

**Diferença entre esta prova didática e uma prova de conceito robusta**: esta entrega demonstra
a arquitetura e o fluxo de dados com dados sintéticos e em escala reduzida; uma PoC robusta para
uso real exigiria o dataset oficial completo (ou dados reais equivalentes), validação
hidrodinâmica, calibração local, testes de generalização entre bacias urbanas distintas, e
avaliação de risco associada (exposição, vulnerabilidade, população, infraestrutura crítica),
que estão **fora do escopo** desta atividade.

## Licença / uso acadêmico

Este repositório é entregue para fins exclusivamente acadêmicos, no contexto da atividade
Americas TechGuard | Período 7 (Centro Universitário SENAI/SC). O artigo científico, o material
suplementar e o repositório oficial referenciados pertencem a seus respectivos autores
(Cao et al., ver link acima) e são usados aqui apenas como referência técnica, sem redistribuição
do dataset original.

## Referências

- Cao et al. — *U-RNN: high-resolution spatiotemporal nowcasting of urban flooding*. ScienceDirect:
  https://www.sciencedirect.com/science/article/pii/S002216942500455X
- Material suplementar (Figshare): https://figshare.com/articles/dataset/Supplementary_data_of_U-RNN_high-resolution_spatiotemporal_nowcasting_of_urban_flooding_/28082549?file=51407804
- Repositório oficial: https://github.com/holmescao/U-RNN
