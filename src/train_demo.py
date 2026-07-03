"""
Pipeline de treino/inferencia do demo de nowcasting de inundacao urbana (U-RNN Lite).

Executa: python src/train_demo.py

Passos:
    1. Gera dataset sintetico (src/synthetic_data.py).
    2. Separa treino/teste.
    3. Instancia o modelo (src/model.py: URNNLite, ConvGRU).
    4. Treina por poucas epocas (leve, roda em CPU em minutos).
    5. Calcula MSE de treino e teste.
    6. Faz predicao em uma sequencia de teste.
    7. Salva todas as evidencias em outputs/.

Funciona em CPU por padrao; usa GPU automaticamente se `torch.cuda.is_available()`.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import URNNLite  # noqa: E402
from synthetic_data import SyntheticDatasetConfig, generate_synthetic_dataset  # noqa: E402
from visualize import save_all_outputs  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "run_log.txt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Hiperparametros deliberadamente pequenos para manter a execucao leve em CPU.
N_SAMPLES = 60
GRID_SIZE = 64
T_IN = 5
T_OUT = 5
EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
SEED = 42


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("urnn_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def batched_indices(n: int, batch_size: int):
    idx = np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def main() -> None:
    logger = setup_logging()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo utilizado: {device}")
    logger.info(f"Versao do PyTorch: {torch.__version__}")
    logger.info(
        "Dataset oficial (Figshare, ~18GB zip / ~116GB extraido) NAO foi extraido "
        "integralmente por limitacao de armazenamento local. Usando dados sinteticos "
        "gerados por src/synthetic_data.py, conforme justificado no README."
    )

    t_start = time.time()

    # 1. Gerar dataset sintetico ---------------------------------------------------------
    logger.info(
        f"Gerando dataset sintetico: N={N_SAMPLES}, grid={GRID_SIZE}x{GRID_SIZE}, "
        f"T_in={T_IN}, T_out={T_OUT}, seed={SEED}"
    )
    config = SyntheticDatasetConfig(
        n_samples=N_SAMPLES, height=GRID_SIZE, width=GRID_SIZE, t_in=T_IN, t_out=T_OUT, seed=SEED
    )
    X, y, meta = generate_synthetic_dataset(config)
    logger.info(f"X shape: {X.shape} | y shape: {y.shape}")

    # 2. Separar treino/teste --------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )
    logger.info(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    # 3. Instanciar modelo --------------------------------------------------------------
    model = URNNLite(in_channels=X.shape[2], base_channels=16, hidden_channels=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo: URNNLite (ConvGRU) | parametros treinaveis: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 4. Treinar por poucas epocas --------------------------------------------------------
    logger.info(f"Iniciando treinamento: {EPOCHS} epocas, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    losses = []
    n_train = X_train_t.shape[0]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        perm = np.random.permutation(n_train)

        for batch_idx in batched_indices(n_train, BATCH_SIZE):
            batch_idx = perm[batch_idx]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(xb, t_out=T_OUT)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        logger.info(f"Epoca {epoch:02d}/{EPOCHS} | MSE treino: {mean_loss:.5f}")

    # 5. Avaliar no teste --------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t, t_out=T_OUT)
        test_mse = criterion(test_pred, y_test_t).item()
        test_mae = torch.mean(torch.abs(test_pred - y_test_t)).item()

    logger.info(f"MSE final (teste): {test_mse:.5f}")
    logger.info(f"MAE final (teste): {test_mae:.5f}")

    # 6. Predicao em uma sequencia de teste para visualizacao -----------------------------
    sample_idx = 0
    with torch.no_grad():
        single_input = X_test_t[sample_idx:sample_idx + 1]
        single_pred = model(single_input, t_out=T_OUT).cpu().numpy()[0, :, 0]  # [T_out, H, W]

    input_seq_vis = X_test[sample_idx, :, 0]  # canal 0 = profundidade passada
    target_seq_vis = y_test[sample_idx, :, 0]
    pred_seq_vis = single_pred

    # 7. Salvar outputs --------------------------------------------------------------
    logger.info("Salvando evidencias em outputs/ (imagens, GIF, metricas)...")
    save_all_outputs(input_seq_vis, target_seq_vis, pred_seq_vis, losses, OUTPUT_DIR)

    elapsed = time.time() - t_start

    metrics = {
        "dataset": {
            "n_samples": N_SAMPLES,
            "grid_size": GRID_SIZE,
            "t_in": T_IN,
            "t_out": T_OUT,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "seed": SEED,
            "source": "synthetic (src/synthetic_data.py) - dataset oficial nao extraido integralmente (ver data/README.md)",
        },
        "model": {
            "name": "URNNLite (ConvGRU-based, U-RNN inspired)",
            "n_parameters": int(n_params),
            "device": str(device),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_train_mse": losses[-1],
            "loss_curve": losses,
        },
        "evaluation": {
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        "runtime_seconds": round(elapsed, 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Metricas salvas em: {METRICS_PATH}")
    logger.info(f"Tempo total de execucao: {elapsed:.1f}s")
    logger.info("Pipeline concluido com sucesso.")


if __name__ == "__main__":
    main()
