"""
Funcoes de visualizacao para o demo de nowcasting de inundacao urbana inspirado em U-RNN.

Gera todas as evidencias visuais exigidas pela atividade: sequencias de entrada,
alvo e predicao, GIF temporal e curva de perda de treinamento.
"""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

FLOOD_CMAP = "Blues"


def _sequence_figure(sequence: np.ndarray, title: str, suptitle: str) -> plt.Figure:
    """sequence: [T, H, W] em [0, 1]. Plota um painel com um frame por coluna."""
    t = sequence.shape[0]
    fig, axes = plt.subplots(1, t, figsize=(2.2 * t, 2.6))
    if t == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(sequence[i], cmap=FLOOD_CMAP, vmin=0, vmax=1)
        ax.set_title(f"t+{i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{suptitle}\n{title}", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Profundidade normalizada")
    return fig


def plot_input_sequence(input_seq: np.ndarray, out_path: Path) -> None:
    """input_seq: [T_in, H, W] canal de profundidade passada."""
    fig = _sequence_figure(
        input_seq,
        title="Input sequence (past flood depth frames)",
        suptitle="U-RNN inspired urban flood nowcasting demo",
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_target_sequence(target_seq: np.ndarray, out_path: Path) -> None:
    fig = _sequence_figure(
        target_seq,
        title="Target future flood maps (ground truth)",
        suptitle="U-RNN inspired urban flood nowcasting demo",
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_sequence(pred_seq: np.ndarray, out_path: Path) -> None:
    fig = _sequence_figure(
        pred_seq,
        title="Predicted future flood maps (URNNLite output)",
        suptitle="U-RNN inspired urban flood nowcasting demo",
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(losses: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", color="#1f77b4")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE loss (treino)")
    ax.set_title("U-RNN inspired urban flood nowcasting demo\nCurva de perda de treinamento")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_comparison_gif(
    input_seq: np.ndarray,
    target_seq: np.ndarray,
    pred_seq: np.ndarray,
    out_path: Path,
    fps: int = 2,
) -> None:
    """
    Gera um GIF animado comparando, lado a lado, a sequencia de entrada (contexto),
    o alvo real e a predicao do modelo, quadro a quadro.
    """
    t_out = target_seq.shape[0]
    frames = []

    cmap = plt.get_cmap(FLOOD_CMAP)

    for step in range(t_out):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

        axes[0].imshow(input_seq[-1], cmap=cmap, vmin=0, vmax=1)
        axes[0].set_title("Last observed input")

        axes[1].imshow(target_seq[step], cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title(f"Target t+{step+1}")

        axes[2].imshow(pred_seq[step], cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title(f"Prediction t+{step+1}")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("U-RNN inspired urban flood nowcasting demo", fontsize=12)
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)


def save_all_outputs(
    input_seq: np.ndarray,
    target_seq: np.ndarray,
    pred_seq: np.ndarray,
    losses: list,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_input_sequence(input_seq, output_dir / "input_sequence.png")
    plot_target_sequence(target_seq, output_dir / "target_sequence.png")
    plot_prediction_sequence(pred_seq, output_dir / "prediction_sequence.png")
    plot_loss_curve(losses, output_dir / "loss_curve.png")
    make_comparison_gif(input_seq, target_seq, pred_seq, output_dir / "flood_nowcasting.gif")
