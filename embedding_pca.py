"""Figure 1 of Liu et al., "Towards Understanding Grokking" (arXiv:2205.10343):
the first two principal components of the learned input embeddings, at
initialization, while the model is merely overfitting, and after it generalizes.

The representation the model groks is circular: embedding k lands near
(cos(2*pi*f*k/p), sin(2*pi*f*k/p)) for some frequency f, so that addition becomes
rotation. Whether that circle is visible in the top two principal components
depends on how many frequencies the model uses. When several are learned at
comparable strength, the top-2 plane is a superposition of their circles and looks
unstructured; projecting onto one frequency's plane recovers the circle. Hence the
two projections offered here.
"""

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

import data
import train as train_module
from config import GrokkingConfig
from model import GrokkingTransformer


def load_config(run_dir: Path) -> GrokkingConfig:
    stored = json.loads((run_dir / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(GrokkingConfig)}
    return GrokkingConfig(**{k: v for k, v in stored.items() if k in fields})


def model_at_epoch(cfg: GrokkingConfig, run_dir: Path, epoch: int):
    """The model as it was after `epoch` optimizer steps.

    Epoch 0 has no checkpoint — training only saves at the end of each chunk — but
    it is exactly reproducible, since train() seeds initialization with cfg.seed.
    """
    if epoch == 0:
        return GrokkingTransformer(cfg, rngs=nnx.Rngs(cfg.seed))
    chunk, rem = divmod(epoch, cfg.checkpoint_every)
    if rem:
        raise ValueError(f"epoch {epoch} is not a multiple of {cfg.checkpoint_every}")
    return train_module.load_checkpoint(cfg, run_dir, chunk - 1)


def number_embeddings(model, p: int) -> np.ndarray:
    """One centered row per integer in [0, p). Row p is the "=" token, which is
    punctuation rather than part of the representation of the group, so it is
    dropped; centering removes the mean embedding, which carries no periodicity."""
    embeddings = np.asarray(model.embed.embedding.value)[:p]
    return embeddings - embeddings.mean(axis=0)


def fourier_power(embeddings: np.ndarray) -> np.ndarray:
    """Fraction of embedding variance carried by each frequency, indexed 0..p//2.

    Taking the DFT along the *integer* axis asks how each embedding coordinate
    varies as a function of k, which is exactly where periodicity would live.
    """
    power = (np.abs(np.fft.rfft(embeddings, axis=0)) ** 2).sum(axis=1)
    return power / power.sum()


def frequency_plane(embeddings: np.ndarray, freq: int) -> np.ndarray:
    """Project onto the 2D plane that frequency `freq` occupies.

    The real and imaginary parts of the DFT component span that plane; orthonormalizing
    them and projecting the *full* embedding (not the filtered one) keeps any departure
    from a circle honest rather than imposing one by construction.
    """
    component = np.fft.rfft(embeddings, axis=0)[freq]
    basis, _ = np.linalg.qr(np.stack([component.real, component.imag], axis=1))
    return embeddings @ basis


def pca_plane(embeddings: np.ndarray) -> np.ndarray:
    """Project onto the top two principal components."""
    _, _, vt = np.linalg.svd(embeddings, full_matrices=False)
    return embeddings @ vt[:2].T


def accuracies(cfg: GrokkingConfig, model) -> tuple[float, float]:
    tokens, labels = data.build_dataset(cfg.p)
    train_idx, test_idx = data.split(len(tokens), cfg.seed, cfg.frac_train)
    correct = np.asarray(model(tokens).argmax(-1)) == labels
    return float(correct[train_idx].mean()), float(correct[test_idx].mean())


def _scatter_labels(ax, coords: np.ndarray, p: int):
    colors = plt.cm.viridis(np.linspace(0, 1, p))
    for k, (x, y) in enumerate(coords):
        ax.text(x, y, f"{k:02d}", color=colors[k], fontsize=7,
                ha="center", va="center", weight="bold")
    pad = 0.10 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(
    run_dir: Path,
    epochs: tuple[int, ...] = (0, 1000, None),
    projection: str = "pca",
    out_path: Path | None = None,
):
    """Reproduce Figure 1: the embeddings at three stages of training.

    projection="pca" is the paper's own choice. projection="frequency" instead fixes
    the plane of the dominant frequency *at the final snapshot* and shows every stage
    in it, which is what the paper does in its Figure 21 (right) to show that some of
    the structure is already present at initialization.
    """
    cfg = load_config(run_dir)
    epochs = tuple(cfg.total_epochs if e is None else e for e in epochs)
    stages = ["Initialization", "Overfitting", "Representation Learning"]

    freq = None
    if projection == "frequency":
        final = number_embeddings(model_at_epoch(cfg, run_dir, epochs[-1]), cfg.p)
        freq = int(np.argmax(fourier_power(final)[1:]) + 1)

    fig, axes = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs), 4.8))
    for ax, epoch, stage in zip(np.atleast_1d(axes), epochs, stages):
        model = model_at_epoch(cfg, run_dir, epoch)
        train_acc, test_acc = accuracies(cfg, model)
        embeddings = number_embeddings(model, cfg.p)
        coords = pca_plane(embeddings) if freq is None else frequency_plane(embeddings, freq)

        _scatter_labels(ax, coords, cfg.p)
        ax.set_title(
            f"{stage} ({epoch} iterations)\n"
            f"train acc: {train_acc:.2f} — val acc: {test_acc:.2f}",
            fontsize=10,
        )

    plane = (
        "their first two principal components"
        if freq is None
        else f"the frequency-{freq} plane of the final embeddings"
    )
    fig.suptitle(f"Input embeddings projected onto {plane} — (a + b) mod {cfg.p}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = out_path or (run_dir / f"embedding_{projection}.png")
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")
    return out_path


def plot_frequency_circles(
    run_dir: Path,
    epoch: int | None = None,
    n_freqs: int = 4,
    out_path: Path | None = None,
):
    """The circular representations the grokked model actually uses: the power
    spectrum of the embeddings, then one panel per dominant frequency."""
    cfg = load_config(run_dir)
    epoch = cfg.total_epochs if epoch is None else epoch
    embeddings = number_embeddings(model_at_epoch(cfg, run_dir, epoch), cfg.p)
    power = fourier_power(embeddings)
    top = (np.argsort(power[1:])[::-1] + 1)[:n_freqs]

    fig, axes = plt.subplots(1, n_freqs + 1, figsize=(4.0 * (n_freqs + 1), 4.6))
    spectrum, panels = axes[0], axes[1:]

    spectrum.bar(np.arange(1, len(power)), power[1:] * 100, color="0.35")
    for f in top:
        spectrum.bar(f, power[f] * 100, color="tab:red")
    spectrum.set_xlabel("frequency")
    spectrum.set_ylabel("% of embedding variance")
    spectrum.set_title(f"Fourier spectrum of the embeddings\n({epoch} iterations)", fontsize=10)
    spectrum.spines[["top", "right"]].set_visible(False)

    for ax, f in zip(panels, top):
        coords = frequency_plane(embeddings, int(f))
        radii = np.linalg.norm(coords, axis=1)
        _scatter_labels(ax, coords, cfg.p)
        ax.set_title(
            f"frequency {f} — {power[f] * 100:.0f}% of variance\n"
            f"radius spread {radii.std() / radii.mean() * 100:.0f}%",
            fontsize=10,
        )

    fig.suptitle(
        f"The grokked representation of Z/{cfg.p}Z: one circle per learned frequency",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = out_path or (run_dir / "embedding_circles.png")
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")
    return out_path
