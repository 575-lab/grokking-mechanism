# Grokking Mechanism

Reproduce grokking on modular addition with JAX, or watch it happen live in a browser with jax-js.

The experiment trains a small one-layer transformer on a fraction of all ordered pairs for `(a + b) mod p`. It first memorizes the training set, remains near chance on held-out pairs, and then abruptly generalizes after extended optimization.

## Python reference implementation

The Python project uses uv, JAX, Flax NNX, and Optax.

```bash
uv sync
uv run python main.py train --run-id grokking-mod113
uv run python main.py plot --run-dir runs/grokking-mod113
```

The default experiment uses `p=113`, a 30% training split, and 40,000 full-batch optimizer steps. Run artifacts include the configuration, split indices, metrics, Orbax checkpoints, and the generated training curve.

## Embedding structure (Liu et al. Figure 1)

`embedding_pca.py` reproduces Figure 1 of [Liu et al., *Towards Understanding Grokking*](https://arxiv.org/abs/2205.10343): the input embeddings at initialization, while the model is merely overfitting, and after it generalizes. The paper uses `p=53`.

```bash
uv run python main.py train --p 53 --frac-train 0.5 --epochs 20000 --run-id grokking-mod53
uv run python main.py embedding-pca --run-dir runs/grokking-mod53 --epochs 0 100 20000 --projection frequency
uv run python main.py circles --run-dir runs/grokking-mod53
```

The paper's own projection (`--projection pca`) does not show a circle here, and that is a property of the model rather than of the plot. This architecture learns **four** frequencies at once — for the seed above, f = 2, 9, 20 and 25, each carrying 19–26% of the embedding variance, which is why the top eight singular values are nearly equal and the top-2 plane holds only 29% of the variance. That plane is a superposition of four circles and looks unstructured.

Projecting onto a single frequency's plane recovers the circle: each of the four is round to within a 7–9% spread in radius. `--projection frequency` shows one such plane across the three training stages; `circles` shows the Fourier spectrum alongside every dominant frequency's circle.

## Interpretability notebook

`notebooks/interpretability.py` is a [marimo](https://docs.marimo.io/) notebook that inspects the grokked checkpoint for the Fourier structure behind modular addition. It is a plain Python file, so it diffs and reviews like source rather than like JSON.

```bash
uv run marimo edit notebooks/interpretability.py   # interactive
uv run marimo run notebooks/interpretability.py    # read-only app
uv run python notebooks/interpretability.py        # execute top to bottom as a script
```

It expects a completed `runs/grokking-mod113` run (config defaults, all 40,000 epochs), and reads the final checkpoint plus `metrics.npz` from it.

## Browser dashboard

Live demo: https://575-lab.github.io/grokking-mechanism/

The `web/` application reimplements the same model with jax-js and trains entirely in the browser. WebGPU is selected when available, with Wasm as the fallback. Training runs in a Web Worker so the live charts and prediction heatmap remain responsive.

Install the mise-managed Node.js toolchain and frontend dependencies:

```bash
mise install
mise run web-install
```

Start the dashboard:

```bash
mise run web-dev
```

Open the localhost URL printed by Vite. The dashboard provides:

- Quick-demo and full-reproduction presets.
- WebGPU/Wasm backend selection.
- Pause, resume, reset, and JSON export.
- Live log-scale loss and accuracy curves.
- Training throughput and learning-phase indicators.
- A modular-addition prediction field that distinguishes training and held-out pairs.

The quick preset is intended for development and backend checks. The full preset matches the Python model dimensions and 40,000-epoch schedule.

## Browser implementation notes

- Model parameters are plain jax-js trees; there is no Flax/NNX module layer in the browser.
- Embeddings use equality-based one-hot matrix multiplication. In jax-js 0.1.18, gather-backed `take` and `nn.oneHot` fail inside the differentiated graph because gather indices are promoted to `float32`.
- Attention uses `nn.dotProductAttention(q, k, v, { isCausal: true })` with tensors shaped `[batch, sequence, heads, headDimension]`.
- All operations carry an explicit batch dimension; the implementation does not depend on partially supported `vmap` paths.
- The development server intentionally avoids cross-origin isolation. This keeps jax-js on its bounded, single-threaded Wasm path; its SharedArrayBuffer backend has a substantially larger allocation footprint and can overflow the allocator on the full preset.
- On Wasm, the runtime accumulates weighted gradients in 128-example chunks and applies AdamW once per epoch. This is mathematically the same full-batch update while avoiding jax-js's approximately 2 GiB Wasm allocator overflow on the full preset. Test evaluation and heatmap inference are also chunked.

WebGPU is exposed only in a secure browser context. `http://localhost:5173` is treated as trustworthy, but opening the development server as `http://noble-bolivar:5173` from another machine normally is not. In that case the dashboard reports the Wasm fallback and uses bounded gradient accumulation. Use HTTPS if WebGPU is required over the LAN.

## Checks

Run browser unit tests and the production build through mise:

```bash
mise run web-test
mise run web-build
```

The tests exercise dataset construction, whole-batch autodiff through embeddings and causal attention, repeated AdamW updates, Wasm gradient accumulation, evaluation, heatmap extraction, and jax-js reference disposal. A full-preset smoke test runs one complete `p=113` epoch on Wasm to guard against allocator regressions.

## Project roadmap

See [PLAN.md](PLAN.md) for the architecture, implementation milestones, browser risks, and follow-up work such as checkpoint persistence and Python/browser curve overlays.
