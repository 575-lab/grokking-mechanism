import argparse
from datetime import datetime
from pathlib import Path

from config import GrokkingConfig

RUNS_DIR = Path(__file__).parent / "runs"


def latest_run_dir() -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit(f"no runs found under {RUNS_DIR}")
    return runs[-1]


def cmd_train(args):
    import train as train_module

    cfg = GrokkingConfig(
        p=args.p,
        frac_train=args.frac_train,
        seed=args.seed,
        total_epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    train_module.train(cfg, run_dir)


def cmd_plot(args):
    import plot as plot_module

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    plot_module.plot(run_dir)


def cmd_embedding_pca(args):
    import embedding_pca

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    epochs = tuple(args.epochs) if args.epochs else (0, 1000, None)
    embedding_pca.plot(run_dir, epochs=epochs, projection=args.projection)


def cmd_trajectory(args):
    import embedding_pca

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    embedding_pca.plot_frequency_trajectory(run_dir, n_freqs=args.n_freqs, stride=args.stride)


def cmd_circles(args):
    import embedding_pca

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    embedding_pca.plot_frequency_circles(run_dir, epoch=args.epoch, n_freqs=args.n_freqs)


def main():
    parser = argparse.ArgumentParser(description="Grokking on modular addition (a + b) mod p")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train and checkpoint a run")
    p_train.add_argument("--p", type=int, default=GrokkingConfig.p, help="modulus")
    p_train.add_argument("--frac-train", type=float, default=GrokkingConfig.frac_train)
    p_train.add_argument("--epochs", type=int, default=GrokkingConfig.total_epochs)
    p_train.add_argument("--checkpoint-every", type=int, default=GrokkingConfig.checkpoint_every)
    p_train.add_argument("--seed", type=int, default=GrokkingConfig.seed)
    p_train.add_argument("--run-id", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_plot = sub.add_parser("plot", help="plot the training curve for a run")
    p_plot.add_argument("--run-dir", type=str, default=None, help="defaults to the most recent run")
    p_plot.set_defaults(func=cmd_plot)

    p_pca = sub.add_parser("embedding-pca", help="PCA of the input embeddings over training")
    p_pca.add_argument("--run-dir", type=str, default=None, help="defaults to the most recent run")
    p_pca.add_argument(
        "--epochs", type=int, nargs="+", default=None,
        help="the snapshots to show (default: 0, 1000, and the final epoch). Runs that "
             "kept only some checkpoints can pass fewer, e.g. --epochs 0 40000",
    )
    p_pca.add_argument(
        "--projection", choices=("pca", "frequency"), default="pca",
        help="'pca' reproduces the paper's panels; 'frequency' fixes the dominant "
             "frequency's plane, where the circle is visible",
    )
    p_pca.set_defaults(func=cmd_embedding_pca)

    p_circ = sub.add_parser("circles", help="one circle per learned Fourier frequency")
    p_circ.add_argument("--run-dir", type=str, default=None, help="defaults to the most recent run")
    p_circ.add_argument("--epoch", type=int, default=None, help="defaults to the final epoch")
    p_circ.add_argument("--n-freqs", type=int, default=4)
    p_circ.set_defaults(func=cmd_circles)

    p_traj = sub.add_parser(
        "trajectory", help="track each learned frequency across every checkpoint"
    )
    p_traj.add_argument("--run-dir", type=str, default=None, help="defaults to the most recent run")
    p_traj.add_argument("--n-freqs", type=int, default=4)
    p_traj.add_argument(
        "--stride", type=int, default=1,
        help="check every Nth checkpoint; 1 uses all of them",
    )
    p_traj.set_defaults(func=cmd_trajectory)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
