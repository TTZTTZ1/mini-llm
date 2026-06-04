from __future__ import annotations

import argparse
import os
from pathlib import Path

_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "mini_llm_cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt


def plot_run(run_dir: Path, output_dir: Path) -> list[Path]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    loss_path = output_dir / "loss_curve.png"
    plt.figure()
    plt.plot(metrics["step"], metrics["train_loss"], label="train_loss")
    plt.plot(metrics["step"], metrics["val_loss"], label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()
    paths.append(loss_path)

    ppl_path = output_dir / "ppl_curve.png"
    plt.figure()
    plt.plot(metrics["step"], metrics["val_ppl"], label="val_ppl")
    plt.xlabel("step")
    plt.ylabel("perplexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ppl_path)
    plt.close()
    paths.append(ppl_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    for path in plot_run(args.run_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
