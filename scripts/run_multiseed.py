"""Run a capacity-matched VideoEENet ablation over multiple random seeds."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from math import sqrt
from pathlib import Path
import numpy as np
import torch

def mean_ci(values: list[float]) -> dict[str, float]:
    from scipy.stats import t
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Confidence intervals require at least two finite seed results")
    mean = float(np.mean(values)); std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    # Exact two-sided 95% t critical values for the common small-seed protocol.
    critical = float(t.ppf(0.975, df=len(values) - 1))
    return {"mean": mean, "std": std, "ci95": critical * std / sqrt(len(values)) if values else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed VideoEENet experiment with 95% confidence intervals.")
    parser.add_argument("--data-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temporal-mode", choices=["convlstm", "spatial_transformer", "hybrid"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31]); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=5); parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--train-split", default="Train"); parser.add_argument("--val-split", default="auto"); parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(); root = Path(args.output_dir); records = []
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("Provide at least two distinct seeds")
    for seed in args.seeds:
        run = root / args.temporal_mode / f"seed_{seed}"
        command = [sys.executable, "scripts/train.py", "--data-root", args.data_root, "--train-split", args.train_split, "--val-split", args.val_split, "--output-dir", str(run), "--temporal-mode", args.temporal_mode, "--seed", str(seed), "--epochs", str(args.epochs), "--seq-len", str(args.seq_len), "--image-size", str(args.image_size), "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim), "--base-channels", str(args.base_channels), "--num-workers", str(args.num_workers), "--resume", "auto"]
        command.extend(['--require-cuda', '--report-after-training'])
        subprocess.run(command, check=True)
        checkpoint = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
        records.append({"seed": seed, **checkpoint["best"]})
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{args.temporal_mode}_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "psnr", "ssim"]); writer.writeheader(); writer.writerows(records)
    summary = {"temporal_mode": args.temporal_mode, "seeds": args.seeds, "hidden_dim": args.hidden_dim, "seq_len": args.seq_len, "psnr": mean_ci([row["psnr"] for row in records]), "ssim": mean_ci([row["ssim"] for row in records])}
    (root / f"{args.temporal_mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
