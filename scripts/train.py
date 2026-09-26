"""RunPod-friendly VideoEENet training with AMP, tqdm, TensorBoard, and resume."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataloader import REVIDEVideoDataset
from model import VideoEENet
from occlusion import apply_occluder, occluder_key, occluder_mask, seeded_occluder_mask


def occlude_batch(frames: torch.Tensor, frame_paths, coverage_range: Tuple[float, float], scope: str, eval_coverage: float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hide part of each window behind an opaque occluder and append the mask as a 4th channel.

    frames [B, T, 3, H, W] -> ([B, T, 4, H, W], masks [B, 1, H, W]). Training draws a coverage
    from coverage_range (and, for scope "mixed", lens or current per sample); evaluation passes
    eval_coverage and uses the deterministic occluder keyed by the target frame path.
    """
    batch, steps, _, height, width = frames.shape
    out, masks = [], []
    for index in range(batch):
        if eval_coverage is not None:
            mask = seeded_occluder_mask(height, width, eval_coverage, occluder_key(frame_paths[index]))
        else:
            mask = occluder_mask(height, width, random.uniform(*coverage_range))
        mask = mask.to(frames.device, frames.dtype)
        sample_scope = random.choice(("lens", "current")) if scope == "mixed" else scope
        window = frames[index].clone()
        channel = torch.zeros(steps, 1, height, width, device=frames.device, dtype=frames.dtype)
        if sample_scope == "lens":
            window = apply_occluder(window, mask); channel[:] = mask
        else:
            window[-1] = apply_occluder(window[-1], mask); channel[-1] = mask
        out.append(torch.cat([window, channel], dim=1)); masks.append(mask)
    return torch.stack(out), torch.stack(masks)


def training_loss(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    l1 = F.l1_loss(prediction, target)
    if kind == "l1":
        return l1, {"l1": float(l1.detach())}
    mse = F.mse_loss(prediction.float(), target.float())
    return mse + 0.1 * l1, {"l1": float(l1.detach()), "mse": float(mse.detach())}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float().clamp(0, 1), target.float().clamp(0, 1)).item()
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))

def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = prediction.float().clamp(0, 1), target.float().clamp(0, 1)
    from pytorch_msssim import ssim as ssim_fn
    return float(ssim_fn(prediction, target, data_range=1.0, size_average=True).item())

@torch.no_grad()
def validate(model: VideoEENet, loader: DataLoader, device: torch.device, amp: bool, max_batches: int = 0, occlusion: Dict[str, object] | None = None) -> Dict[str, float]:
    model.eval(); losses, psnrs, ssims = [], [], []
    for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches and batch_index >= max_batches: break
        frames, target = batch["frames"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)
        if occlusion:
            frames, _ = occlude_batch(frames, batch["frame_path"], (0.0, 0.0), "lens", eval_coverage=occlusion["eval_coverage"])
        with torch.autocast(device_type=device.type, enabled=amp): prediction = model(frames); loss = F.l1_loss(prediction, target)
        losses.append(float(loss)); psnrs.append(psnr(prediction, target)); ssims.append(ssim(prediction, target))
    model.train()
    if not losses: raise RuntimeError("Validation loader is empty.")
    return {"loss": float(np.mean(losses)), "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}

def checkpoint_payload(model: VideoEENet, optimizer: AdamW, scheduler: CosineAnnealingLR, scaler: GradScaler, epoch: int, step: int, best: Dict[str, float], args: argparse.Namespace) -> Dict[str, object]:
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best, "config": vars(args)}

def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

def resolve_resume(value: str, checkpoint_dir: Path) -> Path | None:
    if not value: return None
    if value == "auto":
        path = checkpoint_dir / "last.pt"
        return path if path.is_file() else None
    path = Path(value)
    if not path.is_file(): raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VideoEENet on paired REVIDE sequences.")
    parser.add_argument("--data-root", required=True, help="RunPod root, e.g. /workspace/REVIDE_sequences")
    parser.add_argument("--train-split", default="Train"); parser.add_argument("--val-split", default="auto", help="auto reserves 10 percent of Train sequences, keeping Test untouched")
    parser.add_argument("--test-split", default="Test")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report-after-training", action="store_true")
    parser.add_argument("--output-dir", default="/workspace/videoeennet_runs")
    parser.add_argument("--seq-len", type=int, default=10); parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Augmentation: the original run had only a fixed 0.9x crop, and showed a 5.5 dB
    # val->test gap (26.28 -> 20.76), i.e. it memorised training scenes.
    parser.add_argument("--augment", action="store_true", help="scene-generalisation augmentation")
    parser.add_argument("--aug-scale-min", type=float, default=0.6)
    parser.add_argument("--aug-hflip", type=float, default=0.5)
    parser.add_argument("--aug-treverse", type=float, default=0.25)
    # Gamma jitter OFF by default. It is applied identically to hazy and clean so it is
    # SAFE, but REVIDE is a narrow indoor-lighting domain so colour-balance robustness is
    # not the failure mode here - and bundling it would confound attribution of the gain.
    parser.add_argument("--aug-gamma", type=float, default=0.0)
    parser.add_argument("--aug-vflip", type=float, default=0.0, help="vertical-flip probability (0 = off, the original setting)")
    parser.add_argument("--val-group", choices=["sequence", "scene"], default="sequence", help="scene holds out whole training scenes for validation")
    parser.add_argument("--base-channels", type=int, default=32); parser.add_argument("--hidden-dim", type=int, default=64, help="Use the same value for every ablation.")
    parser.add_argument("--temporal-mode", choices=["convlstm", "spatial_transformer", "hybrid", "single_frame"], default="convlstm")
    parser.add_argument("--attention-heads", type=int, default=4); parser.add_argument("--attention-pool-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--prefetch-factor", type=int, default=2); parser.add_argument("--preprocess", type=str, default="resize", choices=["resize", "crop"]); parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 evaluates all validation windows.")
    parser.add_argument("--resume", default="auto", help="'auto', an explicit .pt path, or empty string to start fresh.")
    parser.add_argument("--no-amp", action="store_true"); parser.add_argument("--seed", type=int, default=1234)
    # Experiments of 2026-09-25.
    parser.add_argument("--output-mode", choices=["sigmoid_sum", "logit_residual"], default="sigmoid_sum", help="logit_residual returns the input exactly when the residual is zero")
    parser.add_argument("--loss", choices=["l1", "mse_l1"], default="l1", help="mse_l1 = MSE + 0.1 * L1")
    parser.add_argument("--occlusion", action="store_true", help="train on windows partly hidden by an opaque occluder; adds a mask input channel")
    parser.add_argument("--occlusion-coverage-min", type=float, default=0.05); parser.add_argument("--occlusion-coverage-max", type=float, default=0.65)
    parser.add_argument("--occlusion-scope", choices=["lens", "current", "mixed"], default="mixed")
    parser.add_argument("--retrieval-index", default="", help="R1: retrieval index JSON (empty = preceding frames)")
    parser.add_argument("--align", choices=["none", "raft"], default="none", help="V10: warp earlier frames to the current one with frozen RAFT before encoding")
    parser.add_argument("--backbone", choices=["compact", "full"], default="compact", help="V11: 'full' = two dual-domain blocks per stage and multiscale skips from the current frame")
    args = parser.parse_args(); set_seed(args.seed)
    if min(args.epochs, args.batch_size, args.save_every, args.image_size) < 1 or args.seq_len < 2:
        parser.error("epochs, batch size, save interval and image size must be positive; seq-len must be >= 2")
    if args.train_split.lower() == args.val_split.lower():
        parser.error("Training and validation splits must differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); use_amp = device.type == "cuda" and not args.no_amp
    if args.require_cuda and device.type != 'cuda':
        raise RuntimeError('CUDA is unavailable; select a GPU pod and a CUDA-enabled PyTorch environment')
    output = Path(args.output_dir); checkpoint_dir, log_dir = output / "checkpoints", output / "logs"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    auto_val = args.val_split == 'auto'
    train_set = REVIDEVideoDataset(args.data_root, args.train_split, args.seq_len, args.image_size, random_crop=True, partition='train' if auto_val else None, split_seed=args.split_seed, augment=args.augment, aug_scale_min=args.aug_scale_min, aug_hflip=args.aug_hflip, aug_treverse=args.aug_treverse, aug_gamma=args.aug_gamma, preprocess=args.preprocess, aug_vflip=args.aug_vflip, val_group=args.val_group, retrieval_index=args.retrieval_index or None)
    val_set = REVIDEVideoDataset(args.data_root, args.train_split if auto_val else args.val_split, args.seq_len, args.image_size, random_crop=False, partition='val' if auto_val else None, split_seed=args.split_seed, preprocess=args.preprocess, val_group=args.val_group, retrieval_index=args.retrieval_index or None)
    train_paths = {str(p.resolve()) for s in train_set.sequences for pair in s['pairs'] for p in pair}
    val_paths = {str(p.resolve()) for s in val_set.sequences for pair in s['pairs'] for p in pair}
    if train_paths & val_paths:
        raise ValueError('Training and validation files overlap')
    loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0}
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_options)
    if not len(train_loader): raise RuntimeError("Training set is smaller than batch size; lower --batch-size.")
    model = VideoEENet(args.base_channels, args.hidden_dim, args.temporal_mode, args.attention_heads, args.attention_pool_size, args.seq_len, output_mode=args.output_mode, in_channels=4 if args.occlusion else 3, align=args.align, backbone=args.backbone).to(device)
    occlusion_val = {"eval_coverage": 0.5 * (args.occlusion_coverage_min + args.occlusion_coverage_max)} if args.occlusion else None
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader)); scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(log_dir)); start_epoch = 0; step = 0; best = {"psnr": -float("inf"), "ssim": -float("inf")}
    resume_path = resolve_resume(args.resume, checkpoint_dir)
    if resume_path:
        saved_config = torch.load(resume_path, map_location="cpu", weights_only=False)["config"]
        for key in ("epochs", "seq_len", "hidden_dim", "base_channels", "temporal_mode", "image_size", "batch_size", "seed", "data_root", "train_split", "val_split", "split_seed", "attention_heads", "attention_pool_size", "aug_vflip", "val_group", "augment", "output_mode", "loss", "occlusion", "occlusion_coverage_min", "occlusion_coverage_max", "occlusion_scope", "retrieval_index", "align", "backbone"):
            if saved_config.get(key) != getattr(args, key):
                raise ValueError(f"Resume configuration mismatch for {key}; use the original run configuration")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint.get("scaler", {})); start_epoch, step, best = int(checkpoint["epoch"]) + 1, int(checkpoint["step"]), checkpoint.get("best", best)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, step {step}.")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Device={device}, AMP={use_amp}, train_windows={len(train_set)}, val_windows={len(val_set)}, val_group={args.val_group}, val_groups={getattr(val_set, 'val_groups', None)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (all used by temporal_mode={args.temporal_mode})")
    for epoch in range(start_epoch, args.epochs):
        model.train(); epoch_losses, epoch_l1 = [], []
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            frames, target = batch["frames"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)
            if args.occlusion:
                frames, _ = occlude_batch(frames, batch["frame_path"], (args.occlusion_coverage_min, args.occlusion_coverage_max), args.occlusion_scope)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(frames); loss, loss_parts = training_loss(prediction, target, args.loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                step += 1
            loss_value = float(loss.detach()); epoch_losses.append(loss_value); epoch_l1.append(loss_parts["l1"]); writer.add_scalar("train/loss", loss_value, step); writer.add_scalar("train/l1_loss", loss_parts["l1"], step); writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        metrics = validate(model, val_loader, device, use_amp, args.max_val_batches, occlusion_val)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'epoch': epoch + 1, 'step': step, 'train_l1': float(np.mean(epoch_l1)), 'train_loss': float(np.mean(epoch_losses)), **metrics}) + '\n')
        writer.add_scalar("val/l1_loss", metrics["loss"], step); writer.add_scalar("val/psnr", metrics["psnr"], step); writer.add_scalar("val/ssim", metrics["ssim"], step)
        writer.add_images("train/current_hazy", frames[:, -1, :3].detach().float().cpu(), step); writer.add_images("train/prediction", prediction.detach().float().cpu(), step); writer.add_images("train/target", target.detach().float().cpu(), step)
        if metrics["psnr"] > best["psnr"]: best["psnr"] = metrics["psnr"]; save_checkpoint(checkpoint_dir / "best_psnr.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, step, best, args))
        if metrics["ssim"] > best["ssim"]: best["ssim"] = metrics["ssim"]; save_checkpoint(checkpoint_dir / "best_ssim.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, step, best, args))
        save_checkpoint(checkpoint_dir / "last.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, step, best, args))
        if (epoch + 1) % args.save_every == 0: save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:03d}.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, step, best, args))
        print(f"Epoch {epoch + 1}: train_loss={np.mean(epoch_losses):.4f}, train_l1={np.mean(epoch_l1):.4f}, val_psnr={metrics['psnr']:.2f}, val_ssim={metrics['ssim']:.4f}")
    writer.close(); print(f"Finished. Best PSNR={best['psnr']:.2f}, best SSIM={best['ssim']:.4f}")
    if args.report_after_training:
        del model, optimizer, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).with_name('report.py')), '--run-dir', str(output), '--data-root', args.data_root, '--test-split', args.test_split], check=True)
if __name__ == "__main__": main()
