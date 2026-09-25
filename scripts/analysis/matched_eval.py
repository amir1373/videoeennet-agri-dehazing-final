"""Evaluate VideoEENet under TRDN's EXACT test protocol: 256x256 CENTER CROP at native
resolution (frames are 2708x1800), instead of resizing the whole frame down to 256x256.

WHY THIS MATTERS: the two models have never been compared on the same task.
  - TRDN   : center-crops 256x256 from the 2708x1800 frame -> ~1.3% of the frame, full detail.
  - VideoEENet: bilinearly resizes the ENTIRE 2708x1800 frame to 256x256 -> ~10x downscale.
Downscaling by 10x averages away sensor noise and fine haze structure, which inflates PSNR.
Until both are scored the same way, no TRDN-vs-VideoEENet claim is defensible.

Matches src/dataset.py of TRDN: random_crop=False at eval -> top=(H-crop)//2, left=(W-crop)//2.
Metrics are VideoEENet's own psnr/ssim so its numbers stay comparable to its other results.
Emits the same per_clip schema the bootstrap CI script consumes.
"""
import json, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import VideoEENet, load_model_state
from pytorch_msssim import ssim as ssim_fn
from skimage.metrics import structural_similarity
import lpips as lpips_lib

CKPT = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "crop"     # "crop" (TRDN protocol) or "resize"
ROOT = os.environ.get("REVIDE_SEQUENCES", "/workspace/datasets/REVIDE_sequences")
SEQ_LEN = 10
SIZE = 256
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
model = VideoEENet(cfg.get("base_channels", 32), cfg.get("hidden_dim", 64),
                   cfg.get("temporal_mode", "convlstm"), cfg.get("attention_heads", 4),
                   cfg.get("attention_pool_size", 8), cfg.get("seq_len", SEQ_LEN)).to(dev)
dropped = load_model_state(model, ck["model"])
if dropped:
    print(f"dropped {len(dropped)} unused legacy tensors (attention/hybrid_fuse)", flush=True)
model.eval()
# TRDN's exact metrics (src/metrics.py, src/losses.py): SSIM on uint8 HWC via scikit-image,
# LPIPS-Alex on [-1, 1]. The pytorch_msssim SSIM used by earlier reports is kept as ssim_msssim.
lpips_model = lpips_lib.LPIPS(net="alex").to(dev).eval()

def to_uint8_hwc(x):
    return (x[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

def trdn_ssim(p, t):
    return float(structural_similarity(to_uint8_hwc(p), to_uint8_hwc(t), channel_axis=2, data_range=255))

def trdn_lpips(p, t):
    with torch.no_grad():
        return float(lpips_model(p.clamp(0, 1) * 2 - 1, t.clamp(0, 1) * 2 - 1).mean())
print(f"loaded {CKPT} (epoch {ck.get('epoch')}, step {ck.get('step')}), mode={MODE}", flush=True)

def load_frame(path):
    """MODE=crop  -> center 256x256 at NATIVE resolution (TRDN's protocol).
       MODE=resize-> whole frame squashed to 256x256 (VideoEENet's original protocol)."""
    im = Image.open(path).convert("RGB")
    if MODE == "resize":
        t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
    W, H = im.size
    crop = min(SIZE, H, W)
    top, left = (H - crop) // 2, (W - crop) // 2
    im = im.crop((left, top, left + crop, top + crop))
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

def psnr(p, t):
    mse = F.mse_loss(p.float().clamp(0, 1), t.float().clamp(0, 1)).item()
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))

scene_dirs = {}
for d in sorted(glob.glob(f"{ROOT}/Test/hazy/*")):
    if os.path.isdir(d):
        scene_dirs.setdefault(os.path.basename(d).split("_seq_")[0], []).append(d)

per_clip = []
for scene, dirs in sorted(scene_dirs.items()):
    ps, ss, ss_ms, lp, nf = [], [], [], [], 0
    for hd in sorted(dirs):
        gd = hd.replace("/hazy/", "/gt/")
        if not os.path.isdir(gd):
            continue
        files = sorted(f for f in os.listdir(hd) if f.lower().endswith(EXTS)
                       and os.path.isfile(os.path.join(gd, f)))
        # dataloader.py indexes windows as range(seq_len-1, len(sequence)): a sliding window
        # ending at every frame, and the model predicts ONLY that last frame (target=clean[-1]).
        if len(files) < SEQ_LEN:
            continue
        hazy_all = torch.stack([load_frame(os.path.join(hd, f)) for f in files])
        gt_all   = torch.stack([load_frame(os.path.join(gd, f)) for f in files])
        for end in range(SEQ_LEN - 1, len(files)):
            hz = hazy_all[end - SEQ_LEN + 1:end + 1].unsqueeze(0).to(dev)
            tgt = gt_all[end].unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = model(hz).clamp(0, 1)
            ps.append(psnr(pred, tgt))
            ss.append(trdn_ssim(pred, tgt))
            ss_ms.append(float(ssim_fn(pred, tgt, data_range=1.0, size_average=True).item()))
            lp.append(trdn_lpips(pred, tgt))
            nf += 1
    if ps:
        per_clip.append({"sequence_name": f"{scene}_run_000", "num_frames": nf,
                         "psnr_mean": float(np.mean(ps)), "ssim_mean": float(np.mean(ss)),
                         "ssim_msssim_mean": float(np.mean(ss_ms)),
                         "lpips_mean": float(np.mean(lp)), "n_windows": len(ps),
                         "psnr_per_window": ps})
        print(f"  {scene:<8} psnr {np.mean(ps):7.4f}  ssim {np.mean(ss):.4f}  lpips {np.mean(lp):.4f}  "
              f"windows {len(ps):3d}  frames {nf}", flush=True)

tot = sum(c["num_frames"] for c in per_clip) or 1
agg = {"psnr": {"mean": sum(c["psnr_mean"] * c["num_frames"] for c in per_clip) / tot},
       "ssim": {"mean": sum(c["ssim_mean"] * c["num_frames"] for c in per_clip) / tot},
       "ssim_msssim": {"mean": sum(c["ssim_msssim_mean"] * c["num_frames"] for c in per_clip) / tot},
       "lpips": {"mean": sum(c["lpips_mean"] * c["num_frames"] for c in per_clip) / tot}}
json.dump({"per_clip": per_clip, "aggregate": agg, "preprocess_mode": MODE,
           "ssim_definition": "scikit-image SSIM on uint8, as TRDN; ssim_msssim = pytorch_msssim",
           "epoch": ck.get("epoch"), "step": ck.get("step"), "config": cfg,
           "checkpoint": CKPT, "crop_size": SIZE, "seq_len": SEQ_LEN,
           "note": "MODE=crop reproduces TRDN's center-crop-256-at-native-resolution protocol."},
          open(OUT, "w"), indent=2)
print(f"\nAGGREGATE  psnr {agg['psnr']['mean']:.4f}  ssim {agg['ssim']['mean']:.4f}  lpips {agg['lpips']['mean']:.4f}")
print("wrote " + OUT)
