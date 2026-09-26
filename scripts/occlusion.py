"""Opaque occluders for the partial-occlusion experiment (VideoEENet side).

A verbatim port of TRDN's src/masks_variable.py shape generators and src/occlusion.py, so that
both models are evaluated on pixel-identical occluders: the same key and coverage give the same
mask in either repository (tests/test_occlusion.py checks the digest of a reference mask).

"lens" scope covers every frame of the window (dirt on the lens); "current" covers only the
final frame (a transient occluder). Occluded pixels are set to mid-grey and, when the model is
built with in_channels=4, a fourth input channel carries the mask.
"""
from __future__ import annotations

import hashlib
import math
import random

import cv2
import numpy as np
import torch


def _as_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask).unsqueeze(0).float().clamp(0, 1)


def variable_rectangle(h: int, w: int, target: float) -> np.ndarray:
    """Rectangle whose AREA is `target` of the frame, with random aspect ratio."""
    area = target * h * w
    aspect = random.uniform(0.45, 2.2)                 # w/h
    rh = min(h, max(1, int(round(math.sqrt(area / aspect)))))
    rw = min(w, max(1, int(round(area / rh))))
    top = random.randint(0, max(0, h - rh))
    left = random.randint(0, max(0, w - rw))
    m = np.zeros((h, w), dtype=np.float32)
    m[top:top + rh, left:left + rw] = 1.0
    return m


def variable_ellipse(h: int, w: int, target: float) -> np.ndarray:
    """Ellipse with area = target*h*w  (pi*a*b = area)."""
    area = target * h * w
    aspect = random.uniform(0.55, 1.8)
    # cv2.ellipse takes SEMI-axes, so solve area = pi * sa * sb directly for them.
    # (Computing full axes and halving them quarters the area - that bug made
    #  ellipse top out at ~20% coverage when 85% was requested.)
    sb = math.sqrt(area / (math.pi * aspect))
    sa = aspect * sb
    sa = int(max(2, min(w * 0.75, sa)))
    sb = int(max(2, min(h * 0.75, sb)))
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    m = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(m, (cx, cy), (sa, sb), random.uniform(0, 180), 0, 360, 1.0, -1)
    return m


def variable_blob(h: int, w: int, target: float) -> np.ndarray:
    """Accumulate overlapping circles until measured coverage reaches target."""
    m = np.zeros((h, w), dtype=np.float32)
    base = max(4, int(min(h, w) * random.uniform(0.06, 0.16)))
    for _ in range(400):
        if m.mean() >= target:
            break
        cv2.circle(m, (random.randint(0, w - 1), random.randint(0, h - 1)),
                   random.randint(base, int(base * 1.9)), 1.0, -1)
    return m


def variable_perlin(h: int, w: int, target: float) -> np.ndarray:
    """Smooth noise field thresholded (bisection) to hit the target coverage."""
    grid = random.choice([4, 6, 8, 12])
    noise = np.random.rand(grid, grid).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = (noise - noise.min()) / max(1e-6, noise.max() - noise.min())
    lo, hi = 0.0, 1.0
    for _ in range(24):                       # bisect the threshold
        mid = (lo + hi) / 2
        if (noise >= mid).mean() > target:
            lo = mid
        else:
            hi = mid
    return (noise >= (lo + hi) / 2).astype(np.float32)


_SHAPES = {"rectangle": variable_rectangle, "ellipse": variable_ellipse,
           "blob": variable_blob, "perlin": variable_perlin}


def variable_occlusion_mask(height: int, width: int, mode: str = "mixed",
                            coverage_min: float = 0.05,
                            coverage_max: float = 0.85) -> torch.Tensor:
    """Mask of shape [1,H,W]; occluded fraction ~ U(coverage_min, coverage_max).

    Sampling coverage uniformly means the model sees small patches AND
    near-total occlusion within one run, so it cannot specialise to one size.
    """
    target = random.uniform(coverage_min, coverage_max)
    shape = random.choice(list(_SHAPES)) if mode in ("mixed", "auto") else mode
    if shape not in _SHAPES:
        raise ValueError(f"Unknown variable mask mode: {mode}")
    return _as_tensor(_SHAPES[shape](height, width, target))

# Rectangles are excluded: a lens obstruction is irregular, not axis-aligned.
OCCLUDER_SHAPES = ("ellipse", "blob", "perlin")
OCCLUDER_FILL = 0.5
MAX_DRAWS = 30


def occluder_mask(height: int, width: int, coverage: float, shape: str | None = None) -> torch.Tensor:
    """[1, H, W] opaque-occluder mask with the requested fraction of pixels covered."""
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1); got {coverage}")
    shape = shape or random.choice(OCCLUDER_SHAPES)
    if shape not in OCCLUDER_SHAPES:
        raise ValueError(f"Unknown occluder shape: {shape}")
    # Shapes placed near the border are clipped and cover less than requested, so redraw
    # until the measured coverage is within 10 % (relative) of the target, keeping the closest.
    best, best_gap = None, float("inf")
    for _ in range(MAX_DRAWS):
        candidate = _SHAPES[shape](height, width, coverage)
        gap = abs(float(candidate.mean()) - coverage)
        if gap < best_gap:
            best, best_gap = candidate, gap
        if gap <= 0.10 * coverage:
            break
    return _as_tensor(best)


def seeded_occluder_mask(height: int, width: int, coverage: float, key: str) -> torch.Tensor:
    """Deterministic occluder for evaluation: the same key and coverage give the same mask."""
    seed = int.from_bytes(hashlib.sha256(f"{key}|{coverage:.4f}".encode()).digest()[:8], "little")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        return occluder_mask(height, width, coverage)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def occluder_key(target_frame_path: str) -> str:
    """Evaluation key shared by both models: the last three components of the resolved hazy
    target-frame path (REVIDE_TRDN symlinks to REVIDE_sequences, so both resolve alike)."""
    import os
    from pathlib import Path

    return "/".join(Path(os.path.realpath(target_frame_path)).parts[-3:])


def apply_occluder(frames: torch.Tensor, mask: torch.Tensor, fill: float = OCCLUDER_FILL) -> torch.Tensor:
    """Replace occluded pixels of [..., 3, H, W] frames with `fill`; mask is [1, H, W]."""
    return frames * (1.0 - mask) + fill * mask
