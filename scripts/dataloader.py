"""REVIDE-compatible paired video sequence dataset for VideoEENet."""
from __future__ import annotations
import random
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HAZY_NAMES, CLEAN_NAMES = {"hazy", "input", "inputs", "fog", "degraded"}, {"gt", "clean", "clear", "target", "groundtruth", "ground_truth"}
def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

def _images(folder: Path) -> List[Path]:
    return sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
def _pair_paths(hazy_dir: Path, clean_dir: Path) -> List[Tuple[Path, Path]]:
    hazy, clean = _images(hazy_dir), _images(clean_dir)
    modalities = {'clean', 'clear', 'degraded', 'fog', 'ground', 'groundtruth', 'gt', 'hazy', 'input', 'target', 'truth'}
    def identity(path):
        return tuple(int(t) if t.isdigit() else t for t in re.findall(r'[a-z]+|\d+', path.stem.lower()) if t not in modalities)
    pairs = list(zip(hazy, clean))
    if len(hazy) != len(clean) or any(h.stem.lower() != c.stem.lower() and (not identity(h) or identity(h) != identity(c)) for h, c in pairs):
        raise ValueError(f"Frame names do not match: {hazy_dir} and {clean_dir}")
    return pairs
def discover_sequences(root: str | Path, split: Optional[str] = None) -> List[Dict[str, object]]:
    """Find `hazy`/`gt` or `hazy`/`clean` REVIDE sequence layouts."""
    root = Path(root)
    # A named split must stay isolated: otherwise a root-level scan can pull
    # Test windows into Train and silently invalidate validation metrics.
    bases = [root / split] if split else [root]
    if not bases[0].is_dir():
        raise FileNotFoundError(f"Dataset split not found: {bases[0]}")
    sequences: List[Dict[str, object]] = []; seen: set[Tuple[str, str]] = set()
    for base in bases:
        if not base.is_dir(): continue
        for directory in [base, *[path for path in base.rglob("*") if path.is_dir()]]:
            children = {path.name.lower(): path for path in directory.iterdir() if path.is_dir()}
            hazy_dirs, clean_dirs = [path for name, path in children.items() if name in HAZY_NAMES], [path for name, path in children.items() if name in CLEAN_NAMES]
            for hazy_dir in hazy_dirs:
                for clean_dir in clean_dirs:
                    hazy_children = {p.name for p in hazy_dir.iterdir() if p.is_dir()}
                    clean_children = {p.name for p in clean_dir.iterdir() if p.is_dir()}
                    if hazy_children != clean_children:
                        raise ValueError(f'Sequence folders differ: {hazy_dir} versus {clean_dir}')
                    pairs = _pair_paths(hazy_dir, clean_dir); key = (str(hazy_dir.resolve()), str(clean_dir.resolve()))
                    if pairs and key not in seen: seen.add(key); sequences.append({"name": directory.name, "pairs": pairs})
                    for hazy_seq in [path for path in hazy_dir.iterdir() if path.is_dir()]:
                        clean_seq = clean_dir / hazy_seq.name
                        pairs = _pair_paths(hazy_seq, clean_seq) if clean_seq.is_dir() else []
                        key = (str(hazy_seq.resolve()), str(clean_seq.resolve()))
                        if pairs and key not in seen: seen.add(key); sequences.append({"name": hazy_seq.name, "pairs": pairs})
    return sorted(sequences, key=lambda s: (s['name'], str(s['pairs'][0][0].relative_to(root))))
# Crop origin is fixed once per CLIP, never per frame: every frame in a window must share the
# same spatial window, otherwise the temporal model sees a moving viewport instead of a moving
# scene and the ConvLSTM is asked to model camera motion that is not there.
_CROP_ORIGIN = {"top": None, "left": None, "random": False}


def _crop_origin(H: int, W: int, crop: int):
    if _CROP_ORIGIN["top"] is not None:
        return _CROP_ORIGIN["top"], _CROP_ORIGIN["left"]
    if _CROP_ORIGIN["random"]:
        return random.randint(0, max(0, H - crop)), random.randint(0, max(0, W - crop))
    return (H - crop) // 2, (W - crop) // 2


def _load_rgb(path: Path, image_size: int, preprocess: str = "resize") -> torch.Tensor:
    """preprocess="resize": squash the whole frame to image_size^2 (original behaviour).
       preprocess="crop"  : take an image_size^2 crop at NATIVE resolution, which is what TRDN
                            does. REVIDE frames are 2708x1800, so resizing is a ~10x downscale
                            that averages away sensor noise and fine haze structure - measured
                            to be worth 2.67 dB, so the two models were never solving the same
                            task. Train crops are random (spatial diversity), eval crops centre
                            (matching TRDN's random_crop=False)."""
    img = Image.open(path).convert("RGB")
    if preprocess == "crop":
        W, H = img.size
        crop = min(image_size, H, W)
        top, left = _crop_origin(H, W, crop)
        img = img.crop((left, top, left + crop, top + crop))
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return F.interpolate(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]

class REVIDEVideoDataset(Dataset):
    """Returns `frames` [T, 3, H, W] plus the clean final frame target."""
    def __init__(self, root: str, split: Optional[str], seq_len: int = 10, image_size: int = 256, random_crop: bool = False, partition: Optional[str] = None, split_seed: int = 1234, augment: bool = False, aug_scale_min: float = 0.6, aug_hflip: float = 0.5, aug_treverse: float = 0.25, aug_gamma: float = 0.3, preprocess: str = "resize", aug_vflip: float = 0.0, val_group: str = "sequence") -> None:
        self.root, self.split, self.seq_len, self.image_size, self.random_crop = Path(root), split, seq_len, image_size, random_crop
        self.preprocess = preprocess
        # Augmentation for scene generalisation. Val/test construct with augment=False.
        self.augment, self.aug_scale_min = augment, aug_scale_min
        self.aug_hflip, self.aug_treverse, self.aug_gamma = aug_hflip, aug_treverse, aug_gamma
        self.aug_vflip = aug_vflip
        self.sequences = discover_sequences(self.root, split); self.index: List[Tuple[int, int]] = []
        if partition is not None:
            if partition not in {'train', 'val'}:
                raise ValueError('partition must be train or val')
            # val_group="sequence" (original): hold out 10% of the 10-frame sequence folders. These
            # are consecutive chunks of the same videos used for training, so validation frames
            # are near-duplicates of training frames. val_group="scene": hold out whole scenes
            # (the code before the first underscore, e.g. C002 covers C002_1..C002_4), so no
            # validation scene is ever seen in training, as on the test set.
            if val_group not in {'sequence', 'scene'}:
                raise ValueError('val_group must be sequence or scene')
            group_of = (lambda name: name.split('_')[0]) if val_group == 'scene' else (lambda name: name)
            names = sorted({group_of(str(s['name'])) for s in self.sequences})
            if len(names) < 2:
                raise ValueError('At least two training groups are needed for a separate validation set')
            ranked = sorted(names, key=lambda name: int.from_bytes(hashlib.sha256(f'{split_seed}:{name}'.encode()).digest()[:8], 'big'))
            val_names = set(ranked[:min(len(names) - 1, max(1, round(len(names) * 0.1)))])
            self.val_groups = sorted(val_names)
            self.sequences = [s for s in self.sequences if (group_of(str(s['name'])) in val_names) == (partition == 'val')]
        for sequence_index, sequence in enumerate(self.sequences): self.index.extend((sequence_index, end) for end in range(seq_len - 1, len(sequence["pairs"])))
        if not self.index: raise FileNotFoundError(f"No paired {seq_len}-frame sequences under {self.root}. Expected paired hazy/gt or hazy/clean folders.")
    def __len__(self) -> int: return len(self.index)
    def __getitem__(self, index: int) -> Dict[str, object]:
        sequence_index, end_index = self.index[index]; sequence = self.sequences[sequence_index]; pairs: Sequence[Tuple[Path, Path]] = sequence["pairs"]
        clip = pairs[end_index - self.seq_len + 1 : end_index + 1]
        if self.preprocess == "crop":
            # one origin for the whole clip; random while training, centre otherwise
            _CROP_ORIGIN["random"] = bool(self.augment or self.random_crop)
            _CROP_ORIGIN["top"] = _CROP_ORIGIN["left"] = None
            with Image.open(clip[0][0]) as _probe:
                _W, _H = _probe.size
            _c = min(self.image_size, _H, _W)
            _CROP_ORIGIN["top"], _CROP_ORIGIN["left"] = _crop_origin(_H, _W, _c)
        hazy = torch.stack([_load_rgb(path, self.image_size, self.preprocess) for path, _ in clip])
        clean = torch.stack([_load_rgb(path, self.image_size, self.preprocess) for _, path in clip])
        _CROP_ORIGIN["top"] = _CROP_ORIGIN["left"] = None
        if self.augment and self.image_size >= 64:
            # --- variable-scale random crop (was a fixed 0.9x, giving almost no spatial diversity)
            scale = random.uniform(self.aug_scale_min, 1.0)
            crop = max(32, int(self.image_size * scale))
            top = random.randint(0, self.image_size - crop)
            left = random.randint(0, self.image_size - crop)
            hazy = hazy[:, :, top:top + crop, left:left + crop]
            clean = clean[:, :, top:top + crop, left:left + crop]
            hazy = F.interpolate(hazy, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            clean = F.interpolate(clean, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            # --- horizontal flip. Identical transform on hazy AND clean, or the pairing breaks.
            if random.random() < self.aug_hflip:
                hazy, clean = torch.flip(hazy, dims=[3]), torch.flip(clean, dims=[3])
            # --- vertical flip (off by default; tested as its own experiment). Real haze gets
            #     denser with depth, which in these scenes often tracks image height, so a flipped
            #     frame can show a haze gradient that does not occur in the test data.
            if self.aug_vflip > 0 and random.random() < self.aug_vflip:
                hazy, clean = torch.flip(hazy, dims=[2]), torch.flip(clean, dims=[2])

            # --- temporal reversal: the clip played backwards is still plausible motion, and the
            #     target stays clean[-1] so the task is unchanged. Gives the ConvLSTM new dynamics.
            if random.random() < self.aug_treverse:
                hazy, clean = torch.flip(hazy, dims=[0]), torch.flip(clean, dims=[0])

            # --- gamma jitter, applied IDENTICALLY to hazy and clean so the transmission
            #     relationship between them is preserved. Kept mild: haze/atmospheric-light
            #     estimation depends on absolute colour statistics, so heavy colour jitter
            #     would teach the model a physically wrong prior.
            if random.random() < self.aug_gamma:
                g = random.uniform(0.85, 1.18)
                hazy, clean = hazy.clamp(0, 1).pow(g), clean.clamp(0, 1).pow(g)
        elif self.random_crop and self.image_size >= 64:
            crop = int(self.image_size * 0.9); top, left = random.randint(0, self.image_size-crop), random.randint(0, self.image_size-crop)
            hazy = F.interpolate(hazy[:, :, top:top+crop, left:left+crop], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False); clean = F.interpolate(clean[:, :, top:top+crop, left:left+crop], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return {"frames": hazy, "target": clean[-1], "clean_frames": clean, "sequence_name": str(sequence["name"]), "frame_path": str(clip[-1][0])}
