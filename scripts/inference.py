"""Sequence-aware VideoEENet inference for image frames stored in one folder."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from model import VideoEENet
from dataloader import natural_key
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
def read_image(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)
def main() -> None:
    parser = argparse.ArgumentParser(description="Dehaze a chronological folder of video frames.")
    parser.add_argument("--input-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--seq-len", type=int, default=10); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); config = checkpoint.get("config", {})
    model = VideoEENet(int(config.get("base_channels", 32)), int(config.get("hidden_dim", 64)), config.get("temporal_mode", "convlstm"), int(config.get("attention_heads", 4)), int(config.get("attention_pool_size", 8)), args.seq_len).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    inputs = sorted((path for path in Path(args.input_dir).iterdir() if path.is_file() and path.suffix.lower() in SUFFIXES), key=natural_key)
    if not inputs: raise FileNotFoundError(f"No supported image frames in {args.input_dir}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); history = []
    for path in tqdm(inputs, desc="VideoEENet inference"):
        history.append(read_image(path)); history = history[-args.seq_len:]; frames = history if len(history) == args.seq_len else [history[0]] * (args.seq_len - len(history)) + history
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(torch.stack(frames).unsqueeze(0).to(device))[0].float().clamp(0, 1)
        Image.fromarray((prediction.cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)).save(output / path.name)
    print(f"Saved {len(inputs)} dehazed frames to {output}")
if __name__ == "__main__": main()
