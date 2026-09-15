#!/usr/bin/env python3
"""Create a manifest for video-frame dehazing datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_frames(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CSV manifest of video frames.")
    parser.add_argument("root", type=Path, help="Root directory containing frame images.")
    parser.add_argument("output", type=Path, help="Output CSV path.")
    args = parser.parse_args()

    frames = find_frames(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame_index", "path"])
        for frame in frames:
            sequence = frame.parent.name
            digits = "".join(ch for ch in frame.stem if ch.isdigit())
            frame_index = int(digits) if digits else ""
            writer.writerow([sequence, frame_index, frame.relative_to(args.root).as_posix()])

    print(f"Wrote {len(frames)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())