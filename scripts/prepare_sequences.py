"""Inspect a REVIDE-style directory and create a reproducible sequence manifest."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from dataloader import discover_sequences
def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest of valid paired video windows."); parser.add_argument("--data-root", required=True); parser.add_argument("--split", default="Train"); parser.add_argument("--seq-len", type=int, default=10); parser.add_argument("--output", default="sequence_manifest.csv"); args = parser.parse_args()
    sequences = discover_sequences(args.data_root, args.split); rows = [[sequence["name"], str(pairs[end][0]), str(pairs[end][1])] for sequence in sequences for pairs in [sequence["pairs"]] for end in range(args.seq_len - 1, len(pairs))]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle: writer = csv.writer(handle); writer.writerow(["sequence", "hazy_target", "clean_target"]); writer.writerows(rows)
    print(f"Found {len(sequences)} paired sequences and {len(rows)} valid {args.seq_len}-frame windows. Manifest: {output.resolve()}")
if __name__ == "__main__": main()
