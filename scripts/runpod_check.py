"""Verify CUDA, complete paired image inventory, and non-overlapping splits."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from dataloader import REVIDEVideoDataset


def check(root, output, seq_len=10, size=256, seed=1234, require_cuda=True):
    if require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    report = {'cuda': torch.cuda.is_available(), 'dataset_root': str(Path(root).resolve()), 'splits': {}}
    seen = {}
    for label, split, partition in [('train', 'Train', 'train'), ('val', 'Train', 'val'), ('test', 'Test', None)]:
        dataset = REVIDEVideoDataset(str(root), split, seq_len, size, partition=partition, split_seed=seed)
        files = set()
        for sequence in dataset.sequences:
            dimensions = set()
            for hazy, clean in sequence['pairs']:
                for path in (hazy, clean):
                    with Image.open(path) as image:
                        dimensions.add(image.size)
                        image.verify()
                    files.add(str(path.resolve()))
            if len(dimensions) != 1: raise ValueError(f"Inconsistent dimensions in {sequence['name']}")
        for prior, paths in seen.items():
            if files & paths: raise ValueError(f'{label} overlaps {prior}')
        seen[label] = files
        sample = dataset[0]
        report['splits'][label] = {'sequences': [s['name'] for s in dataset.sequences], 'windows': len(dataset), 'files': len(files), 'frames_shape': list(sample['frames'].shape), 'target_shape': list(sample['target'].shape)}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    report['status'] = 'PASS'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--split-seed', type=int, default=1234)
    args = parser.parse_args()
    print(json.dumps(check(args.data_root, args.output, args.seq_len, args.image_size, args.split_seed), indent=2))
