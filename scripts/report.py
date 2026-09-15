"""Full-test evaluation and audited visual artifacts from a saved VideoEENet run."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from dataloader import REVIDEVideoDataset
from model import VideoEENet
from train import psnr, ssim


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def rgb(tensor):
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def build_report(run_dir, data_root, test_split='Test', device_name=None):
    run_dir = Path(run_dir)
    output = run_dir / 'artifacts'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'figure_checklist.json', {'status': 'RUNNING'})
    checkpoint_path = run_dir / 'checkpoints' / 'best_psnr.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if test_split.lower() in {config['train_split'].lower(), config['val_split'].lower()}:
        raise ValueError('Final test split was used for training or checkpoint selection')
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = REVIDEVideoDataset(str(data_root), test_split, config['seq_len'], config['image_size'])
    dataset_paths = {str(p.resolve()) for s in dataset.sequences for pair in s['pairs'] for p in pair}
    for split in {config['train_split'], config['val_split']} - {'auto'}:
        used = REVIDEVideoDataset(str(data_root), split, config['seq_len'], config['image_size'])
        if dataset_paths & {str(p.resolve()) for s in used.sequences for pair in s['pairs'] for p in pair}:
            raise ValueError('Test files overlap training or validation files')
    model = VideoEENet(config['base_channels'], config['hidden_dim'], config['temporal_mode'], config['attention_heads'], config['attention_pool_size'], config['seq_len']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    selected = set(np.linspace(0, len(dataset) - 1, min(4, len(dataset)), dtype=int).tolist())
    rows, samples, worst, animation = [], [], [], []
    previous = None
    feature = {}
    hook_module = model.temporal if config['temporal_mode'] == 'convlstm' else model.attention
    hook = hook_module.register_forward_hook(lambda module, inputs, result: feature.update(memory=result.detach().float().cpu()))
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc='Full test and artifacts'):
            sample = dataset[index]
            frames = sample['frames'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                prediction = model(frames).float()
            if device.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = {'index': index, 'sequence': sample['sequence_name'], 'frame': sample['frame_path'], 'psnr': psnr(prediction, target), 'ssim': ssim(prediction, target), 'input_psnr': psnr(frames[:, -1], target), 'seconds': elapsed, 'ifd': None, 'target_ifd': None, 'temporal_delta_l1': None}
            pred_cpu, gt_cpu = prediction[0].cpu(), target[0].cpu()
            seq_id, end = dataset.index[index]
            if previous is not None and previous[0] == seq_id and previous[1] + 1 == end:
                pd, gd = pred_cpu - previous[2], gt_cpu - previous[3]
                row.update(ifd=float(pd.abs().mean()), target_ifd=float(gd.abs().mean()), temporal_delta_l1=float((pd - gd).abs().mean()))
            previous = (seq_id, end, pred_cpu, gt_cpu)
            rows.append(row)
            item = (index, sample, pred_cpu, feature['memory'][0], row)
            if index in selected: samples.append(item)
            worst.append(item)
            worst = sorted(worst, key=lambda value: value[-1]['psnr'])[:3]
            if seq_id == 0 and len(animation) < 24:
                panel = np.concatenate([rgb(sample['frames'][-1]), rgb(pred_cpu), rgb(sample['target'])], axis=1)
                animation.append(Image.fromarray((panel * 255).round().astype('uint8')))
    hook.remove()
    with (output / 'per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {key: float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None for key in ('psnr', 'ssim', 'input_psnr', 'seconds', 'ifd', 'target_ifd', 'temporal_delta_l1')}
    write_json(output / 'metrics.json', {'checkpoint': str(checkpoint_path), 'checkpoint_step': checkpoint['step'], 'config': config, 'frames_evaluated': len(rows), 'frames_available': len(dataset), 'aggregation': 'mean per frame', 'aggregate': aggregate, 'selected_indices': sorted(selected), 'temporal_definition': 'Unwarped consecutive prediction and GT differences; IFD must be read with PSNR. Not a flow-warped metric.'})
    generated = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            path = output / f'{name}.{ext}'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated.append(path)
        plt.close(fig)
    def comparison(items, name, zoom=False):
        fig, axes = plt.subplots(len(items), 4, figsize=(12, 3 * len(items)), squeeze=False)
        for axis, (index, sample, prediction, _, row) in zip(axes, items):
            images = [rgb(sample['frames'][-1]), rgb(sample['target']), rgb(prediction)]
            error = np.abs(images[2] - images[1]).mean(axis=2)
            for col, data in enumerate([*images, error]):
                if zoom:
                    h, w = data.shape[:2]; data = data[h//3:2*h//3, w//3:2*w//3]
                axis[col].imshow(data, cmap='magma' if col == 3 else None, vmin=0, vmax=1)
                axis[col].set_title(['Input', 'Target', f"Prediction {row['psnr']:.2f} dB", 'Mean absolute error [0,1]'][col])
                axis[col].axis('off')
            axis[0].set_ylabel(f'Sample {index}')
        save(fig, name)
    comparison(samples, 'qualitative_and_error_maps')
    comparison(samples, 'zoomed_crops', zoom=True)
    comparison(worst, 'lowest_psnr_cases')
    sample = samples[0][1]
    fig, axes = plt.subplots(2, config['seq_len'], figsize=(2 * config['seq_len'], 4), squeeze=False)
    for t in range(config['seq_len']):
        for r, key in enumerate(('frames', 'clean_frames')):
            axes[r, t].imshow(rgb(sample[key][t])); axes[r, t].axis('off'); axes[r, t].set_title(f'{key} t={t}')
    save(fig, 'input_target_sequence')
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4), squeeze=False)
    for axis, item in zip(axes[0], samples):
        axis.imshow(item[3].square().mean(0).sqrt(), cmap='viridis'); axis.set_title(f'Memory RMS: sample {item[0]}'); axis.axis('off')
    save(fig, 'temporal_feature_activation')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for axis, key in zip(axes, ('psnr', 'ssim', 'seconds')):
        axis.hist([r[key] for r in rows], bins=min(20, len(rows))); axis.set_xlabel(key); axis.set_ylabel('Frames')
    save(fig, 'metric_distributions')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot([r['psnr'] for r in rows], label='Prediction'); axes[0].plot([r['input_psnr'] for r in rows], label='Input'); axes[0].set_ylabel('PSNR (dB)'); axes[0].legend()
    for key in ('ifd', 'target_ifd', 'temporal_delta_l1'):
        axes[1].plot([np.nan if r[key] is None else r[key] for r in rows], label=key)
    axes[1].set_xlabel('Test window index'); axes[1].set_ylabel('Mean absolute difference'); axes[1].legend()
    save(fig, 'temporal_diagnostics')
    history_path = run_dir / 'history.jsonl'
    if not history_path.is_file(): raise FileNotFoundError('Training curves require history.jsonl from the updated trainer')
    history = {r['epoch']: r for r in (json.loads(line) for line in history_path.read_text().splitlines() if line.strip())}
    history = [history[e] for e in sorted(history)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for axis, keys in zip(axes, (('train_l1', 'loss'), ('psnr',), ('ssim',))):
        for key in keys: axis.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
        axis.set_xlabel('Epoch'); axis.legend()
    save(fig, 'training_curves')
    if len(animation) < 2: raise ValueError('Temporal animation requires at least two windows in the first test sequence')
    animation[0].save(output / 'temporal_comparison.gif', save_all=True, append_images=animation[1:], duration=150, loop=0)
    generated.append(output / 'temporal_comparison.gif')
    page = '<h1>VideoEENet test report</h1><p>Input | Target | Prediction in static grids. GIF: Input | Prediction | Target.</p>'
    page += '<p>Memory RMS is a feature activation, not an attention-weight map. This model has no RAFT or reference selector.</p>'
    page += ''.join(f'<h2>{html.escape(p.stem)}</h2><img style="max-width:100%" src="{p.name}">' for p in generated if p.suffix in {'.png', '.gif'})
    (output / 'index.html').write_text(page, encoding='utf-8')
    generated += [output / name for name in ('per_frame.csv', 'metrics.json', 'index.html')]
    inventory = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in generated}
    if not all(item['bytes'] for item in inventory.values()): raise RuntimeError('Empty report artifact')
    write_json(output / 'figure_checklist.json', {'status': 'PASS', 'files': inventory, 'not_applicable': ['RAFT flow maps', 'reference selection weights', 'VAE ceiling'], 'attention_note': 'Feature activation maps are provided; raw attention probabilities are not exported.'})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--test-split', default='Test')
    args = parser.parse_args()
    print(build_report(args.run_dir, args.data_root, args.test_split))
