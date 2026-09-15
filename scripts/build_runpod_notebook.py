"""Generate the repository-backed RunPod execution notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def section(title, source):
    cells.extend([nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(source)])

cells.append(nbf.v4.new_markdown_cell('# VideoEENet GPU training and complete visual reports\nRun on RunPod. Core computation lives in scripts/. Training and evaluation consume real paired images. Run the data check before starting a job.'))
section('## 1. Repository and installation', '''from pathlib import Path
import os, subprocess, sys
REPO_ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'scripts/report.py').is_file()), Path('/workspace/videoeennet-agri-dehazing-final'))
if not REPO_ROOT.is_dir():
    subprocess.run(['git', 'clone', 'https://github.com/amir1373/videoeennet-agri-dehazing-final.git', str(REPO_ROOT)], check=True)
os.chdir(REPO_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)''')
section('## 2. Paths and experiment settings', '''DATASET_ROOT = Path('/workspace/datasets/REVIDE')
RUN_ROOT = Path('/workspace/videoeennet_runs/convlstm_seed1234')
SEQ_LEN, IMAGE_SIZE, EPOCHS, BATCH_SIZE = 10, 256, 30, 1
SEED, SPLIT_SEED, HIDDEN_DIM = 1234, 1234, 64
MODE = 'convlstm'  # convlstm, spatial_transformer, hybrid
RUN_TRAINING = False
RUN_REPORT_ONLY = False
RUN_ROOT.mkdir(parents=True, exist_ok=True)
def command(script, *args):
    return [sys.executable, str(REPO_ROOT / 'scripts' / script), *map(str, args)]''')
section('## 3. GPU and complete dataset check', '''subprocess.run(command('runpod_check.py', '--data-root', DATASET_ROOT, '--output', RUN_ROOT / 'dataset_check.json', '--seq-len', SEQ_LEN, '--image-size', IMAGE_SIZE, '--split-seed', SPLIT_SEED), check=True)''')
section('## 4. Inspect real data', '''sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from dataloader import REVIDEVideoDataset
import matplotlib.pyplot as plt
dataset = REVIDEVideoDataset(str(DATASET_ROOT), 'Train', SEQ_LEN, IMAGE_SIZE, partition='train', split_seed=SPLIT_SEED)
sample = dataset[0]
print({key: tuple(value.shape) for key, value in sample.items() if hasattr(value, 'shape')})
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, tensor, title in zip(axes, [sample['frames'][0], sample['frames'][-1], sample['target']], ['Oldest hazy', 'Current hazy', 'Target']):
    axis.imshow(tensor.permute(1, 2, 0)); axis.set_title(title); axis.axis('off')
plt.tight_layout()''')
section('## 5. Training command: GPU, resume and automatic report', '''TRAIN_COMMAND = command('train.py', '--data-root', DATASET_ROOT, '--output-dir', RUN_ROOT, '--val-split', 'auto', '--seq-len', SEQ_LEN, '--image-size', IMAGE_SIZE, '--batch-size', BATCH_SIZE, '--epochs', EPOCHS, '--hidden-dim', HIDDEN_DIM, '--temporal-mode', MODE, '--seed', SEED, '--split-seed', SPLIT_SEED, '--require-cuda', '--report-after-training', '--resume', 'auto')
import shlex
print(shlex.join(TRAIN_COMMAND))
if RUN_TRAINING:
    subprocess.run(TRAIN_COMMAND, check=True)
else:
    print('Set RUN_TRAINING=True to train. For a connection-independent run, use the SSH section.')''')
section('## 6. Regenerate reports without retraining', '''if RUN_REPORT_ONLY:
    subprocess.run(command('report.py', '--run-dir', RUN_ROOT, '--data-root', DATASET_ROOT), check=True)''')
section('## 7. Check and view every generated visual', '''import json
from IPython.display import display, Image, HTML
artifacts = RUN_ROOT / 'artifacts'
checklist = artifacts / 'figure_checklist.json'
if checklist.is_file():
    status = json.loads(checklist.read_text())
    print(status)
    if status.get('status') != 'PASS':
        raise RuntimeError('Report incomplete: rerun the report cell')
    for path in sorted(artifacts.glob('*.png')):
        display(Image(filename=str(path)))
    display(Image(filename=str(artifacts / 'temporal_comparison.gif')))
else:
    print('Reports will be generated after training completes.')''')
cells.append(nbf.v4.new_markdown_cell('''## 8. Training through SSH
Use the SSH command from RunPod Connect on your computer, then inside the pod:

```bash
cd /workspace/videoeennet-agri-dehazing-final
tmux new -s videoeenet
python scripts/runpod_check.py --data-root /workspace/datasets/REVIDE --output /workspace/videoeennet_runs/dataset_check.json
python scripts/train.py --data-root /workspace/datasets/REVIDE --output-dir /workspace/videoeennet_runs/convlstm_seed1234 --val-split auto --seq-len 10 --image-size 256 --batch-size 1 --epochs 30 --require-cuda --report-after-training --resume auto
```
Detach with Ctrl+B then D; reconnect using `tmux attach -t videoeenet`. tmux survives SSH disconnection, not pod termination. Use a persistent volume for data, runs and reports. After pod restart run the same command to resume from the last completed epoch. Keep the original epoch budget and model configuration.

Reports: PNG/PDF curves, metrics distributions, comparisons/error maps/zooms, lowest-PSNR cases, sequence grid, temporal diagnostics, memory activations, GIF, CSV, JSON and HTML. The checklist lists all files and hashes. Raw attention weights and flow maps are not exported for VideoEENet. Never interpret memory activations as attention probabilities.'''))
notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}})
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'notebooks' / 'VideoEENet_RunPod.ipynb')
