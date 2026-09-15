import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from runpod_check import check


def test_train_to_complete_report(tmp_path):
    data = tmp_path / 'data'
    for split, names in [('Train', ['a', 'b']), ('Test', ['c'])]:
        for name in names:
            for kind in ['hazy', 'gt']:
                folder = data / split / kind / name
                folder.mkdir(parents=True)
                for frame in range(5):
                    grid = np.arange(32 * 32 * 3).reshape(32, 32, 3)
                    image = ((grid + frame * 5 + (20 if kind == 'hazy' else 0)) % 256).astype('uint8')
                    Image.fromarray(image).save(folder / f'{frame}.png')
    inventory = check(data, tmp_path / 'inventory.json', seq_len=3, size=32, require_cuda=False)
    assert set(inventory['splits']['train']['sequences']).isdisjoint(inventory['splits']['val']['sequences'])
    run = tmp_path / 'run'
    command = [sys.executable, str(ROOT / 'scripts/train.py'), '--data-root', str(data), '--output-dir', str(run), '--seq-len', '3', '--image-size', '32', '--batch-size', '1', '--epochs', '1', '--base-channels', '2', '--hidden-dim', '4', '--num-workers', '0', '--no-amp', '--report-after-training']
    subprocess.run(command, check=True, cwd=ROOT, timeout=180)
    # Execute the real notebook on the local fixture, replacing only setup,
    # paths and the GPU gate. Training/report subprocesses are exercised above.
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(ROOT / 'notebooks/VideoEENet_RunPod.ipynb', as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    code_cells[0].source = f"from pathlib import Path\nimport os, subprocess, sys\nREPO_ROOT=Path({str(ROOT)!r})\nos.chdir(REPO_ROOT)"
    code_cells[1].source = code_cells[1].source.replace("Path('/workspace/datasets/REVIDE')", f"Path({str(data)!r})").replace("Path('/workspace/videoeennet_runs/convlstm_seed1234')", f"Path({str(run)!r})").replace('10, 256, 30, 1', '3, 32, 1, 1')
    code_cells[2].source = f"sys.path.insert(0, str(REPO_ROOT / 'scripts'))\nfrom runpod_check import check\ncheck(DATASET_ROOT, RUN_ROOT / 'notebook_check.json', SEQ_LEN, IMAGE_SIZE, require_cuda=False)"
    NotebookClient(notebook, timeout=180, kernel_name='python3').execute(cwd=str(ROOT))
    checklist = json.loads((run / 'artifacts/figure_checklist.json').read_text())
    assert checklist['status'] == 'PASS'
    assert 'temporal_comparison.gif' in checklist['files']
    for filename in checklist['files']:
        assert (run / 'artifacts' / filename).stat().st_size > 0
    metrics = json.loads((run / 'artifacts/metrics.json').read_text())
    assert metrics['frames_evaluated'] == metrics['frames_available'] == 3
    subprocess.run(command, check=True, cwd=ROOT, timeout=180)
