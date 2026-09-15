import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataloader import discover_sequences, _pair_paths
from model import VideoEENet
from train import save_checkpoint, ssim
from run_multiseed import mean_ci


def test_missing_split_never_falls_back(tmp_path):
    (tmp_path / 'Test').mkdir()
    with pytest.raises(FileNotFoundError):
        discover_sequences(tmp_path, 'Train')


def test_pairing_and_temporal_order(tmp_path):
    hazy, clean = tmp_path / 'hazy', tmp_path / 'gt'
    hazy.mkdir(); clean.mkdir()
    for folder in (hazy, clean):
        for name in ('10.png', '2.png', '1.png'):
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(folder / name)
    assert [p.name for p, _ in _pair_paths(hazy, clean)] == ['1.png', '2.png', '10.png']
    (clean / '2.png').unlink()
    with pytest.raises(ValueError):
        _pair_paths(hazy, clean)


@pytest.mark.parametrize('mode', ['convlstm', 'spatial_transformer', 'hybrid'])
def test_temporal_backward(mode):
    model = VideoEENet(2, 4, temporal_mode=mode, attention_pool_size=2, max_seq_len=3)
    frames = torch.rand(1, 3, 3, 17, 19, requires_grad=True)
    output = model(frames)
    output.mean().backward()
    assert output.shape == (1, 3, 17, 19)
    assert torch.isfinite(frames.grad).all()
    assert frames.grad[:, 0].abs().sum() > 0


def test_checkpoint_and_metrics(tmp_path):
    path = tmp_path / 'last.pt'
    save_checkpoint(path, {'step': 5})
    assert torch.load(path)['step'] == 5
    assert not path.with_suffix('.pt.tmp').exists()
    image = torch.rand(1, 3, 32, 32)
    assert ssim(image, image) == pytest.approx(1.0)
    assert mean_ci([1., 2., 3.])['ci95'] > 2
    with pytest.raises(ValueError):
        mean_ci([1.])


def test_modality_frame_pairing(tmp_path):
    hazy, clean = tmp_path / 'hazy', tmp_path / 'clean'
    hazy.mkdir(); clean.mkdir()
    for folder, name in [(hazy, 'hazy_001.png'), (clean, 'clean_001.png')]:
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(folder / name)
    assert len(_pair_paths(hazy, clean)) == 1
