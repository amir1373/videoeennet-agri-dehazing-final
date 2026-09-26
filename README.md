# VideoEENet: Temporal Video Dehazing

For GPU training, SSH/tmux, validated Train/validation/Test separation, and automatic
post-training figures, follow [RUNPOD.md](RUNPOD.md) and
[the RunPod notebook](notebooks/VideoEENet_RunPod.ipynb). Use `--val-split auto`
and `--require-cuda --report-after-training` for the complete workflow.

VideoEENet dehazes the current video frame using a chronological window of the current frame plus its previous nine hazy frames. Its per-frame encoder is a simplified version of EENet's frequency/spatial dual-domain design, taken from the author's own EENet implementation ([amir1373/EENet-Dehazing](https://github.com/amir1373/EENet-Dehazing)), and it adds ConvLSTM temporal memory before reconstruction.

## Architecture

`10 hazy frames -> EENet dual-domain encoder -> ConvLSTM temporal memory -> decoder -> clean final frame`

The model input is `[B, T, 3, H, W]`, with `T=10` by default. The output is the clean reconstruction of the final frame, `[B, 3, H, W]`. The model is fully convolutional and pads dimensions internally to multiples of four during inference.

## RunPod Setup

```bash
git clone https://github.com/amir1373/videoeennet-agri-dehazing-final.git
cd videoeennet-agri-dehazing-final
pip install -r requirements.txt
```

Set `--data-root` to the mounted REVIDE location, for example `/workspace/REVIDE_sequences`.

The loader discovers this common paired layout:

```text
REVIDE_sequences/
  Train/
    hazy/<sequence_name>/<frame>.png
    gt/<sequence_name>/<frame>.png
  Test/
    hazy/<sequence_name>/<frame>.png
    gt/<sequence_name>/<frame>.png
```

It also accepts a sequence directory containing sibling `hazy` and `gt` (or `clean`) directories. Hazy and clean filenames should match.

## Inspect the Dataset

Create a CSV of valid sliding 10-frame windows without copying images:

```bash
python scripts/prepare_sequences.py --data-root /workspace/REVIDE_sequences --split Train --seq-len 10 --output /workspace/videoeennet_runs/train_manifest.csv
```

## Train

This enables CUDA AMP, tqdm batch progress, TensorBoard logging, validation PSNR/SSIM, and automatic resume from `last.pt`:

```bash
python scripts/train.py --data-root /workspace/REVIDE_sequences --train-split Train --val-split Test --output-dir /workspace/videoeennet_runs --seq-len 10 --image-size 256 --batch-size 2 --epochs 30 --resume auto
```

Use `--resume ''` to explicitly start a new run. To resume a specific checkpoint, give its full path.

`--temporal-mode` selects a controlled temporal ablation: `convlstm`, `spatial_transformer`, or `hybrid`. The spatial transformer uses per-patch temporal tokens rather than one globally pooled token per frame. Keep `--hidden-dim 64` fixed for every mode so recurrent capacity does not confound the comparison.

Checkpoints contain model, optimizer, scheduler, AMP scaler, epoch, global step, best metrics, and configuration. The run writes `last.pt`, `best_psnr.pt`, `best_ssim.pt`, and periodic `epoch_###.pt` files to `<output-dir>/checkpoints/`.

```bash
tensorboard --logdir /workspace/videoeennet_runs/logs --bind_all
```

## Multi-Seed Protocol

Single runs are not sufficient for a model ranking. Run each temporal mode with the identical data split, resolution, sequence length, optimizer settings, and seeds. This command records per-seed metrics and writes mean, standard deviation, and a 95% confidence interval:

```bash
python scripts/run_multiseed.py --data-root /workspace/REVIDE_sequences --output-dir /workspace/videoeennet_runs/ablations --temporal-mode hybrid --seeds 7 19 31 --epochs 30 --seq-len 5 --hidden-dim 64
```

Repeat for `convlstm` and `spatial_transformer`. Repeat the protocol for `--seq-len 3`, `5`, and `7` before making sequence-length claims. The code reports results; it does not make unearned performance claims.

## RunPod Notebook

Open [notebooks/VideoEENet_RunPod.ipynb](notebooks/VideoEENet_RunPod.ipynb) in Jupyter on RunPod. It imports repository modules, provides data/sequence/debug visualizations, launches resumable training, evaluates a checkpoint, runs inference, and starts the controlled multi-seed protocol without duplicating model code.

## Inference

Pass a chronologically named folder of hazy frames. The first nine predictions repeat the first frame to fill the temporal window.

```bash
python scripts/inference.py --input-dir /workspace/REVIDE_sequences/Test/hazy/sequence_001 --output-dir /workspace/videoeennet_outputs/sequence_001 --checkpoint /workspace/videoeennet_runs/checkpoints/best_psnr.pt --seq-len 10
```

## Verification

```bash
python -c "from scripts.model import shape_test; print(shape_test())"
```

Expected output includes `frames: (1, 10, 3, 64, 64)` and `prediction: (1, 3, 64, 64)`.

## Notes

### Verification Limits

The multi-seed summaries currently describe best validation scores, not held-out test performance. Do not use them as final paper test results. Use a separate validation split for checkpoint selection and reserve Test for final evaluation; the example `--val-split Test` is exploratory only. Missing split folders and mismatched frame names now raise errors instead of silently falling back. Resume requires the original training configuration, including the epoch budget, and restarts at the last completed epoch. Epoch checkpoints do not recover unfinished-epoch progress. SSIM requires `pytorch-msssim`; the former global-statistics fallback has been removed, so old fallback scores are not comparable.

Equal hidden dimensions control recurrent width, not total model parameter count. The transformer/hybrid comparisons still require actual multi-seed experiments and reporting of parameter counts before scientific conclusions. Local CPU tests do not establish CUDA memory requirements or quality on REVIDE.

- This is a supervised temporal dehazing model, not a diffusion model.
- Validation is against the clean final frame in each REVIDE window.
- No training results are claimed until you run this pipeline on your data.

## References

VideoEENet builds on:

- Y. Cui, Q. Wang, C. Li, W. Ren, and A. Knoll, "EENet: An effective and efficient network for single image dehazing," *Pattern Recognition*, vol. 158, art. no. 111074, 2025, doi: [10.1016/j.patcog.2024.111074](https://doi.org/10.1016/j.patcog.2024.111074). Official code: [github.com/c-yn/EENet](https://github.com/c-yn/EENet).
- X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, "Convolutional LSTM network: A machine learning approach for precipitation nowcasting," in *Proc. NeurIPS*, 2015, pp. 802–810.
- X. Zhang et al., "Learning to restore hazy video: A new real-world dataset and a new method," in *Proc. IEEE/CVF CVPR*, 2021, pp. 9239–9248 (the REVIDE dataset).

## Citation

Citation placeholder: add the final paper citation once the manuscript is ready.

## Matched-protocol options (branch `retrain-2026-09`)

- The model now builds only the modules its `--temporal-mode` uses: the default `convlstm` model
  has 1,805,443 parameters, all trained. `load_model_state()` still reads older checkpoints.
- `--temporal-mode single_frame`: the identical network given only the final frame of each window
  (no temporal information), for testing whether temporal information helps.
- `--aug-vflip P`: vertical flip with probability P (default 0).
- `--val-group scene`: validation holds out whole training scenes instead of 10-frame chunks of
  training videos.
- `scripts/analysis/matched_eval.py` scores under the crop protocol with TRDN's exact metrics
  (scikit-image SSIM on uint8, LPIPS-Alex) and records per-window PSNR.

## Further options (2026-09-25/26)

All default to off; with none set, the model and training are unchanged (a seed-1234 reproduction
matches the earlier reference exactly).

| option (`scripts/train.py`) | effect |
|---|---|
| `--output-mode logit_residual` | output `sigmoid(logit(current) + residual)`, which returns the input for a zero residual |
| `--loss mse_l1` | MSE + 0.1 × L1 instead of L1 |
| `--align raft` | warp earlier frames onto the current one with frozen RAFT-small before encoding |
| `--backbone full` | two dual-domain blocks per encoder stage and a U-Net decoder with current-frame skips (3.66 M parameters) |
| `--retrieval-index FILE` | references retrieved from up to 30 earlier frames (`scripts/retrieval.py`) |
| `--occlusion`, `--occlusion-coverage-min/max`, `--occlusion-scope` | train on opaque occluders with a mask input channel (`scripts/occlusion.py`) |

`scripts/analysis/matched_eval.py` takes an optional coverage and scope (`… crop 0.35 lens`), reads
`RETRIEVAL_INDEX` and writes every prediction to `SAVE_PREDICTIONS=file.npz`; per-window SSIM and
LPIPS are now recorded.
