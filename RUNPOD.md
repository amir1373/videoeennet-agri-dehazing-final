# RunPod GPU training, SSH and reports

## Dataset contract

Set the same absolute dataset root in both projects, for example
`/workspace/datasets/REVIDE`. This is a configurable example, not a verified
location on your pod. Expected extracted layout:

```text
REVIDE/
  Train/hazy/sequence_01/0001.png
  Train/gt/sequence_01/0001.png
  Test/hazy/sequence_99/0001.png
  Test/gt/sequence_99/0001.png
```

Sequence-first folders (`Train/sequence_01/hazy` and `gt`) also work.
`clean` is accepted instead of `gt`. Frame names must match exactly or differ
only by known modality tokens such as `hazy_001` and `clean_001`.
Frames are naturally sorted, and Train/Test must refer to separate files.
This does not decode raw video archives. Extract videos to ordered frame folders first.
Use `--val-split auto`: a deterministic 10 percent of whole Train sequence names
is held out using the same SHA256/name/seed partition as TRDN. Test is never used
for checkpoint selection. Keep `--split-seed 1234` fixed across runs.
VideoEENet resizes to the selected image size; TRDN crops. These preprocessing
protocols must be reported and matched before claiming a controlled comparison.

## Notebook

Open `notebooks/VideoEENet_RunPod.ipynb` on the RunPod Jupyter server.
Set the paths in section 2. Section 3 checks CUDA, reads/verifies every paired
image, checks dimensions and overlap, and exports exact sequence membership.
Section 5 trains with AMP and then automatically calls the full-test visual report.
Section 6 regenerates reports without retraining. Section 7 displays the outputs.

## SSH

On your computer use the exact SSH command shown in RunPod Connect (host, port
and key path depend on your pod). In that SSH session:

```bash
cd /workspace/videoeennet-agri-dehazing-final
git pull --ff-only
python -m pip install -r requirements.txt
tmux new -s videoeenet
python scripts/runpod_check.py --data-root /workspace/datasets/REVIDE --output /workspace/videoeennet_runs/dataset_check.json
python scripts/train.py --data-root /workspace/datasets/REVIDE --output-dir /workspace/videoeennet_runs/convlstm_seed1234 --val-split auto --seq-len 10 --image-size 256 --batch-size 1 --epochs 30 --require-cuda --report-after-training --resume auto
```

Install tmux in the pod if it is absent. Detach with Ctrl+B then D. Reconnect with
`tmux attach -t videoeenet`. GPU training continues when SSH disconnects. A pod
shutdown stops it; restart with the identical training command to load the last
completed epoch. Keep dataset, checkpoints and artifacts on a persistent volume.
Never start two jobs writing the same run directory. To inspect progress use
TensorBoard on the run's logs directory. Training output includes tqdm progress.

## Final artifact inventory

Each run writes `artifacts/index.html`, a full-test per-frame CSV and metrics JSON,
plus PNG and PDF versions of:

- Training loss and validation PSNR/SSIM curves.
- Full-test metric distributions and input-versus-prediction PSNR traces.
- Deterministically selected input/target/prediction comparisons and error maps.
- Center zooms and explicitly labeled lowest-PSNR examples.
- Hazy/clean input sequence grid.
- Temporal feature RMS activations (not attention probabilities).
- Prediction/target IFD and temporal difference-error traces.

It also writes a consecutive-frame comparison GIF and a JSON checklist of files,
sizes and SHA256 hashes. A report failure fails the command, while checkpoints
remain usable. Regenerate using:

```bash
python scripts/report.py --run-dir /workspace/videoeennet_runs/convlstm_seed1234 --data-root /workspace/datasets/REVIDE
```

No flow maps/reference weights/VAE ceiling apply to this architecture. Raw
attention probabilities are not exported. IFD is an unwarped diagnostic; interpret
it with PSNR. A PASS checklist establishes file completeness, not scientific
significance. Multi-seed, multi-variant, temporal-length and second-dataset results
still require actual training runs. No RunPod runtime or real dataset was accessible
when authoring these instructions.
