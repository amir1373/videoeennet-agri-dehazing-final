#!/bin/bash
# Temporal-variant ablation on a shared VideoEENet backbone.
# Runs the two modes the repo already supports but which were never trained:
# spatial_transformer and hybrid. convlstm is already done (aug_seed1234, TEST 21.4653).
# Everything except --temporal-mode is held identical to aug_seed1234 so the comparison
# isolates the temporal stage: same augmentation bundle, seed, schedule, split and protocol.
set -uo pipefail
cd /workspace/repos/videoeennet-agri-dehazing-final
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/pylibs
for MODE in spatial_transformer hybrid; do
  OUT=/workspace/videoeennet_runs/aug_${MODE}_seed1234
  echo "=========== $MODE  $(date -u +%FT%TZ) ==========="
  python3 scripts/train.py \
    --data-root /workspace/datasets/REVIDE_sequences \
    --output-dir "$OUT" \
    --val-split auto --split-seed 1234 --seed 1234 \
    --seq-len 10 --image-size 256 --batch-size 1 --epochs 30 \
    --temporal-mode "$MODE" \
    --augment --aug-scale-min 0.6 --aug-hflip 0.5 --aug-treverse 0.25 --aug-gamma 0.0 \
    --num-workers 16 --prefetch-factor 6 \
    --require-cuda --report-after-training --resume auto
  rc=$?
  n=$(wc -l < "$OUT/history.jsonl" 2>/dev/null || echo 0)
  echo "--- $MODE exit=$rc epochs=$n/30 ---"
  # Only claim completion on a real 30-epoch history, never on the exit code alone.
  [ "$n" -ge 30 ] && touch "/workspace/videoeennet_runs/${MODE}_COMPLETE"
done
touch /workspace/videoeennet_runs/VARIANTS_DONE
echo "=== variants finished $(date -u +%FT%TZ) ==="
