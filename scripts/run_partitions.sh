#!/usr/bin/env bash
# whole 2F, partitioned: one MCMC model per recording area (own images + neighbours within 2 m), all in the shared
# floor frame (no world normalisation), 3M Gaussians each, 60k steps; then ownership-crop + merge + held-out eval
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in p1 p2 p3 p4; do
  [ -f results/EB_part_$p/stats/val_step59999.json ] && continue
  rm -rf results/EB_part_$p
  scripts/train_mcmc.sh EB_part_$p --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 --opacity-reg 0.001 --scale-reg 0.001 \
    || echo "[$(date +%H:%M)] $p FAILED"
  echo "[$(date +%H:%M)] part $p: $(cat results/EB_part_$p/stats/val_step59999.json 2>/dev/null)"
done
python -I scripts/merge_partitions.py 2>&1 | grep -v Warning
echo PARTS_DONE
