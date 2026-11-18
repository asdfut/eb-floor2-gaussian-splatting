#!/usr/bin/env bash
# depth-supervised partitions (lidar depth, unseen init points removed), then merge + prune + 3-height video.
# per-partition depth weight and MCMC noise are scaled with the scene scale (world-frame training):
#   depth_lambda = 0.05 / scene_scale,  noise_lr = 5e4 * 1.21 / scene_scale^2
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "[d]ata-dir $HOME/gsplat_eb/data/EB_partD_p2 " >/dev/null; do sleep 30; done
for spec in "p1 0.00275 183" "p3 0.0042 418" "p4 0.0021 109"; do
  set -- $spec
  [ -f results/EB_partD_$1/stats/val_step59999.json ] && continue
  rm -rf results/EB_partD_$1
  scripts/train_mcmc.sh EB_partD_$1 --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 \
    --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $3 --depth-loss --depth-lambda $2 || echo "$1 FAILED"
  echo "[$(date +%H:%M)] partD $1: $(cat results/EB_partD_$1/stats/val_step59999.json 2>/dev/null)"
done
echo "[$(date +%H:%M)] partD p2: $(cat results/EB_partD_p2/stats/val_step59999.json 2>/dev/null)"
python -I scripts/merge_partitions.py --prefix EB_partD_ --out results/EB_floor2_gsD 2>&1 | grep -v Warning
python -I scripts/render_floor_video.py --ckpt results/EB_floor2_gsD/ckpts/ckpt_merged.pt --out results/EB_floor2_gsD/videos --modes stack 2>&1 | grep -v Warning
echo PARTSD_DONE
