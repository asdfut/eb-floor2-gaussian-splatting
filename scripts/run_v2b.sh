#!/usr/bin/env bash
# v2b: v2 alignment (lidar-junction consistent, T265 scale) + the stable depth recipe (depth >= 1 m, no surface grid)
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in p1 p2 p3 p4; do
  [ -f data/EB_v2partD_$p/sparse/0/images.bin ] || MIN_D=1.0 python -I scripts/make_depth_dataset.py data/EB_v2part_$p data/EB_v2partD_$p 2>&1 | grep -v Warning | tail -1
done
for spec in "p2 0.002 100" "p1 0.00275 183" "p3 0.0042 418" "p4 0.0021 109"; do set -- $spec
  [ -f results/EB_v2partD_$1/stats/val_step59999.json ] && continue
  rm -rf results/EB_v2partD_$1
  scripts/train_mcmc.sh EB_v2partD_$1 --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 \
    --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $3 --depth-loss --depth-lambda $2 || echo "$1 FAILED"
  echo "[$(date +%H:%M)] v2b $1: $(cat results/EB_v2partD_$1/stats/val_step59999.json 2>/dev/null)"
done
python -I scripts/merge_partitions.py --prefix EB_v2partD_ --data $ROOT/data/EB_floor2v2_final --out results/EB_floor2_v2 \
  --carve_r 0 --prune_views 0 --min_opacity 0 2>&1 | grep -v Warning
python -I scripts/render_floor_video.py --ckpt results/EB_floor2_v2/ckpts/ckpt_merged.pt --data $ROOT/data/EB_floor2v2_final \
  --route data/floor2/route_v2.npy --out results/EB_floor2_v2/videos --modes stack 2>&1 | grep -v Warning
echo V2B_DONE
