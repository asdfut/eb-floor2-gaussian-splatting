#!/usr/bin/env bash
# lidar-primary pipeline, stage B: depth datasets -> 4 partitions (MCMC + lidar depth) -> blended merge -> videos
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in p1 p2 p3 p4; do
  [ -f data/EB_liopartD_$p/sparse/0/images.bin ] || MIN_D=1.0 python -I scripts/make_depth_dataset.py data/EB_liopart_$p data/EB_liopartD_$p 2>&1 | grep -v Warning | tail -1
done
for p in p2 p1 p3 p4; do
  [ -f results/EB_liopartD_$p/stats/val_step59999.json ] && continue
  SS=$(python -I -c "
import sys; sys.path.insert(0,'$HOME/gsplat/examples')
from datasets.colmap import Parser
print(Parser('data/EB_liopartD_$p',factor=1,normalize=False,test_every=8).scene_scale*1.1)" 2>/dev/null | tail -1)
  LAM=$(python3 -c "print(round(0.05/$SS,5))"); NOISE=$(python3 -c "print(round(5e4*1.21/$SS**2,1))")
  echo "[$(date +%H:%M)] $p: scene scale $SS -> depth_lambda $LAM, noise_lr $NOISE"
  rm -rf results/EB_liopartD_$p
  scripts/train_mcmc.sh EB_liopartD_$p --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 \
    --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $NOISE --depth-loss --depth-lambda $LAM || echo "$p FAILED"
  echo "[$(date +%H:%M)] lio $p: $(cat results/EB_liopartD_$p/stats/val_step59999.json 2>/dev/null)"
done
python -I scripts/merge_partitions.py --prefix EB_liopartD_ --data $ROOT/data/EB_lio_floor --out results/EB_lio_floor_gs \
  --carve_r 0 --prune_views 0 --min_opacity 0 --blend_m 1.5 2>&1 | grep -v Warning
python -I scripts/render_floor_video.py --ckpt results/EB_lio_floor_gs/ckpts/ckpt_merged.pt --data $ROOT/data/EB_lio_floor \
  --route data/floor2_lio/route.npy --out results/EB_lio_floor_gs/videos --modes stack 2>&1 | grep -v Warning
cd results/EB_lio_floor_gs/videos
for f in floor2_3heights_front floor2_h3_front; do for sp in "1.5 45" "2 60"; do set -- $sp; g=$(( $2 / 2 ))
  ffmpeg -v error -y -i $f.mp4 -vf "setpts=PTS/$1" -r $2 -c:v libx264 -crf 18 -preset medium -g $g -keyint_min $g -sc_threshold 0 \
    -pix_fmt yuv420p -movflags +faststart ${f}_x$1.mp4; done; done
echo LIO_TRAIN_DONE
