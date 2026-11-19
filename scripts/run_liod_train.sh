#!/usr/bin/env bash
# dense keyframes + 8 partitions + exposure compensation + lidar wall regulariser -> results/EB_liod_floor_gs
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PARTS="p1a p1b p2a p2b p3a p3b p4a p4b"
declare -A FZ=([p1]=-0.205 [p2]=-0.155 [p3]=-0.185 [p4]=-0.175)
for q in $PARTS; do
  [ -f data/EB_liodpart_$q/sparse/0/images.bin ] || PART_MAP=data/floor2_lio/part8.json VIS_R=12 VIS_DEG=35 \
    python -I scripts/subset_dataset.py data/EB_liod_floor data/EB_liodpart_$q $q 2.0 -0.0252 ${FZ[${q:0:2}]} 2>&1 | grep -v Warn
  [ -f data/EB_liodpartD_$q/sparse/0/images.bin ] || MIN_D=1.0 python -I scripts/make_depth_dataset.py data/EB_liodpart_$q data/EB_liodpartD_$q 2>&1 | grep -v Warning | tail -1
done
[ -f data/floor2_lio/surface_normals_d.npz ] || python -I scripts/make_surface_normals.py \
  $(echo $PARTS | sed 's/\([a-z0-9]*\)/data\/EB_liodpart_\1/g; s/ /,/g') data/floor2_lio/surface_normals_d.npz 0.05 2 2>&1 | grep -v Warn
for q in $PARTS; do
  [ -f results/EB_liodpartD_$q/stats/val_step59999.json ] && continue
  SS=$(python -I -c "
import sys; sys.path.insert(0,'$HOME/gsplat/examples')
from datasets.colmap import Parser
print(Parser('data/EB_liodpartD_$q',factor=1,normalize=False,test_every=8).scene_scale*1.1)" 2>/dev/null | tail -1)
  LAM=$(python3 -c "print(round(0.05/$SS,5))"); NOISE=$(python3 -c "print(round(5e4*1.21/$SS**2,1))")
  echo "[$(date +%H:%M)] $q: $(ls data/EB_liodpartD_$q/images | wc -l) images, scene scale $SS -> depth_lambda $LAM, noise_lr $NOISE"
  rm -rf results/EB_liodpartD_$q
  scripts/train_mcmc.sh EB_liodpartD_$q --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 \
    --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $NOISE --depth-loss --depth-lambda $LAM \
    --exposure-comp --wall-normals $ROOT/data/floor2_lio/surface_normals_d.npz || echo "$q FAILED"
  echo "[$(date +%H:%M)] $q: $(cut -c1-100 results/EB_liodpartD_$q/stats/val_step59999.json 2>/dev/null)"
done
python -I scripts/merge_partitions.py --parts $PARTS --part_map data/floor2_lio/part8.json --prefix EB_liodpartD_ \
  --data $ROOT/data/EB_liod_floor --out results/EB_liod_floor_gs --carve_r 0 --prune_views 0 --min_opacity 0 --blend_m 1.5 2>&1 | grep -v Warning
python -I scripts/render_floor_video.py --ckpt results/EB_liod_floor_gs/ckpts/ckpt_merged.pt --data $ROOT/data/EB_liod_floor \
  --route data/floor2_lio/route.npy --walls_dir data/floor2_lio --clear 0.55 --hfov 75 --pitch -4 \
  --out results/EB_liod_floor_gs/videos --modes stack 2>&1 | grep -v Warning | tail -2
cd results/EB_liod_floor_gs/videos
for f in floor2_3heights_front floor2_h3_front; do for sp in "1.5 45" "2 60"; do set -- $sp; g=$(( $2 / 2 ))
  ffmpeg -v error -y -i $f.mp4 -vf "setpts=PTS/$1" -r $2 -c:v libx264 -crf 18 -preset medium -g $g -keyint_min $g -sc_threshold 0 \
    -pix_fmt yuv420p -movflags +faststart ${f}_x$1.mp4; done; done
echo LIOD_TRAIN_DONE
