#!/usr/bin/env bash
# lidar-primary pipeline, stage A: sessions (FAST-LIO poses) -> per-place merge (rigid) -> per-place COLMAP BA refine
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
mkdir -p sessions_lio_dense logs
prep() { [ -f sessions_lio_dense/$1/session.npz ] || python -I scripts/bag_to_colmap.py bags/$1 sessions_lio_dense/$1 --prep_only --odom_dir lio/level2 --kf_dist 0.05 --kf_deg 2.5 > logs/prep_liod_$1.txt 2>&1; echo "[$(date +%H:%M)] prep $1: $(grep -E 'keyframes|primary' logs/prep_liod_$1.txt | tr '\n' ' ')"; }
for p in p1 p2 p3 p4; do
  for h in h1 h2 h3; do prep EB_${p}_$h & done; wait
done
for p in p1 p2 p3 p4; do
  [ -f data/EB_${p}_liod_merged/sparse/0/points3D.bin ] || python -I scripts/merge_heights.py data/EB_${p}_liod_merged \
     sessions_lio_dense/EB_${p}_h1 sessions_lio_dense/EB_${p}_h2 sessions_lio_dense/EB_${p}_h3 --rigid_only --planar > logs/merge_liod_$p.txt 2>&1
  echo "[$(date +%H:%M)] merged $p: $(grep -E 'base is|done ->' logs/merge_liod_$p.txt | tr '\n' ' ')"
done
conda activate colmap
for p in p1 p2 p3 p4; do
  [ -f data/EB_${p}_liod_merged/refined.npz ] || python scripts/refine_colmap.py data/EB_${p}_liod_merged > logs/refine_liod_$p.txt 2>&1
  echo "[$(date +%H:%M)] refined $p: $(grep -E 'mounting|after BA mean|pose change|kept' data/EB_${p}_liod_merged/colmap/refine_log.txt | tail -6 | tr '\n' ' ')"
done
echo LIOD_PREP_DONE
