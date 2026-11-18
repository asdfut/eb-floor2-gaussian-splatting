#!/usr/bin/env bash
# lidar-primary pipeline, stage A: sessions (FAST-LIO poses) -> per-place merge (rigid) -> per-place COLMAP BA refine
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
mkdir -p sessions_lio logs
prep() { [ -f sessions_lio/$1/session.npz ] || python -I scripts/bag_to_colmap.py bags/$1 sessions_lio/$1 --prep_only --odom_dir lio/level2 > logs/prep_lio_$1.txt 2>&1; echo "[$(date +%H:%M)] prep $1: $(grep -E 'keyframes|primary' logs/prep_lio_$1.txt | tr '\n' ' ')"; }
for p in p1 p2 p3 p4; do
  for h in h1 h2 h3; do prep EB_${p}_$h & done; wait
done
for p in p1 p2 p3 p4; do
  [ -f data/EB_${p}_lio_merged/sparse/0/points3D.bin ] || python -I scripts/merge_heights.py data/EB_${p}_lio_merged \
     sessions_lio/EB_${p}_h1 sessions_lio/EB_${p}_h2 sessions_lio/EB_${p}_h3 --rigid_only --planar > logs/merge_lio_$p.txt 2>&1
  echo "[$(date +%H:%M)] merged $p: $(grep -E 'base is|done ->' logs/merge_lio_$p.txt | tr '\n' ' ')"
done
conda activate colmap
for p in p1 p2 p3 p4; do
  [ -f data/EB_${p}_lio_merged/refined.npz ] || python scripts/refine_colmap.py data/EB_${p}_lio_merged > logs/refine_lio_$p.txt 2>&1
  echo "[$(date +%H:%M)] refined $p: $(grep -E 'mounting|after BA mean|pose change|kept' data/EB_${p}_lio_merged/colmap/refine_log.txt | tail -6 | tr '\n' ' ')"
done
echo LIO_PREP_DONE
