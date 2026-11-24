#!/usr/bin/env bash
# 2026-11-23 re-recording at h1 and h2 (same rig/day-style as EB_1121 h3): poses for each bag, resumable.
# FAST-LIO (lio/<bag>.tum) -> relocalise_bag.py vs the v2 floor lidar map (lio/floor/<bag>.tum)
# -> bag_to_colmap.py --prep_only --odom_dir lio/floor --no_planar, 0.05 m / 2.5 deg keyframes (sessions_lio_dense/<bag>)
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
mkdir -p lio/floor sessions_lio_dense logs
ts() { date '+%m-%d %H:%M'; }
BAGS=${BAGS:-"EB_1123_h1 EB_1123_h2"}
# EB_1123_h2: the automatic wall-band init picked the wrong corridor end (inliers 0.41 -> tracking 0.55-0.69, a different
# corridor). Both 11-23 runs start at the same spot (same door in the first frames) -> start from EB_1123_h1's floor pose
# (x y yaw_deg, lidar z 0.6 from the ceiling distance 1.97 m): inliers 0.89-0.97, 2.8 cm over the first 60 s.
declare -A INIT=([EB_1123_h2]="6.108 -1.715 2.215 0.6")

for b in $BAGS; do          # FAST-LIO uses ROS (one at a time: same domain/topics)
  if [ ! -s lio/$b.tum ]; then
    echo "[$(ts)] FAST-LIO $b"
    ( conda deactivate; lio_ws/run/run_fastlio.sh bags/$b lio/$b.tum 2.0 ) > logs/fastlio_$b.txt 2>&1
    tail -1 logs/fastlio_$b.txt
  fi
done
for b in $BAGS; do
  if [ ! -s lio/floor/$b.tum ]; then
    echo "[$(ts)] relocalise $b"
    python -I scripts/relocalise_bag.py bags/$b lio/$b.tum lio/floor/$b.tum ${INIT[$b]:+--init ${INIT[$b]}} --plot lio/floor/${b}_reloc.png > logs/reloc_$b.txt 2>&1 \
      || { echo "RELOC FAILED $b"; tail -5 logs/reloc_$b.txt; continue; }
    tail -1 logs/reloc_$b.txt
  fi
done
for b in $BAGS; do
  [ -s lio/floor/$b.tum ] || continue
  if [ ! -f sessions_lio_dense/$b/session.npz ]; then
    echo "[$(ts)] keyframes $b"
    python -I scripts/bag_to_colmap.py bags/$b sessions_lio_dense/$b --prep_only --odom_dir lio/floor --no_planar \
      --kf_dist 0.05 --kf_deg 2.5 > logs/prep_liod_$b.txt 2>&1 || echo "PREP FAILED $b"
    grep -E "keyframes|primary" logs/prep_liod_$b.txt | tr '\n' ' '; echo
  fi
done
echo "[$(ts)] RUN_1123_PREP_DONE"
