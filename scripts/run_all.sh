#!/usr/bin/env bash
# merge h1/h2/h3 per place (skips finished merges), then train MCMC on each merged dataset
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
merge() {
  [ -f data/EB_$1_merged/sparse/0/points3D.bin ] && return
  while pgrep -f "merge_heights.py data/EB_$1_merged" >/dev/null; do sleep 20; done  # already running
  [ -f data/EB_$1_merged/sparse/0/points3D.bin ] && return
  python -I scripts/merge_heights.py data/EB_$1_merged sessions/EB_$1_h1 sessions/EB_$1_h2 sessions/EB_$1_h3 > logs/merge_$1.txt 2>&1
  echo "[$(date +%H:%M)] merged $1: $(grep -E 'done ->' logs/merge_$1.txt)"
}
train() {
  scripts/train_mcmc.sh EB_$1_merged --steps-scaler 1.5
  echo "[$(date +%H:%M)] trained $1: $(cat results/EB_$1_merged/stats/val_step44999.json 2>/dev/null)"
}
mkdir -p logs
train p1 & T=$!
merge p3; merge p4
wait $T
for p in p2 p3 p4; do train $p; done
echo ALL_DONE
