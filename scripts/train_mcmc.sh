#!/usr/bin/env bash
# Train 3DGS with the MCMC strategy on a dataset made by bag_to_colmap.py / merge_heights.py.
# usage: train_mcmc.sh <dataset name under data/, e.g. EB_p1_merged> [extra simple_trainer args...]
#  --visible-adam : only visible Gaussians are updated. Without it MCMC collapses on these long
#                   corridors (opacity/scale reg drives off-screen Gaussians to zero -> 97% dead).
#  no --depth-loss: gsplat's disparity-L1 explodes on near lidar points (spikes ~57 vs rgb 0.1).
#  --strategy.noise-lr 5e4 (default 5e5): with pose-opt the default MCMC position noise made it
#                   collapse at ~10k steps; 10x lower noise is stable (experiments X1-X4).
#  no --use-bilateral-grid: the per-image 16x16 color grid fits a blurred image by itself
#                   (train loss 0.1) while all Gaussians die -> flat-color renders (PSNR 15).
set -eo pipefail
S=$1; shift
R=${OUT_NAME:-$S}   # result dir name (default = dataset name)
ROOT=$HOME/gsplat_eb
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
set -u
cd ~/gsplat/examples
mkdir -p $ROOT/results/$R
python simple_trainer.py mcmc \
  --data-dir $ROOT/data/$S --data-factor 1 --result-dir $ROOT/results/$R \
  --strategy.cap-max 3000000 \
  --strategy.noise-lr 50000 \
  --visible-adam \
  --pose-opt \
  --eval-steps 2000 7000 30000 --save-steps 7000 30000 --ply-steps 7000 30000 \
  --save-ply \
  --disable-viewer \
  "$@" 2>&1 | tr '\r' '\n' | grep --line-buffered -vE "^\s*$" | awk 'NR%200==1 || /PSNR|Step|Error|Traceback|saved/ {print; fflush()}' > $ROOT/results/$R/train_log.txt
