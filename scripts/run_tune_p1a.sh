#!/usr/bin/env bash
# p1a tuning tests (same recipe as run_liod_train.sh) : A = 5M Gaussians, B = max-scale 0.3 m
ROOT=$HOME/gsplat_eb; cd $ROOT
COMMON="--steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr 214.5 --depth-loss --depth-lambda 0.00298 --exposure-comp --wall-normals $ROOT/data/floor2_lio/surface_normals_d.npz"
for v in "A --max-scale 1.0 --strategy.cap-max 5000000" "B --max-scale 0.3"; do
  set -- $v; n=$1; shift
  rm -rf data/X_p1a_$n results/X_p1a_$n; ln -s $ROOT/data/EB_liodpartD_p1a data/X_p1a_$n
  echo "[$(date +%H:%M)] $n: $*"
  scripts/train_mcmc.sh X_p1a_$n $COMMON "$@" || echo "$n FAILED"
  echo "[$(date +%H:%M)] $n: $(cut -c1-110 results/X_p1a_$n/stats/val_step59999.json 2>/dev/null)"
done
echo TUNE_DONE
