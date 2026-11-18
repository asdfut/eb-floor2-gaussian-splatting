#!/usr/bin/env bash
# final: build BA-refined datasets as they become ready and train MCMC (60k steps) on each
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
for p in p1 p2 p3 p4; do
  until [ -f data/EB_${p}_merged/refined.npz ]; do sleep 30; done
  [ -f data/EB_${p}_final/sparse/0/points3D.bin ] || python -I scripts/build_final_dataset.py data/EB_${p}_merged data/EB_${p}_final 2>&1 | grep -v Warning
  scripts/train_mcmc.sh EB_${p}_final --steps-scaler 2
  echo "[$(date +%H:%M)] trained $p: $(cat results/EB_${p}_final/stats/val_step59999.json 2>/dev/null)"
done
echo ALL_DONE
