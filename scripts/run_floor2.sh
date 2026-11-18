#!/usr/bin/env bash
# whole 2F: global BA over all places -> final dataset -> MCMC training (120k steps)
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh
D=data/EB_floor2_merged
if [ ! -f $D/refined.npz ]; then
  conda activate colmap
  python scripts/refine_colmap.py $D --skip_mount --cross_k 10 --relax_first 18 --rounds 3 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"
  conda deactivate
fi
conda activate gs
[ -f data/EB_floor2_final/sparse/0/points3D.bin ] || \
  python -I scripts/build_final_dataset.py $D data/EB_floor2_final 2>&1 | grep --line-buffered -v Warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/train_mcmc.sh EB_floor2_final --steps-scaler 4 || {
  # (4M cap + --packed collapsed at step 8k on this data: flat renders, 3 it/s) -> 3M, unpacked; OOM fallback 2.5M
  echo "3M run failed, retrying with cap 2.5M"
  mv results/EB_floor2_final results/EB_floor2_final_fail_3M
  scripts/train_mcmc.sh EB_floor2_final --steps-scaler 4 --strategy.cap-max 2500000; }
echo "[$(date +%H:%M)] floor2: $(cat results/EB_floor2_final/stats/val_step119999.json 2>/dev/null)"
