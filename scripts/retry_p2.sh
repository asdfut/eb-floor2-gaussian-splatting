#!/usr/bin/env bash
# p2 OOM'd at the 3M cap (step ~6600); rerun after run_final.sh finishes, with packed rasterization + expandable segments
ROOT=$HOME/gsplat_eb; cd $ROOT
while pgrep -f "run_final.sh" >/dev/null; do sleep 60; done
mv results/EB_p2_final results/EB_p2_final_oom_$(date +%H%M)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/train_mcmc.sh EB_p2_final --steps-scaler 2 --packed
echo "[$(date +%H:%M)] trained p2: $(cat results/EB_p2_final/stats/val_step59999.json 2>/dev/null)"
