#!/usr/bin/env bash
# after all training (run_final.sh + retry_p2.sh) is done: smooth forward + 360 videos for every place and height
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
while pgrep -f "run_final.sh|retry_p2.sh" >/dev/null; do sleep 60; done
for mode in front 360; do
  for p in p1 p2 p3 p4; do
    for h in h1 h2 h3; do
      [ -f results/EB_${p}_final/pano/EB_${p}_final_${h}_${mode}.mp4 ] && continue
      flag=""; [ $mode = front ] && flag="--front"
      python scripts/render_pano.py EB_${p}_final --heights $h $flag 2>&1 | grep -v Warning
    done
  done
done
echo PANO_DONE
