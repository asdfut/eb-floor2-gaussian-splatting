#!/usr/bin/env bash
# re-render v2 videos on the corrected route (p1 hall lanes in the recorded direction)
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/EB_liod_floor_gs_cf/videos_v3; rm -rf $O
python -I scripts/render_floor_video.py --ckpt results/EB_liod_floor_gs_cf/ckpts/ckpt_merged.pt --data $ROOT/data/EB_liod_floor \
  --route data/floor2_lio/route_v3.npy --walls_dir data/floor2_lio --clear 0.55 --snap_view 20 --hfov 75 --pitch -4 --out $O --modes stack 2>&1 | grep -v -i warn | tail -2
cd $O
for f in floor2_3heights_front floor2_h3_front; do for sp in "1.5 45" "2 60"; do set -- $sp; g=$(( $2 / 2 ))
  ffmpeg -v error -y -i $f.mp4 -vf "setpts=PTS/$1" -r $2 -c:v libx264 -crf 18 -preset medium -g $g -keyint_min $g -sc_threshold 0 \
    -pix_fmt yuv420p -movflags +faststart ${f}_x$1.mp4; done; done
ls -la *.mp4 | awk '{print $5, $9}'
echo VIDEOS_V3_DONE
