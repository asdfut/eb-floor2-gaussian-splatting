#!/usr/bin/env bash
# Whole-floor model from the NEW recordings at all three heights (user choice 2026-11-24):
#   EB_1121 (h3, 11-21) + EB_1123_h2 + EB_1123_h1 (11-23). Same v2/EB_1121 recipe and 8-partition split.
# Poses: scripts/run_1123_prep.sh (FAST-LIO -> relocalise vs v2 floor map -> sessions_lio_dense/<bag>)
#        -> build_session_dataset.py init (3 sessions, one camera each) -> refine_colmap.py (ONE joint BA, so the
#        heights are tied together by cross-height matches) -> build_session_dataset.py refined = data/EB_new3h_floor.
# Fixed test: EB_1121_gs's 149 held-out h3 images are excluded from training -> direct h3 comparison with EB_1121_gs.
# Resumable: finished steps are skipped.
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PARTS="p1a p1b p2a p2b p3a p3b p4a p4b"
declare -A FZ=([p1]=-0.205 [p2]=-0.155 [p3]=-0.185 [p4]=-0.175)
I=data/EB_new3h_init; D=data/EB_new3h_floor; PM=data/floor2_lio/part8_new3h.json; OUT=results/EB_new3h_gs
TEST=results/EB_1121_gs/heldout_names.json
ts() { date '+%m-%d %H:%M'; }

for s in EB_1121 EB_1123_h2 EB_1123_h1; do
  [ -f sessions_lio_dense/$s/session.npz ] || { echo "missing sessions_lio_dense/$s (run run_1123_prep.sh)"; exit 1; }
done
if [ ! -f $I/sparse/0/images.bin ]; then
  echo "[$(ts)] init dataset (3 sessions)"
  python -I scripts/build_session_dataset.py init sessions_lio_dense/EB_1121,sessions_lio_dense/EB_1123_h2,sessions_lio_dense/EB_1123_h1 \
    $I --prefix n --tag h3,h2,h1 2>&1 | grep -v -i warn | grep -E "session|lidar points|init cloud|done"
fi
if [ ! -f $I/refined.npz ]; then
  echo "[$(ts)] COLMAP refine (joint BA, lidar pose priors)"
  conda activate colmap
  python scripts/refine_colmap.py $I > logs/refine_new3h.txt 2>&1 || { echo REFINE FAILED; exit 1; }
  conda activate gs
  grep -E "pose change|kept" $I/colmap/refine_log.txt | tail -2
fi
if [ ! -f $D/sparse/0/images.bin ]; then
  python -I scripts/build_session_dataset.py refined $I $D 2>&1 | grep -v -i warn | tail -2
fi
if [ ! -f $PM ]; then   # image -> partition of the nearest v2 camera (xy), so the partitions match v2's
  python -I - <<'EOF'
import json, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/gsplat/examples'))
from pycolmap import SceneManager
from scipy.spatial import cKDTree
def cams(d):
    m = SceneManager(d + '/sparse/0/'); m.load_images()
    return [m.images[i].name for i in m.images], np.array([(-m.images[i].R().T @ m.images[i].tvec)[:2] for i in m.images])
on, oc = cams('data/EB_liod_floor'); pm = json.load(open('data/floor2_lio/part8.json'))
keep = [k for k, n in enumerate(on) if n in pm]; lab = np.array([pm[on[k]] for k in keep]); tree = cKDTree(oc[keep])
nn, nc = cams('data/EB_new3h_floor'); d, j = tree.query(nc)
out = {n: str(lab[k]) for n, k in zip(nn, j)}
json.dump(out, open('data/floor2_lio/part8_new3h.json', 'w'))
u, c = np.unique(list(out.values()), return_counts=True)
h = np.array([n.split('_')[1] for n in nn])
print('part map:', dict(zip(u, c.tolist())), '| heights:', {k: int((h == k).sum()) for k in ('h1', 'h2', 'h3')},
      f'| nearest v2 camera: median {np.median(d):.2f} m, max {d.max():.2f} m')
EOF
fi

for q in $PARTS; do
  N=N3H_$q
  if [ -f results/$N/ckpts/ckpt_colourfix.pt ]; then echo "  $q: done, skip"; continue; fi
  if [ ! -f data/$N/sparse/0/images.bin ]; then
    rm -rf data/${N}_raw data/$N
    EXCLUDE=$TEST PART_MAP=$PM VIS_R=12 VIS_DEG=35 python -I scripts/subset_dataset.py $D data/${N}_raw $q 2.0 -0.0252 ${FZ[${q:0:2}]} 2>&1 | grep -v Warn
    MIN_D=1.0 python -I scripts/make_depth_dataset.py data/${N}_raw data/$N 2>&1 | grep -v Warning | tail -1
    rm -rf data/${N}_raw
  fi
  SS=$(python -I -c "
import sys; sys.path.insert(0,'$HOME/gsplat/examples')
from datasets.colmap import Parser
print(Parser('data/$N',factor=1,normalize=False,test_every=8).scene_scale*1.1)" 2>/dev/null | tail -1)
  LAM=$(python3 -c "print(round(0.05/$SS,5))"); NOISE=$(python3 -c "print(round(5e4*1.21/$SS**2,1))")
  echo "[$(ts)] $q: $(ls data/$N/images | wc -l) images, scene scale $SS -> depth_lambda $LAM, noise_lr $NOISE"
  if [ -f results/$N/stats/val_step59999.json ] && ls results/$N/ckpts/ckpt_59999_rank0.pt >/dev/null 2>&1; then
    echo "  $q: already trained, calibrate only"
  else
    rm -rf results/$N
    scripts/train_mcmc.sh $N --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 1.0 \
      --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $NOISE --depth-loss --depth-lambda $LAM \
      --exposure-comp --wall-normals $ROOT/data/floor2_lio/surface_normals_d.npz || echo "$q TRAIN FAILED"
    echo "[$(ts)] $q trained: $(cut -c1-100 results/$N/stats/val_step59999.json 2>/dev/null)"
  fi
  python -I scripts/colour_calibrate.py results/$N data/$N 2>&1 | grep -v -i warn | tail -1
  if [ -f results/$N/ckpts/ckpt_colourfix.pt ]; then
    rm -rf results/$N/ckpts/ckpt_[0-9]*_rank0.pt results/$N/renders results/$N/videos
  else echo "$q CALIBRATE FAILED"; fi
done

if [ ! -f $OUT/eval_fair.json ]; then
  python -I scripts/merge_partitions.py --parts $PARTS --part_map $PM --prefix N3H_ --ckpt_name ckpt_colourfix.pt \
    --data $ROOT/$D --out $OUT --carve_r 0 --prune_views 0 --min_opacity 0 --blend_m 1.5 2>&1 | grep -v Warning | tail -2
  python -I scripts/eval_colourfit.py $OUT $ROOT/$D 2>&1 | grep -v -i warn | tail -1
  echo "[$(ts)] merged + evaluated on its own held-out images"
fi
# comparisons (exposure-fair PSNR): h3 vs EB_1121_gs on its 149 fixed test images; v2 at the new held-out cameras;
# new model on v2's fixed 483-image test set (cross-day)
[ -f $OUT/eval_on_1121test.json ] || python -I scripts/eval_colourfit.py $OUT $ROOT/$D $TEST $OUT/eval_on_1121test.json 2>&1 | grep -v -i warn | tail -1
if [ ! -f $OUT/eval_v2_on_new.json ]; then
  python -I -c "
import json; json.dump(list(json.load(open('$OUT/eval.json'))['per_image']), open('$OUT/heldout_names.json','w'))"
  python -I scripts/eval_colourfit.py results/EB_liod_floor_gs_cf $ROOT/$D $OUT/heldout_names.json $OUT/eval_v2_on_new.json 2>&1 | grep -v -i warn | tail -1
  python -I scripts/eval_colourfit.py results/EB_1121_gs $ROOT/$D $OUT/heldout_names.json $OUT/eval_1121_on_new.json 2>&1 | grep -v -i warn | tail -1
fi
[ -f $OUT/eval_new_on_v2test.json ] || python -I scripts/eval_colourfit.py $OUT $ROOT/data/EB_liod_floor \
  data/floor2_lio/floor_test483.json $OUT/eval_new_on_v2test.json 2>&1 | grep -v -i warn | tail -1
echo "[$(ts)] evaluation done"

if [ ! -f $OUT/videos/floor2_h3_front_x2.mp4 ]; then
  python -I scripts/render_floor_video.py --ckpt $OUT/ckpts/ckpt_merged.pt --data $ROOT/$D \
    --route data/floor2_lio/route_v3.npy --walls_dir data/floor2_lio --clear 0.55 --snap_view 20 --hfov 75 --pitch -4 \
    --out $OUT/videos --modes stack front 2>&1 | grep -v -i warn | tail -4
  ( cd $OUT/videos && for f in floor2_3heights_front floor2_h3_front floor2_h2_front floor2_h1_front; do
      [ -f $f.mp4 ] || continue
      for sp in "1.5 45" "2 60"; do set -- $sp; g=$(( $2 / 2 ))
        ffmpeg -v error -y -i $f.mp4 -vf "setpts=PTS/$1" -r $2 -c:v libx264 -crf 18 -preset medium -g $g \
          -keyint_min $g -sc_threshold 0 -pix_fmt yuv420p -movflags +faststart ${f}_x$1.mp4; done; done )
  echo "[$(ts)] videos done"
fi
echo RUN_NEW3H_DONE
