#!/usr/bin/env bash
# v2: lidar-junction-consistent place alignment (T265 scale kept), depth + lidar-surface constrained MCMC partitions
ROOT=$HOME/gsplat_eb; cd $ROOT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -x
[ -f data/EB_floor2v2_merged/sparse/0/images.bin ] || python -I scripts/build_floor2.py --sweeps 0 --T_file place_T_v2.npz \
  --scale_file "" --simscale_file place_simscale_v2.npz --out $ROOT/data/EB_floor2v2_merged 2>&1 | grep -v Warning
[ -f data/EB_floor2v2_final/sparse/0/images.bin ] || python -I scripts/filter_names.py data/EB_floor2v2_merged data/EB_floor2_final data/EB_floor2v2_final 2>&1 | grep -v Warning
for pz in "p1 -0.23" "p2 -0.21" "p3 -0.19" "p4 -0.19"; do set -- $pz
  [ -f data/EB_v2part_$1/sparse/0/images.bin ] || python -I scripts/subset_dataset.py data/EB_floor2v2_final data/EB_v2part_$1 ${1}_ 2.0 -0.0242 $2 2>&1 | grep -v Warning
  [ -f data/EB_v2partD_$1/sparse/0/images.bin ] || MIN_D=0.5 python -I scripts/make_depth_dataset.py data/EB_v2part_$1 data/EB_v2partD_$1 2>&1 | grep -v Warning | tail -1
done
[ -f data/floor2/surface_grid_v2.npz ] || python -I scripts/make_surface_grid.py data/EB_v2part_p1,data/EB_v2part_p2,data/EB_v2part_p3,data/EB_v2part_p4 data/floor2/surface_grid_v2.npz 0.05 0.15 2>&1 | grep -v Warning
set +x
for spec in "p2 0.002 100" "p1 0.00275 183" "p3 0.0042 418" "p4 0.0021 109"; do set -- $spec
  [ -f results/EB_v2partD_$1/stats/val_step59999.json ] && continue
  rm -rf results/EB_v2partD_$1
  scripts/train_mcmc.sh EB_v2partD_$1 --steps-scaler 2 --no-normalize-world-space --near-plane 0.15 --max-scale 0.3 \
    --opacity-reg 0.001 --scale-reg 0.001 --strategy.noise-lr $3 --depth-loss --depth-lambda $2 \
    --surface-grid $ROOT/data/floor2/surface_grid_v2.npz || echo "$1 FAILED"
  echo "[$(date +%H:%M)] v2 $1: $(cat results/EB_v2partD_$1/stats/val_step59999.json 2>/dev/null)"
done
python -I scripts/merge_partitions.py --prefix EB_v2partD_ --data $ROOT/data/EB_floor2v2_final --out results/EB_floor2_v2 \
  --carve_r 0 --prune_views 0 --min_opacity 0 2>&1 | grep -v Warning
echo V2_DONE
