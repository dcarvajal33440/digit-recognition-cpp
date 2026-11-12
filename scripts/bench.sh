#!/usr/bin/env bash
# Reproducible performance sweep. Prints CSV lines:
#   label,kernel,type,threads,hidden,train_s,img_per_s,test_acc,peak_rss_mib
# usage: scripts/bench.sh [outfile]   (run from the project root, after `make`)
set -euo pipefail
out=${1:-results/bench.csv}
mkdir -p "$(dirname "$out")"
echo "label,kernel,type,threads,hidden,train_s,img_per_s,test_acc,peak_rss_mib" > "$out"

run() {  # label, binary, env-threads, args...
    local label=$1 bin=$2 thr=$3; shift 3
    local line
    line=$(OMP_NUM_THREADS=$thr "$bin" --csv "$@" | grep '^CSV,' | sed 's/^CSV,//')
    echo "$label,$line" | tee -a "$out"
}

H="--hidden 128 --lr 0.05 --mu 0.9 --batch 64"

# --- optimisation ladder (same network, same data; fewer epochs for the slow ones) ---
run "0 baseline (-O2; naive loops)"      build/train_mlp_base 1 $H --kernel naive     --epochs 1
run "1 +O3 -march=native (naive loops)"        build/train_mlp      1 $H --kernel naive     --epochs 1
run "2 +loop reorder (stride-1 inner)"  build/train_mlp      1 $H --kernel reordered --epochs 3
run "3 +cache blocking"                 build/train_mlp      1 $H --kernel blocked   --epochs 3
run "4 +register-tiled SIMD kernel"     build/train_mlp      1 $H --kernel tiled     --epochs 5
run "5 +4 threads (OpenMP)"             build/train_mlp      2 $H --kernel tiled     --epochs 5

# --- batch size x threads (does threading pay once the matrices get bigger?) ---
for b in 32 64 128 256 512; do
  for t in 1 2 4 6; do
    run "batch $b; $t thr" build/train_mlp $t --hidden 128 --lr 0.05 --mu 0.9 --batch $b --kernel tiled --epochs 3
  done
done

# --- a wider network, where each product is large enough to thread ---
for t in 1 2 4 6; do
  run "784-512-256-10 b256; $t thr" build/train_mlp $t --hidden 512,256 --lr 0.05 --mu 0.9 --batch 256 --kernel tiled --epochs 3
done

# --- precision and memory ---
run "float64"                build/train_mlp 1 $H --kernel tiled --epochs 3 --double
run "float32 + preload (f32 dataset copy)" build/train_mlp 1 $H --kernel tiled --epochs 3 --preload
