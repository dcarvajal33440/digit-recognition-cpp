# digit-recognition-cpp
A neural network for MNIST written with no ML framework and no linear-algebra library: matrices, GEMM kernels,
activations, forward pass, backpropagation, and gradient descent are all in `src/`. Build order follows the
learning path: **one neuron → ten neurons → multilayer network → measure → optimize**.

```
make            # builds everything (g++, C++17, OpenMP)
make test       # GEMM kernels vs. a double-precision reference + numerical gradient check of backprop
build/neuron --all                                   # stage 1: ten one-vs-rest neurons
build/train_mlp --hidden 512,256 --epochs 15 --lr 0.03 --lr-decay 0.9 --batch 64   # stage 2: the MLP
build/bench_gemm                                     # GFLOP/s of every GEMM kernel on the network's shapes
scripts/bench.sh                                     # the full performance sweep -> results/bench.csv
```
Fetch MNIST with `scripts/get_mnist.sh` (downloads the four IDX files into `data/` and checks their sizes), or
place them there yourself. The loader validates magic numbers and file sizes.

## Results

All numbers are from this repo's `results/` files, on a AMD Ryzen 5 5600H (AVX2, ~5 GiB RAM), g++ 13.3 on Ubuntu 24.04. Expect different absolute numbers on other hardware.

### Accuracy on the 10,000 unseen test images

| Model | Test accuracy |
|---|---|
| Single sigmoid neuron, "is it a 0?" (binary) | 99.24% (but always answering "no" scores 90.2%, so see precision/recall in `results/single_neuron.txt`) |
| Ten one-vs-rest neurons, pick the most confident | **91.45%** |
| MLP 784-128-10, ReLU, momentum SGD, 10 epochs | **98.13%** |
| MLP 784-512-256-10, 15 epochs, LR decay | **98.37%** (9,837 / 10,000) |

Training uses 55k images, 5k are held out for validation and the 10k test images are only scored at the end.
Caveat: I printed test accuracy for four candidate configs and picked the one with the best *validation*
accuracy (it also had the best test accuracy, so the ranking didn't change). Train accuracy reaches 100% while
validation plateaus near 98.3%, so this is the ceiling for an unregularized MLP on raw pixels; dropout,
augmentation or convolutions would be the next step.

### Correctness checks (`make test`)
* **456 GEMM cases** (4 kernels × 3 transpose variants × float/double × ragged shapes) match a double-precision
  reference. Worst error: 5e-5 (float).
* **Backprop vs. finite differences**, ReLU / sigmoid / tanh, naive and optimized kernels: max relative
  error 1e-6 or better (`results/tests.txt`).

### Performance (784-128-10, batch 64, single runs; expect ±5–10% run-to-run noise)

| Step | Training throughput | vs. baseline |
|---|---|---|
| 0. Baseline: `-O2`, textbook i-j-k triple loop | 5,700 img/s | 1× |
| 1. + `-O3 -march=native` (loops unchanged) | 5,900 | 1.0× — no gain: the strided inner loop can't vectorize |
| 2. + loop reordering (stride-1 inner loop) | 48,800 | 8.5× |
| 3. + cache blocking | 44,900 | 7.9× — **slightly worse**: everything already fits in L2 |
| 4. + register-tiled SIMD micro-kernel | 134,000 | **23×** |
| 5. + 2 OpenMP threads | 133,500 | 23× — no gain at batch 64 |

Other measurements (`results/bench.csv`):
* **Batch size × threads** (img/s, 1 thread → 2 threads): batch 64: 141k → 132k; batch 128: 192k → 201k;
  **batch 256: 216k → 252k**; batch 512: 150k → 176k. Threading only pays once each matrix product is big enough.
  Bigger batches also converge more slowly per epoch (96.1% after 3 epochs at batch 256 vs. 97.2% at 64), so
  throughput alone isn't the right yardstick; time-to-accuracy is.
* **Wider net** (784-512-256-10, batch 256): 40.8k → 54.8k img/s with 2 threads.
* **float32 vs. float64**: 141k vs. 97k img/s (1.5×).
* **Inference**: 10k test images in ~11 ms (≈850k img/s) for 784-128-10; ~54 ms for 784-512-256-10.
* **Raw GEMM** (`bench_gemm`): the tiled kernel reaches ~170 GFLOP/s on one core vs. ~1.8 for the naive loop.

### Memory
| | Peak RSS |
|---|---|
| 784-128-10, dataset as uint8 converted per batch | **63 MiB** (was 94 MiB before the loader fix) |
| same, with a float32 copy of the dataset preloaded | 273 MiB, and no faster |
| 784-512-256-10 | 77 MiB (params + grads + momentum 6.1 MiB, batch workspaces 6.3 MiB) |

## Two bugs that performance measurement found
1. **The "optimized" kernel was the slowest.** The first register-blocked micro-kernel held its accumulators
   in a `T acc[MR][NR]` array. GCC kept that array in memory, and the assembly showed a stack load and store
   around *every* FMA (`vmovaps 1600(%rsp)` …), so it ran at ~4 GFLOP/s, 7× slower than the plain reordered
   loop. Declaring accumulators as SIMD vector types (`__attribute__((vector_size))`) let them live in
   registers: 4 → 171 GFLOP/s. Narrow edge tiles (the 10-wide softmax layer) are zero-padded into the same
   fast kernel (5.8 → 18.5 GFLOP/s there).
2. **Dataset loading used 2× the memory.** The loader read each file into a temporary buffer and copied it;
   reading straight into the final buffer cut peak RSS from 94 to 63 MiB.

## What did *not* help (and a hypothesis I haven't verified)
Cache blocking and threading at small batch size were no gain or a small loss. Threading the momentum update of
the 100k-parameter layer made training slower, so it is now enabled only for layers over 262k parameters. I suspect
the two cores bounce the optimizer state between their private L2 caches, but I did not confirm that with
hardware counters.

## Layout
| File | Role |
|---|---|
| `src/tensor.hpp` | 64-byte-aligned row-major matrix, tiled transpose |
| `src/gemm.hpp` | GEMM at four levels: naive, reordered, blocked, tiled SIMD + threads (`nn`, `tn`, `nt` variants) |
| `src/activations.hpp` | ReLU, sigmoid (overflow-safe), tanh, softmax (max-shifted), derivatives |
| `src/mlp.hpp` | Layers, forward, backprop, momentum SGD, per-phase profiler, memory accounting |
| `src/mnist.hpp` | IDX loader; images stay uint8 and are converted per batch |
| `src/neuron.cpp` | Stage 1: single neuron with analytic gradient check |
| `src/train_mlp.cpp` | Stage 2: training, validation split, confusion matrix, timing, RSS |
| `src/test_*.cpp`, `src/bench_gemm.cpp`, `scripts/bench.sh` | Tests and benchmarks |

## Not done
No model save/load, no regularization or augmentation, no convolutional layers, no GPU. Performance numbers
are for this machine; the SIMD width is chosen at compile time from the target ISA (AVX2 → 8 lanes).

## License

MIT, see [LICENSE](LICENSE).
