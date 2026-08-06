# GTX 16xx (Turing without tensor cores): patches

The GTX 16xx cards (TU116/TU117) are Turing chips **without tensor cores**.
llama.cpp already detects them by name in `ggml-cuda.cu` and warns at startup,
but still dispatches them to the MMA kernels, which are then **emulated**. On a
GTX 1660 SUPER that costs roughly 60% of prompt-reading throughput.

The stock advice in that warning — build with
`CMAKE_CUDA_ARCHITECTURES=61-virtual` — is no longer possible with CUDA 13,
which removed Pascal support.

Two branches here, independent of each other:

| branch | what it is |
|---|---|
| `dp4a-fix-mmid-max-batch` | **bug fix**, affects any forward-JIT build, not only these cards |
| `sm75-no-turing-mma` | new `GGML_CUDA_NO_TURING_MMA` option, default OFF |

> Diagnosis and patches came out of a working session with AI agents
> (August 2026). Every claim below was measured on real hardware and every
> line reviewed. **No upstream PR is open** — take whatever is useful.

## 1. `dp4a-fix-mmid-max-batch` — the crash

Symptom, after ~26 requests:

```
CUDA error: invalid argument
  in ggml_cuda_kernel_launch at common.cuh
  in mul_mat_vec_q_switch_ncols_dst<(ggml_type)13>   // Q5_K
```

`get_mmvq_mmid_max_batch()` on the host selected the per-type table from the
**physical** compute capability, while the `__launch_bounds__` of
`mul_mat_vec_q_moe` are fixed by `__CUDA_ARCH__` at **compile** time. Running
PTX built for compute_61 on a Turing device, the host allowed up to 8 tokens
(Turing table) while the kernel carried Pascal launch bounds (5 warps x 32 =
160 threads for Q5_K), so a `MUL_MAT_ID` with 6-8 tokens launched 192-256
threads per block and CUDA rejected the launch.

**This is not a consequence of the architecture workaround.** Any build running
forward-JIT'd PTX can hit it — an sm_86-only build on Ada included. It was
reported independently as
[ggml-org/llama.cpp#24064](https://github.com/ggml-org/llama.cpp/issues/24064)
(GTX 1660 Ti, `ubatch 5..8`, same diagnosis) and closed by the stale bot
without a fix.

The patch derives the table from `ggml_cuda_highest_compiled_arch(cc)`,
mirroring the preprocessor selection in device code. Native builds are
unaffected: there the compiled arch equals the physical one.

An audit of all 13 host uses of `ggml_cuda_highest_compiled_arch`, of every
site using physical `cc` for launch geometry, and of the device-side
`__CUDA_ARCH__` guards found exactly **one other instance** of the same defect
(`allreduce.cu`, requires exactly 2 GPUs — left untouched and documented). No
silent-corruption paths were found.

## 2. `sm75-no-turing-mma` — the option

Adds `-DGGML_CUDA_NO_TURING_MMA=ON`. Default OFF; without it the build is
byte-for-byte what it was.

With it, and building natively for sm_75:

- `TURING_MMA_AVAILABLE` (device) and `turing_mma_available()` (host) are forced
  off **in sync**, so every kernel selection keyed on them (mmq, mmf, fattn,
  lightning-indexer) falls back to dp4a/vec/tile.
- The MMQ config selector picks the Pascal tables on both host and device,
  since the Ampere tables are tuned for the mma kernels.
- The startup warning is keyed on `turing_mma_available()`, so it disappears
  when the option is active — which doubles as the check that it worked.

The point is that the build stays **honest**: it targets the real architecture,
so every `__CUDA_ARCH__`-dependent launch bound is correct by construction, and
the fix in branch 1 is not needed.

## Measurements

GTX 1660 SUPER 6 GB (336 GB/s, 22 SM) · Ryzen 7 5700G · 32 GB DDR4-3600
(57.6 GB/s) · Debian 13, glibc 2.41 · Qwen3.6-35B-A3B UD-Q5_K_M with
`--n-cpu-moe 39`, KV cache q4_0, flash-attn.

Note the bottleneck: 39 of 40 expert layers live in system RAM, which is ~6x
slower than VRAM. The card sits at ~54 W of 125 even at 97% reported
utilisation. Everything below is bounded by that.

`llama-bench`, **9 repetitions** — fewer is not enough here: with `-r 3` the
same build reported 55.7 t/s and with `-r 9` it reported 86.4.

| build | prompt (pp512) | generation (tg64) |
|---|---|---|
| stock, emulated MMA | 86.4 ± 1.4 | 30.8 ± 0.3 |
| Vulkan | 107.7 ± 3.5 | 21.6 ± 0.1 |
| **CUDA 12.9, `61-virtual` (dp4a)** | **219.2 ± 5.1** | **30.8 ± 0.1** |
| CUDA 13.3, sm_75 + `NO_TURING_MMA` | 148.4 ± 45.4 | 31.3 ± 1.4 |

**2.5x on prompt reading, no loss on generation.** With the whole model
resident on the GPU (0.5B smoke test) the gap against emulated MMA is **3.4x**:
5809 vs 1728 t/s on pp512. On a real 71,256-token prompt, wall-clock went from
19.7 min to ~10 min.

The last row has 31% spread and was taken after hours of page-cache thrashing;
it is **not** a trustworthy comparison yet — same ballpark, needs re-measuring
on a cold machine.

Correctness, `test-backend-ops -b CUDA0`, on both branches:

```
MUL_MAT          1186/1186
MUL_MAT_ID        865/865
FLASH_ATTN_EXT   2884/2884
```

## Notes if you build CUDA 12.x yourself

Only relevant for the `61-virtual` route; `sm75-no-turing-mma` builds with a
stock CUDA 13 and GCC 15.

1. **CUDA 12.x against glibc 2.41** is widely reported as a blocker. It is not.
   glibc 2.41 declares `sinpi`, `cospi`, `sinpif`, `cospif` as `noexcept(true)`
   and CUDA declares them without it — **only the exception specifier differs**.
   Four lines in `crt/math_functions.h`.
2. **CUDA 12.9 caps at GCC 14**, and not because of the version check
   (`-allow-unsupported-compiler` gets past that): its frontend does not know
   the builtins GCC 15's libstdc++ uses (`__is_pointer`, `__array_rank`).
3. The redistributable tarballs use `lib/`, `nvcc` looks in `lib64/`. Symlink.
4. **`LD_LIBRARY_PATH` beats `RUNPATH`.** If it points at a different CUDA, the
   binaries silently load the wrong runtime with no error at all, and every
   measurement is garbage. Check with
   `env -u LD_LIBRARY_PATH ldd bin/llama-bench | grep cudart`.
