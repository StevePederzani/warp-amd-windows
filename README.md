# Warp on AMD, on Windows

A port of [NVIDIA Warp](https://github.com/NVIDIA/warp) 1.19 to AMD GPUs on Windows, through a HIP backend.

Warp is a Python framework for writing GPU kernels for simulation, graphics, robotics, and differentiable programming. Upstream
Warp requires CUDA, and AMD's own fork, [ROCm Warp](https://github.com/ROCm/warp), lists the Instinct MI325X data-center GPU as
its supported hardware. This port runs Warp on a consumer AMD GPU under Windows. I wanted machines like mine to be able to render,
train, and try out ideas such as quantum circuit emulation with Warp.

The port was developed and tested on a mini PC with an AMD Ryzen AI MAX+ 395, whose integrated Radeon 8060S GPU (gfx1151,
RDNA 3.5) shares 128 GB of memory with the CPU.

## Status

- Tested on the Radeon 8060S under Windows 11. Other RDNA 3 and 3.5 GPUs should work through AMD's generic target
  (gfx11-generic), but they have not been tested.
- Warp's own test suite is used to check the port. Tests that cannot run on this platform are skipped, each with its reason.
- This is an independent project, not affiliated with or endorsed by NVIDIA or AMD.

## Features

The port keeps Warp's API, and an AMD GPU appears as a `cuda` device. Beyond running Warp's kernels through HIP, it adds the
following, because HIP on Windows lacks them or handles them differently from CUDA.

- Graph capture and replay on Windows, including allocations made inside a capture. HIP on Windows has no graph memory nodes, so
  such allocations are made outside stream order and owned by the graph.
- `wp.capture_if` and `wp.capture_while` inside a capture. HIP has no conditional graph nodes, so the capture is split into
  segments, and the host evaluates each branch and loop condition between them, with the same semantics as a conditional node.
- Thread-block clusters, which CUDA supports on Hopper and Blackwell GPUs, emulated in software with NVIDIA's block numbering.
- Up to 227 KiB of shared memory per block, the limit on NVIDIA's largest GPUs. An AMD workgroup has 64 KiB of LDS, so whatever
  does not fit is placed in a per-module pool in device memory.
- Tile GEMM, FFT, and dense solver operations, which use NVIDIA's MathDx libraries under CUDA, generated as AMD device functions.
  Half and bfloat16 matrix multiplies use AMD's WMMA instructions.
- Correctly rounded float32 `exp` and `pow`, enabled with the module option `correctly_rounded`. Hard cases are resolved with
  Feynman's algorithm for logarithms in 128-bit fixed point. Results were checked against CORE-MATH on all 4,294,967,296 inputs
  for `exp` and on CORE-MATH's 129,187 hard-to-round pairs for `pow`.
- GPU memory sharing between processes (IPC) on Windows, through NT handles. Upstream Warp supports IPC on Linux only.
- One GPU shown as several Warp devices. `WARP_HIP_DEVICES=n` makes each AMD GPU appear as n devices (`cuda:0` to `cuda:n-1`), each
  with an equal share of the GPU's memory, its own memory pool and its own context, so code written for several GPUs runs on one, and
  Warp's own multi-GPU tests run on a single-GPU machine. Their kernels share the whole GPU: HIP on Windows cannot hold a stream to a
  part of a GPU's compute units. Two shares of one GPU always reach each other's memory, and as with two GPUs, work on one device's
  stream is ordered with another device's only by the caller.
- A pointer into a stream-ordered allocation is reported as pool memory, as CUDA's driver reports it. HIP does not answer CUDA's
  memory-pool pointer query, so the port keeps each such allocation's range.
- CPU arrays that GPU kernels can read and write directly on an integrated GPU such as the Radeon 8060S. Warp's CPU allocations
  come from pinned memory, which also made a 256 MiB copy to the GPU about four times faster in our measurement. Set
  `WARP_HIP_PINNED_CPU=0` to allocate CPU arrays from ordinary memory instead. A build with AddressSanitizer
  (`--sanitize=address`) allocates them from ordinary memory, so that AddressSanitizer can report an overflow past an array.
- Stream priorities. HIP on Windows runs a high-priority stream's work as it runs any other stream's, so the port schedules
  high-priority work ahead of lower-priority work itself. A copy within the GPU on a lower-priority stream runs as one kernel
  whose thread blocks wait while high-priority work runs, as a CUDA copy's thread blocks do, and other lower-priority work waits
  behind a gate. HIP gives a process 4 hardware queues by default and puts every stream past them on a shared queue, where a
  high-priority stream's work waits behind the other stream's, so the port raises the count to 16 (`GPU_MAX_HW_QUEUES`) unless it
  is already set. The count belongs to the process's HIP runtime: when Warp loads the runtime, a framework that shares it later,
  such as PyTorch's ROCm build, gets the same count, and when another framework loaded it first, the count is that runtime's.
- A kernel that traps, including Warp's own runtime checks such as a zero slice step, is reported as error 719 at the next
  synchronization and every one after it, and a failed device assert (such as a tile index out of bounds in debug mode) as error
  710, as on CUDA. HIP's own trap under Windows ends the kernel without reporting an error, so a failed check went unnoticed.
- `clock64()` and `clock()` in native functions (`wp.func_native`) count GPU cycles in 64 bits. On RDNA 3 and 3.5 GPUs (gfx11),
  HIP's cycle counter is 20 bits wide and wraps about every 0.36 ms on the Radeon 8060S, so a kernel that waits for a number of
  cycles to pass waited forever. The port computes the count from the GPU's 64-bit real-time counter instead.
- Kernel `printf` output that follows redirection of the process's stdout, managed memory on the APU, CPU kernel launches split
  across threads (`warp.config.cpu_threads`), and 3D textures larger than the hardware's 3D image limit.

## Requirements

- Windows 11
- Python 3.12
- Visual Studio 2026 Build Tools with the C++ workload (MSVC 14.51)
- Git
- AMD's ROCm SDK for Windows 10.1.0, installed from AMD's Python package index (see Build)
- An AMD Radeon GPU (tested on the Radeon 8060S, gfx1151)

## Build

In PowerShell, from a clone of this repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1151]==10.1.0"
rocm-sdk init
pip install numpy
python build_lib.py --rocm-path="$(rocm-sdk path --root)" --hip-arch=gfx1151 --no-use-libmathdx
pip install -e ".[examples]"
python warp\examples\core\example_quantum_circuit.py
```

The first build downloads Warp's prebuilt LLVM, which is used to compile CPU kernels. For a different GPU, change `--hip-arch` and
the `device-gfx1151` package to match it.

### rocPRIM Fix

The build includes a fix for a bug in AMD's rocPRIM 4.6.0, which ships with ROCm 10.1.0. When a radix sort's bit range ends at the
key's top bit, rocPRIM builds its bit mask with a shift by the key's full width (`T(1) << 64` for a 64-bit key sorted on bits 32
to 64). That shift is undefined behavior, the GPU wraps it around, and the sort compares the wrong bits. Warp's cuBQL BVH builder
sorts this way, so its trees came out wrong. The build copies rocPRIM's headers to `_build/rocprim-fixed`, corrects that line in
the copy, and compiles against the copy. The installed SDK is not modified, and a rocPRIM that is already fixed is used as it is.

## Usage

```python
import warp as wp

wp.init()  # lists the devices, with the AMD GPU as cuda:0


@wp.kernel
def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


x = wp.full(1024, 1.0, dtype=float, device="cuda:0")
y = wp.zeros(1024, dtype=float, device="cuda:0")
wp.launch(saxpy, dim=1024, inputs=[2.0, x, y], device="cuda:0")
print(y.numpy()[:4])  # [2. 2. 2. 2.]
```

[Warp's documentation](https://nvidia.github.io/warp/) applies, apart from the differences listed under Known Limitations.

## Testing

Warp's test suite runs as it does upstream. This runs every test module, with test classes in eight parallel processes:

```powershell
python -m warp.tests -s autodetect -j 8
```

## Test Results

Warp's own test suite (`warp/tests`) was run with its own parallel runner (`python -m warp.tests -s autodetect -j 8`, 254 test
classes in eight processes) on a clean clone and build of this repository (Windows 11, AMD Radeon 8060S (gfx1151), AMD's ROCm SDK
10.1.0 (HIP 7.16)). Its modules that use PyTorch were run again under AMD's ROCm build of PyTorch, and some modules were run again
in other configurations, each a row below.

| Run | Modules | Tests Run | Failures | Errors | Skipped |
|---|---|---|---|---|---|
| Warp's suite | 214 | 10,250 | 0 | 0 | 148 |
| Its PyTorch modules (ROCm PyTorch) | 7 | 343 | 0 | 0 | 4 |
| test_copy alone, with memory free | 1 | 36 | 0 | 0 | 0 |
| Pinned CPU arenas off (WARP_HIP_PINNED_CPU=0) | 3 | 185 | 0 | 0 | 10 |
| NumPy 2.3 | 1 | 63 | 0 | 0 | 7 |
| One GPU as two logical devices (WARP_HIP_DEVICES=2) | 9 | 258 | 0 | 0 | 16 |
| CPU-only build | 6 | 151 | 0 | 0 | 15 |
| AddressSanitizer build | 1 | 4 | 0 | 0 | 0 |

Warp's suite skipped 148 tests, each by its own check of what it needs. By what would let them run:

| What They Need | Tests | Passed in Another Row Above |
|---|---:|---:|
| NVIDIA Hardware | 25 | 0 |
| Linux | 78 | 0 |
| Two GPUs | 24 | 21 |
| Nothing: Warp Skips Them Everywhere | 5 | 0 |
| A Configuration of Their Own, Run Above | 12 | 12 |
| Nothing: Their Other Device's Copy Ran | 4 | 0 |

<details><summary>NVIDIA Hardware: 25 tests</summary>

No AMD GPU runs these, on any operating system.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.test_atomic_cas` | a spin lock built on atomic compare-and-swap, contended by the threads of one warp | 24 on the GPU: `test_cas_2d_float32`, `test_cas_2d_float64`, `test_cas_2d_int32`, `test_cas_2d_int64`, `test_cas_2d_uint32`, `test_cas_2d_uint64`, `test_cas_3d_float32`, `test_cas_3d_float64`, `test_cas_3d_int32`, `test_cas_3d_int64`, `test_cas_3d_uint32`, `test_cas_3d_uint64`, `test_cas_4d_float32`, `test_cas_4d_float64`, `test_cas_4d_int32`, `test_cas_4d_int64`, `test_cas_4d_uint32`, `test_cas_4d_uint64`, `test_cas_float32`, `test_cas_float64`, `test_cas_int32`, `test_cas_int64`, `test_cas_uint32`, `test_cas_uint64` | They need independent thread scheduling within a warp, which NVIDIA GPUs have from sm_70; an AMD wave runs its lanes in lockstep, and Warp skips these tests on older NVIDIA GPUs the same way. |  |
| `warp.tests.test_apic_load_from_bytes` | a compiled module loaded from bytes | 1 on the GPU: `test_load_from_bytes_ptx_module` | It loads a PTX module, NVIDIA's intermediate code, which AMD GPUs cannot run. |  |

</details>

<details><summary>Linux: 78 tests</summary>

Each needs a library built for AMD GPUs, or an operating system mechanism, that Linux has and Windows does not.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.interop.test_jax` | Warp kernels called from JAX (jax_kernel, jax_callable, the XLA FFI), their autodiff, graph caching and multi-device runs | 57 on the GPU: `test_bf16_interop_jax`, `test_device_conversion`, `test_ffi_callback`, `test_ffi_jax_callable_cache_argnames`, `test_ffi_jax_callable_cache_stage_argnames`, `test_ffi_jax_callable_graph_cache`, `test_ffi_jax_callable_graph_replay_skips_module_load`, `test_ffi_jax_callable_in_out`, `test_ffi_jax_callable_scale_constant`, `test_ffi_jax_callable_scale_static`, `test_ffi_jax_cuda_requires_cuda_support`, `test_ffi_jax_kernel_add`, `test_ffi_jax_kernel_autodiff_2d`, `test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output`, `test_ffi_jax_kernel_autodiff_jit_of_grad_simple`, `test_ffi_jax_kernel_autodiff_mat22`, `test_ffi_jax_kernel_autodiff_multi_output`, `test_ffi_jax_kernel_autodiff_per_call_override_rejected`, `test_ffi_jax_kernel_autodiff_simple`, `test_ffi_jax_kernel_autodiff_static_required`, `test_ffi_jax_kernel_autodiff_vec2`, `test_ffi_jax_kernel_block_dim_autodiff`, `test_ffi_jax_kernel_block_dim_tile`, `test_ffi_jax_kernel_cache_argnames`, `test_ffi_jax_kernel_diagonal`, `test_ffi_jax_kernel_in_out`, `test_ffi_jax_kernel_launch_dims_autodiff_basic`, `test_ffi_jax_kernel_launch_dims_autodiff_gradient`, `test_ffi_jax_kernel_launch_dims_autodiff_separate_cache`, `test_ffi_jax_kernel_launch_dims_autodiff_vmap`, `test_ffi_jax_kernel_launch_dims_custom`, `test_ffi_jax_kernel_launch_dims_default`, `test_ffi_jax_kernel_output_dims_autodiff_still_blocked`, `test_ffi_jax_kernel_rejects_oversized_explicit_scalar_tid_launch_dims`, `test_ffi_jax_kernel_rejects_oversized_explicit_tuple_tid_launch_dims`, `test_ffi_jax_kernel_rejects_oversized_inferred_scalar_tid_launch_dims`, `test_ffi_jax_kernel_rejects_oversized_inferred_tuple_tid_launch_dims`, `test_ffi_jax_kernel_scale_vec_constant`, `test_ffi_jax_kernel_scale_vec_static`, `test_ffi_jax_kernel_sincos`, `test_ffi_jax_kernel_subscript_autodiff`, `test_ffi_jax_kernel_subscript_scalar`, `test_ffi_jax_kernel_subscript_vec`, `test_ffi_jax_kernel_validates_all_target_block_dims_during_tracing`, `test_ffi_jax_mixed_devices`, `test_ffi_vmap_add_broadcast_all`, `test_ffi_vmap_add_sequential`, `test_ffi_vmap_lookup_broadcast_all`, `test_ffi_vmap_lookup_sequential`, `test_ffi_vmap_rowsum_broadcast_all`, `test_ffi_vmap_rowsum_sequential`, `test_jax_kernel_accepts_oversized_compile_time_dead_scalar_tid_launch_dims_ffi`, `test_jax_kernel_basic_ffi`, `test_jax_kernel_launch_dims_ffi`, `test_jax_kernel_multiarg_ffi`, `test_jax_kernel_scalar_ffi`, `test_jax_kernel_vecmat_ffi` | They need JAX on the AMD GPU, and JAX has no build for AMD GPUs on Windows (their CPU cases ran). |  |
| `warp.tests.interop.test_jax` | Warp kernels called from JAX (jax_kernel, jax_callable, the XLA FFI), their autodiff, graph caching and multi-device runs | 1: `test_device_lookup_jax_distributed_cpu` | It needs JAX's multi-process CPU backend, which runs on Linux and macOS only. |  |
| `warp.tests.interop.test_paddle` | arrays shared with Paddle: conversion, zero-copy, autograd | 10 on the GPU: `test_array_ctype_from_paddle`, `test_device_conversion`, `test_direct`, `test_from_paddle`, `test_from_paddle_slices`, `test_from_paddle_zero_strides`, `test_paddle_autograd`, `test_paddle_retain_grad_from_paddle`, `test_paddle_zerocopy`, `test_to_paddle` | They need Paddle on the AMD GPU, and Paddle has no build for AMD GPUs on Windows (their CPU cases ran). |  |
| `warp.tests.interop.test_dlpack` | arrays shared without a copy with other frameworks through DLPack | 4 on the GPU: `test_dlpack_jax_to_warp`, `test_dlpack_jax_to_warp_v2`, `test_dlpack_warp_to_jax`, `test_dlpack_warp_to_jax_v2` | They need JAX on the AMD GPU, and JAX has no build for AMD GPUs on Windows (their CPU cases ran). |  |
| `warp.tests.interop.test_dlpack` | arrays shared without a copy with other frameworks through DLPack | 3 on the GPU: `test_dlpack_paddle_to_warp`, `test_dlpack_warp_to_paddle`, `test_dlpack_warp_to_paddle_v2` | They need Paddle on the AMD GPU, and Paddle has no build for AMD GPUs on Windows (their CPU cases ran). |  |
| `warp.tests.test_allocator` | custom memory allocators, per device | 1: `test_rmm_allocator_multi_gpu` | It needs RAPIDS rmm, which is Linux-only. |  |
| `warp.tests.test_apic` | saving and loading a capture's modules and graphs | 1: `test_capture_save_source_fallback_detects_source_changes (its "replace during read" case)` | It replaces a file that is open for reading, which Windows does not allow. |  |
| `warp.tests.test_core_library_binary` | the native library's binary | 1: `test_linux_core_library_tls_allocation_size` | It covers a Linux-only mechanism (ELF thread-local storage). |  |

</details>

<details><summary>Two GPUs: 24 tests</summary>

The test machine has one GPU.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.cuda.test_unified_memory` | which memory each device may read: pools, peers, managed and CPU memory | 5: `test_unified_memory_checked_warns_for_unowned_cuda_mempool_pointer`, `test_unified_memory_device_can_access_uses_mempool_state_when_target_mempools_enabled`, `test_unified_memory_verify_rejects_mempool_allocation_without_mempool_access`, `test_unified_memory_verify_uses_mempool_access_for_cuda_mempool_allocations`, `test_unified_memory_verify_uses_parent_allocator_for_cuda_mempool_slices` | They need two GPUs that can reach each other's memory pools (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_unified_memory` | which memory each device may read: pools, peers, managed and CPU memory | 2: `test_unified_memory_verify_uses_parent_allocator_for_default_cuda_slices`, `test_unified_memory_verify_uses_peer_access_for_default_cuda_allocations` | They need two GPUs that can read each other's memory (peer access; the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_unified_memory` | which memory each device may read: pools, peers, managed and CPU memory | 1 on the GPU: `test_unified_memory_managed_array_cross_device_graph_capture` | It needs two GPUs (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_multigpu` | work scoped to one GPU and moved between two | 5: `test_multigpu_nesting`, `test_multigpu_pingpong`, `test_multigpu_pingpong_streams`, `test_multigpu_scoped_device`, `test_multigpu_set_device` | They need two GPUs (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_mempool` | memory pools, and one GPU's access to another's pool | 1: `test_mempool_access` | It needs two GPUs that can reach each other's memory pools (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_mempool` | memory pools, and one GPU's access to another's pool | 1: `test_mempool_access_exceptions_unsupported` | It needs two GPUs that cannot reach each other's memory (the test machine has one). |  |
| `warp.tests.cuda.test_peer` | one GPU reading and writing another's memory (peer access) | 1: `test_peer_access` | It needs two GPUs that can read each other's memory (peer access; the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.cuda.test_peer` | one GPU reading and writing another's memory (peer access) | 1: `test_peer_access_exceptions_unsupported` | It needs two GPUs that cannot reach each other's memory (the test machine has one). |  |
| `warp.tests.cuda.test_streams` | streams, events and graphs across GPUs | 2: `test_stream_arg_graph_mgpu`, `test_stream_scope_graph_mgpu` | They need two GPUs (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.test_allocator` | custom memory allocators, per device | 2: `test_per_device_isolation`, `test_set_cuda_allocator_broadcasts_to_all_devices` | They need two GPUs (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.test_graph` | CUDA graph capture and launch | 1: `test_graph_launch_array_access_mode_checked_mempool_access_cuda_capture` | It needs two GPUs that can reach each other's memory pools (the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.test_graph` | CUDA graph capture and launch | 1: `test_graph_launch_array_access_mode_checked_peer_access_cuda_capture` | It needs two GPUs that can read each other's memory (peer access; the test machine has one). | all, passed in "One GPU as two logical devices (WARP_HIP_DEVICES=2)" above |
| `warp.tests.interop.test_jax` | Warp kernels called from JAX (jax_kernel, jax_callable, the XLA FFI), their autodiff, graph caching and multi-device runs | 1: `test_device_lookup_jax_distributed_cuda` | It needs two GPUs (the test machine has one). |  |

</details>

<details><summary>Nothing: Warp Skips Them Everywhere: 5 tests</summary>

Warp's own tests skip these on every platform.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.interop.test_jax` | Warp kernels called from JAX (jax_kernel, jax_callable, the XLA FFI), their autodiff, graph caching and multi-device runs | 4: `test_ffi_jax_callable_pmap_mul`, `test_ffi_jax_callable_pmap_multi_output`, `test_ffi_jax_callable_pmap_multi_stage`, `test_ffi_jax_kernel_autodiff_pmap_multi_output` | They are skipped by Warp on every platform (flaky, and they need two GPUs). |  |
| `warp.tests.test_graph` | CUDA graph capture and launch | 1 on the GPU: `test_cuda_graph_alloc_retained_release` | It is skipped by Warp on every platform as flaky. |  |

</details>

<details><summary>A Configuration of Their Own, Run Above: 12 tests</summary>

Each needs a configuration of its own, and ran in its row of the table above.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.test_sanitize` | AddressSanitizer's checks on CPU kernels | 4: `test_cpu_block_fibers_rejected`, `test_heap_buffer_overflow_detected`, `test_inbounds_runs_clean`, `test_logical_bound_overflow_detected` | They need the AddressSanitizer build (run in its own row above). | all, passed in "AddressSanitizer build" above |
| `warp.tests.test_copy` | array copies | 2: `test_copy_large_stride_cpu_cuda0`, `test_copy_large_stride_cuda0_cpu` | They need 12 GB free on the GPU and the CPU at once, which the parallel run had in use (run alone in its own row above). | all, passed in "test_copy alone, with memory free" above |
| `warp.tests.test_examples` | Warp's example programs | 2 on the CPU and GPU: `test_core.example_torch` | They need PyTorch (run in the PyTorch row above). | all, passed in "Its PyTorch modules (ROCm PyTorch)" above |
| `warp.tests.cuda.test_unified_memory` | which memory each device may read: pools, peers, managed and CPU memory | 1 on the GPU: `test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported` | It needs a GPU that cannot read CPU memory (run with WARP_HIP_PINNED_CPU=0 in its own row above). | all, passed in "Pinned CPU arenas off (WARP_HIP_PINNED_CPU=0)" above |
| `warp.tests.test_allocator` | custom memory allocators, per device | 1 on the GPU: `test_torch_caching_allocator` | It needs PyTorch (run in the PyTorch row above). | all, passed in "Its PyTorch modules (ROCm PyTorch)" above |
| `warp.tests.test_cuda_profiler` | the CUDA profiler hooks | 1: `test_noop_on_cpu_build` | It needs Warp built without GPU support (run in its own row above). | all, passed in "CPU-only build" above |
| `warp.tests.test_types` | Warp's types and their NumPy conversions | 1: `test_numpy_dtype_legacy_fallback` | It needs NumPy older than 2.4 (run with NumPy 2.3 in its own row above). | all, passed in "NumPy 2.3" above |

</details>

<details><summary>Nothing: Their Other Device's Copy Ran: 4 tests</summary>

Each is one device's copy of a test whose other copy ran.

| Module | What It Checks | Tests | Why They Skip | Ran Elsewhere |
|---|---|---|---|---|
| `warp.tests.test_graph` | CUDA graph capture and launch | 3 on the GPU: `test_graph_copy_noncontiguous_cpu_rejected`, `test_graph_fill_indexed_cpu_rejected`, `test_graph_fill_noncontiguous_cpu_rejected` | They are for the CPU device only; their GPU cases are covered by other tests. |  |
| `warp.tests.test_indexedarray` | indexed arrays | 1: `test_indexedarray_in_struct_to_device_transfer_cpu` | It is the CPU copy of a GPU-only test, which skips itself (the GPU copy ran). |  |

</details>

The PyTorch run skipped 4: 2 need two GPUs (the test machine has one); 1 is the CPU copy of a GPU-only test, which skips itself
(the GPU copy ran); 1 needs RAPIDS rmm, which is Linux-only.

The correctly rounded `exp` and `pow` were checked against CORE-MATH on this build: `exp` on 4,294,967,296 inputs on the GPU and
the CPU, with 0 differences; `pow` on 201,455,779 inputs on the GPU and the CPU, with 0 differences.

## Known Limitations

- Only the Radeon 8060S under Windows 11 has been tested, on a machine with a single GPU. Tests that need two GPUs ran on that GPU
  shown as two logical devices (their row in Test Results); the ones that need two GPUs unable to reach each other's memory are
  skipped.
- AMD GPUs do not run PTX. A request for PTX output returns the AMD code object instead.
- GPU kernels cannot access CPU memory that Warp did not allocate, such as a NumPy array wrapped without a copy, because HIP on
  Windows does not support pageable memory access. A launch that passes such an array to a GPU kernel is refused with an error.
- Warp's interoperability with JAX and Paddle works on the CPU only. Neither JAX nor Paddle has a build for AMD GPUs on Windows.

## Credits

This port builds on NVIDIA Warp 1.19.0 and on AMD's ROCm Warp, whose HIP backend it carries to Warp 1.19 and to Windows. Both are
licensed under Apache-2.0. NOTICE.md lists who wrote which parts.

NVIDIA and Warp are trademarks of NVIDIA. AMD, Radeon, Ryzen, and ROCm are trademarks of AMD. They are used here only to describe
compatibility.

## License

Apache-2.0. See LICENSE.md.

## Contact

Steve Pederzani, steve.pederzani@gmail.com
