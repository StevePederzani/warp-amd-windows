# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

"""Tile kernels that need more shared memory per block than an AMD GPU's LDS holds (64 KiB per workgroup).

On AMD GPUs such a kernel keeps its block's shared memory at two levels, on chip as far as its launch's LDS allows and the rest
in a pool of slots in device memory. This is virtual shared memory (``vsm_place`` under WP_HIP_VSM in tile.h, ``_hip_vsm_plan``
in context.py). Every test checks the results exactly against NumPy. On AMD GPUs each test also checks whether the kernel really
was planned, meaning its launch asks for its on-chip tier (the LDS minus its static LDS) and its slot holds what it needs, or
really was not. On NVIDIA GPUs the same kernels run in real shared memory where it is large enough, and are skipped where it is
not.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64


def _fits(device, nbytes: int) -> bool:
    return device.max_shared_memory_per_block >= nbytes + 4 * 1024


def _hooks(kernel, device, block_dim=BLOCK_DIM):
    module_exec = kernel.module.load(device, block_dim)
    return module_exec, module_exec.get_kernel_hooks(kernel)


def _planned(module_exec, kernel, direction="forward") -> int:
    """Return what the kernel holds in its slot, as its module's metadata records it, or 0 for real LDS."""
    name = (
        wp._src.codegen.cuda_kernel_forward_name(kernel)
        if direction == "forward"
        else wp._src.codegen.cuda_kernel_backward_name(kernel)
    )
    return int(module_exec.meta.get(name + "_vsm_bytes", 0))


def _tier(module_exec, kernel, direction="forward") -> int:
    """Return the on-chip tier a planned kernel's launch asks for, the device's LDS minus the kernel's static LDS."""
    name = (
        wp._src.codegen.cuda_kernel_forward_name(kernel)
        if direction == "forward"
        else wp._src.codegen.cuda_kernel_backward_name(kernel)
    )
    return int(module_exec.meta.get(name + "_vsm_lds", 0))


def _check_tier(test, module_exec, kernel, launch_bytes, direction="forward"):
    tier = _tier(module_exec, kernel, direction)
    test.assertGreater(tier, 0)
    test.assertLess(tier, 65536)
    test.assertEqual(launch_bytes, tier)


def test_vsm_past_lds_forward(test, device):
    """A 128x128 float tile (64 KiB) plus the allocator's counters exceeds the LDS, and the result is exact."""
    if not _fits(device, 128 * 128 * 4):
        test.skipTest("Requires more than 64 KiB of shared memory per block")

    @wp.kernel(module="unique", enable_backward=False)
    def double_tile(src: wp.array2d[float], dst: wp.array2d[float]):
        i = wp.tid()
        tile = wp.tile_load(src, shape=(128, 128), offset=(i * 128, 0), storage="shared")
        wp.tile_store(dst, wp.tile_map(wp.add, tile, tile), offset=(i * 128, 0))

    blocks = 300  # Many more blocks than slots on any GPU, so slots are claimed, freed, and claimed again.
    values = np.random.default_rng(1).standard_normal((blocks * 128, 128)).astype(np.float32)
    src = wp.array(values, device=device)
    dst = wp.zeros_like(src)
    module_exec, hooks = _hooks(double_tile, device)
    wp.launch_tiled(double_tile, dim=[blocks], inputs=[src, dst], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_array_equal(dst.numpy(), values * 2.0)
    if device.is_hip:
        test.assertGreater(_planned(module_exec, double_tile), 65536 - 4 * BLOCK_DIM)
        _check_tier(test, module_exec, double_tile, hooks.forward_launch_smem_bytes)
        test.assertEqual(hooks.forward_smem_bytes, _planned(module_exec, double_tile))


def test_vsm_hopper_sized(test, device):
    """216 KiB of tiles per block, close to the most NVIDIA's largest parts allow, with two tiles live at once, read crosswise."""
    if not _fits(device, 2 * 64 * 432 * 4):
        test.skipTest("Requires 216 KiB of shared memory per block")

    @wp.kernel(module="unique", enable_backward=False)
    def mix(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        ta = wp.tile_load(a, shape=(64, 432), offset=(i * 64, 0), storage="shared")
        tb = wp.tile_load(b, shape=(64, 432), offset=(i * 64, 0), storage="shared")
        wp.tile_store(out, wp.tile_map(wp.sub, ta, wp.tile_map(wp.mul, tb, tb)), offset=(i * 64, 0))

    blocks = 40
    rng = np.random.default_rng(2)
    a_np = rng.standard_normal((blocks * 64, 432)).astype(np.float32)
    b_np = rng.standard_normal((blocks * 64, 432)).astype(np.float32)
    a, b = wp.array(a_np, device=device), wp.array(b_np, device=device)
    out = wp.zeros_like(a)
    module_exec, hooks = _hooks(mix, device)
    wp.launch_tiled(mix, dim=[blocks], inputs=[a, b, out], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_allclose(out.numpy(), a_np - b_np * b_np, rtol=1e-6, atol=1e-6)
    if device.is_hip:
        test.assertGreaterEqual(_planned(module_exec, mix), 2 * 64 * 432 * 4)


def test_vsm_backward_only(test, device):
    """A forward kernel that fits in the LDS and a backward kernel that exceeds it, in one module."""
    if not _fits(device, 2 * 64 * 128 * 4):
        test.skipTest("Requires more than 64 KiB of shared memory per block")

    @wp.kernel(module="unique")
    def copy_tile(src: wp.array2d[float], dst: wp.array2d[float]):
        tile = wp.tile_load(src, shape=(64, 128), storage="shared")
        wp.tile_store(dst, tile)

    values = np.arange(64 * 128, dtype=np.float32).reshape(64, 128)
    src = wp.zeros((64, 128), dtype=float, device=device, requires_grad=True)
    dst = wp.zeros_like(src)
    src_grad = wp.zeros_like(src)
    dst_grad = wp.array(values, device=device)
    module_exec, hooks = _hooks(copy_tile, device)
    wp.launch_tiled(copy_tile, dim=[1], inputs=[src, dst], block_dim=BLOCK_DIM, device=device)
    wp.launch_tiled(
        copy_tile,
        dim=[1],
        inputs=[src, dst],
        adj_inputs=[src_grad, dst_grad],
        adjoint=True,
        block_dim=BLOCK_DIM,
        device=device,
    )
    np.testing.assert_array_equal(src_grad.numpy(), values)
    if device.is_hip:
        test.assertEqual(_planned(module_exec, copy_tile, "forward"), 0)
        test.assertGreater(hooks.forward_smem_bytes, 0)
        test.assertGreater(_planned(module_exec, copy_tile, "backward"), 65536 - 4 * BLOCK_DIM)
        _check_tier(test, module_exec, copy_tile, hooks.backward_launch_smem_bytes, "backward")
        test.assertEqual(hooks.backward_smem_bytes, _planned(module_exec, copy_tile, "backward"))


def test_vsm_two_levels(test, device):
    """Three 24 KiB tiles live at once (72 KiB). On AMD GPUs the first two fit in the block's on-chip tier and the third goes to
    its slot, so both levels are used in one kernel, read crosswise and exact, over many more blocks than slots."""
    if not _fits(device, 3 * 48 * 128 * 4):
        test.skipTest("Requires more than 64 KiB of shared memory per block")

    @wp.kernel(module="unique", enable_backward=False)
    def three(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        ta = wp.tile_load(a, shape=(48, 128), offset=(i * 48, 0), storage="shared")
        tb = wp.tile_load(b, shape=(48, 128), offset=(i * 48, 0), storage="shared")
        tc = wp.tile_load(c, shape=(48, 128), offset=(i * 48, 0), storage="shared")
        wp.tile_store(out, wp.tile_map(wp.sub, wp.tile_map(wp.mul, ta, tb), tc), offset=(i * 48, 0))

    blocks = 500
    rng = np.random.default_rng(5)
    a_np, b_np, c_np = (rng.standard_normal((blocks * 48, 128)).astype(np.float32) for _ in range(3))
    a, b, c = (wp.array(x, device=device) for x in (a_np, b_np, c_np))
    out = wp.zeros_like(a)
    module_exec, hooks = _hooks(three, device)
    wp.launch_tiled(three, dim=[blocks], inputs=[a, b, c, out], block_dim=BLOCK_DIM, device=device)
    # a * b - c contracts to one fused multiply-add on the GPU, with one rounding instead of NumPy's two, so compare to float32 precision.
    np.testing.assert_allclose(out.numpy(), a_np * b_np - c_np, rtol=1e-6, atol=1e-6)
    if device.is_hip:
        test.assertGreaterEqual(_planned(module_exec, three), 3 * 48 * 128 * 4)
        _check_tier(test, module_exec, three, hooks.forward_launch_smem_bytes)
        # The on-chip tier holds two of the three tiles.
        test.assertGreaterEqual(_tier(module_exec, three), 2 * 48 * 128 * 4)
        test.assertLess(_tier(module_exec, three), 3 * 48 * 128 * 4)


def test_vsm_mixed_module(test, device):
    """Two kernels in one module, one that fits in the LDS and one that exceeds it, each exact and launched in turn many times."""
    if not _fits(device, 128 * 132 * 4):
        test.skipTest("Requires more than 64 KiB of shared memory per block")

    @wp.kernel(enable_backward=False)
    def small(src: wp.array2d[float], dst: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(src, shape=(32, 128), offset=(i * 32, 0), storage="shared")
        wp.tile_store(dst, wp.tile_map(wp.neg, t), offset=(i * 32, 0))

    @wp.kernel(enable_backward=False)
    def large(src: wp.array2d[float], dst: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(src, shape=(132, 128), offset=(i * 132, 0), storage="shared")
        wp.tile_store(dst, wp.tile_map(wp.add, t, t), offset=(i * 132, 0))

    small_np = np.random.default_rng(3).standard_normal((32 * 50, 128)).astype(np.float32)
    large_np = np.random.default_rng(4).standard_normal((132 * 50, 128)).astype(np.float32)
    s_src, l_src = wp.array(small_np, device=device), wp.array(large_np, device=device)
    s_dst, l_dst = wp.zeros_like(s_src), wp.zeros_like(l_src)
    for _ in range(5):
        wp.launch_tiled(small, dim=[50], inputs=[s_src, s_dst], block_dim=BLOCK_DIM, device=device)
        wp.launch_tiled(large, dim=[50], inputs=[l_src, l_dst], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_array_equal(s_dst.numpy(), -small_np)
    np.testing.assert_array_equal(l_dst.numpy(), large_np * 2.0)
    if device.is_hip:
        module_exec, _ = _hooks(large, device)
        test.assertEqual(_planned(module_exec, small), 0)
        test.assertGreater(_planned(module_exec, large), 65536)


def test_vsm_graph_capture(test, device):
    """Replay a captured graph with a kernel that uses a slot. The pool belongs to the module, so nothing is set at launch."""
    if not _fits(device, 128 * 128 * 4):
        test.skipTest("Requires more than 64 KiB of shared memory per block")

    @wp.kernel(module="unique", enable_backward=False)
    def scale_tile(src: wp.array2d[float], dst: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(src, shape=(128, 128), offset=(i * 128, 0), storage="shared")
        wp.tile_store(dst, wp.tile_map(wp.add, wp.tile_map(wp.add, t, t), t), offset=(i * 128, 0))

    values = np.random.default_rng(5).standard_normal((16 * 128, 128)).astype(np.float32)
    src = wp.array(values, device=device)
    dst = wp.zeros_like(src)
    wp.load_module(module=scale_tile.module, device=device, block_dim=BLOCK_DIM)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch_tiled(scale_tile, dim=[16], inputs=[src, dst], block_dim=BLOCK_DIM, device=device)
    for _ in range(3):
        dst.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(dst.numpy(), values * 3.0, rtol=1e-6)


def test_vsm_beyond_virtual_maximum(test, device):
    """A kernel that exceeds the device's maximum even with virtual shared memory reports the shortfall at launch."""
    if not device.is_hip:
        test.skipTest("The virtual maximum is the AMD port's")
    n = device.max_shared_memory_per_block // 4 + 1024

    @wp.kernel(module="unique", enable_backward=False)
    def too_big(out: wp.array[float]):
        t = wp.tile_zeros(shape=n, dtype=float, storage="shared")
        wp.tile_store(out, t)

    out = wp.zeros(n, dtype=float, device=device)
    with suppress_native_error_output(), test.assertRaises(RuntimeError):
        wp.launch_tiled(too_big, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)


class TestTileVSM(unittest.TestCase):
    pass


devices = get_selected_cuda_test_devices()
add_function_test(TestTileVSM, "test_vsm_past_lds_forward", test_vsm_past_lds_forward, devices=devices)
add_function_test(TestTileVSM, "test_vsm_hopper_sized", test_vsm_hopper_sized, devices=devices)
add_function_test(TestTileVSM, "test_vsm_backward_only", test_vsm_backward_only, devices=devices)
add_function_test(TestTileVSM, "test_vsm_two_levels", test_vsm_two_levels, devices=devices)
add_function_test(TestTileVSM, "test_vsm_mixed_module", test_vsm_mixed_module, devices=devices)
add_function_test(TestTileVSM, "test_vsm_graph_capture", test_vsm_graph_capture, devices=devices)
add_function_test(TestTileVSM, "test_vsm_beyond_virtual_maximum", test_vsm_beyond_virtual_maximum, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
