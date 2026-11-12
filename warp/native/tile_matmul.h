// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.
//
// tile_matmul: cooperative scalar GEMM and the tile_matmul / tile_matmul_acc
// entry templates.
//
// Performance note: the cooperative scalar GEMM (`partitioned_gemm::scalar_matmul`)
// is not solely a fallback for builds without libmathdx. For small tiles
// and certain shape/dtype combinations it outperforms cuBLASDx because it
// avoids the LTO compilation cost and the per-launch setup overhead
// inherent to libmathdx. Users can deliberately route a kernel through
// the scalar path on a libmathdx-enabled build by setting the module
// option `enable_mathdx_gemm=False` (or globally via
// `wp.config.enable_mathdx_gemm = False`). The crossover point is
// shape- and dtype-dependent -- benchmark your configuration.

#pragma once

#include "tile.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

// Whether a device pass calls linked GEMMs through the dispatch's function slots. NVIDIA's do with libmathdx, and AMD's do with
// the HIP device library (warp.cu, with the module compiled with WP_ENABLE_MATHDX=1). AMD GPUs never define __CUDA_ARCH__, so
// upstream's `!defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0` would keep them on the header GEMM.
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && WP_ENABLE_MATHDX
#define WP_MATHDX_GEMM_LINKED 1
#else
#define WP_MATHDX_GEMM_LINKED 0
#endif

namespace wp {

// TF32 input precision without cuBLASDx. AMD GPUs have no TF32 unit and no cuBLASDx, so MathDx's precision="tf32" is honored the
// way the format is defined. Each float32 operand is rounded to TF32's 10-bit mantissa as it is read (round to nearest even,
// dropping 13 bits, with infinities and NaNs kept), and the products are summed in float32. The dispatch passes this marker in
// the LTO function slots (tile_matmul in builtins.py) only on AMD GPUs with enable_mathdx_gemm on, since NVIDIA only honors it
// on cuBLASDx's path.
struct hip_tf32_t {
};
template <typename F> struct wp_is_hip_tf32 {
    static constexpr bool value = false;
};
template <> struct wp_is_hip_tf32<hip_tf32_t> {
    static constexpr bool value = true;
};

inline CUDA_CALLABLE float tf32_round(float f)
{
    unsigned int u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) != 0x7f800000u) {
        const unsigned int lsb = (u >> 13) & 1u;
        u = (u + 0x0fffu + lsb) & 0xffffe000u;
    }
    float r;
    memcpy(&r, &u, sizeof(r));
    return r;
}

template <bool TF32, typename T> inline CUDA_CALLABLE T tf32_if(T x)
{
    if constexpr (TF32)
        return T(tf32_round(float(x)));
    else
        return x;
}

namespace partitioned_gemm {

template <typename T> inline CUDA_CALLABLE const T& index(const T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <typename T> inline CUDA_CALLABLE T& index(T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <int PartitionM, int PartitionN, typename Tile> struct partition_t {
    static constexpr int M = PartitionM;
    static constexpr int N = PartitionN;
    static constexpr int Stride = Tile::Layout::Shape::dim(1);

    using T = typename Tile::Type;

    inline partition_t(Tile& A)
    {
        data = A.data.ptr;

        // todo: do ceil div for non-multiples of M,N
        shape[0] = Tile::Layout::Shape::dim(0) / PartitionM;
        shape[1] = Tile::Layout::Shape::dim(1) / PartitionN;
    }

    // underlying data
    T* data;

    // partition dimensions
    int shape[2];
};

template <typename Partition> inline int partition_size(const Partition& part) { return part.shape[0] * part.shape[1]; }

// returns the x, y coordinates of a tile given a linear index
template <typename Partition> inline void partition_coord(const Partition& part, const int t, int& i, int& j)
{
    i = t / part.shape[1];
    j = t % part.shape[1];
}

template <typename Partition> inline auto partition_load(const Partition& tile, int i, int j)
{
    mat_t<Partition::M, Partition::N, typename Partition::T> out;

    const int tile_i = i * Partition::M;
    const int tile_j = j * Partition::N;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            out.data[i][j] = partitioned_gemm::index(tile.data, tile_i + i, tile_j + j, Partition::Stride);
        }
    }

    return out;
}

template <typename Partition, typename Value>
inline void partition_store(const Partition& tile, int i, int j, const Value& value)
{
    const int tile_i = Partition::M * i;
    const int tile_j = Partition::N * j;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            index(tile.data, tile_i + i, tile_j + j, Partition::Stride) = value.data[i][j];
        }
    }
}


template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE void matmul(TileA& A, TileB& B, TileC& out)
{
    const int TILE_M = 4;
    const int TILE_N = 4;
    const int TILE_K = 4;

    auto A_tile = partition_t<TILE_M, TILE_K, TileA>(A);
    auto B_tile = partition_t<TILE_K, TILE_N, TileB>(B);
    auto C_tile = partition_t<TILE_M, TILE_N, TileC>(out);

    // static_assert(is_same<typename TileA::Type, typename TileB::Type>::value);

    const int length = partition_size(C_tile);

    for (int t = WP_TILE_THREAD_IDX; t < length; t += WP_TILE_BLOCK_DIM) {
        int i, j;
        partition_coord(C_tile, t, i, j);

        // accumulator
        auto sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k = 0; k < A_tile.shape[1]; k++) {
            const auto a = partition_load(A_tile, i, k);
            const auto b = partition_load(B_tile, k, j);

            sum += mul(a, b);
        }

        partition_store(C_tile, i, j, sum);
    }
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) && WP_TILE_WARP_SIZE == 32
// Matrix tiles on AMD's matrix instructions (WMMA), which is the part of MathDx's GEMM that AMD GPUs can run. RDNA 3 and 3.5
// multiply a 16x16 half (or bfloat16) matrix by another into a 16x16 float accumulator in one instruction of a 32-wide wave
// (v_wmma_f32_16x16x16_f16 and _bf16). The lane layout was measured against NumPy rather than assumed. For A, lane % 16 is the
// row and its 16 halves are the k values. For B, lane % 16 is the column. The accumulator's element e in lane l is row
// 2e + l / 16, column l % 16. A half or bfloat16 GEMM whose sizes are multiples of 16, in a block of whole waves, is computed
// here. Each wave takes 16x16 tiles of C in turn and reads the operands through their tiles' compile-time strides, so the
// backward pass's transposed views work unchanged. It accumulates in float, as MathDx does for half precision, and writes
// alpha * A B, plus beta * C when accumulating. As in BLAS, C is not read when beta is zero. Every other GEMM takes the scalar
// path below.
template <typename E> struct hip_wmma_kind {
    static constexpr int value = 0;
};
template <> struct hip_wmma_kind<half> {
    static constexpr int value = 1;
};
#ifndef WP_NO_BFLOAT16  // A module that uses no bfloat16 is compiled without the type.
template <> struct hip_wmma_kind<wp_bfloat16> {
    static constexpr int value = 2;
};
#endif
template <typename E> struct hip_wmma_kind<const E> : hip_wmma_kind<E> {};

template <
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename ElemA,
    typename ElemB,
    typename ElemC,
    typename T>
inline CUDA_CALLABLE void
hip_wmma_matmul(const ElemA* __restrict__ a_ptr, const ElemB* __restrict__ b_ptr, ElemC* __restrict__ c_ptr, T& alpha, T& beta)
{
    typedef _Float16 hip_v16h __attribute__((ext_vector_type(16)));
    typedef __bf16 hip_v16bf __attribute__((ext_vector_type(16)));
    typedef float hip_v8f __attribute__((ext_vector_type(8)));

    constexpr int M = LayoutC::Shape::dim(0);
    constexpr int N = LayoutC::Shape::dim(1);
    constexpr int K = LayoutA::Shape::dim(1);
    constexpr int sa0 = LayoutA::Stride::dim(0);
    constexpr int sa1 = LayoutA::Stride::dim(1);
    constexpr int sb0 = LayoutB::Stride::dim(0);
    constexpr int sb1 = LayoutB::Stride::dim(1);
    constexpr int sc0 = LayoutC::Stride::dim(0);
    constexpr int sc1 = LayoutC::Stride::dim(1);
    constexpr int tiles_n = N / 16;
    constexpr int tiles = (M / 16) * tiles_n;
    constexpr int waves = WP_TILE_BLOCK_DIM / 32;

    const int lane = WP_TILE_THREAD_IDX & 31;
    const int wave = WP_TILE_THREAD_IDX >> 5;
    const int l16 = lane & 15;
    const float alpha_f = float32(alpha);
    const float beta_f = float32(beta);
    const bool read_c = Accumulate && beta_f != 0.0f;

    // Every lane of a wave takes the same tiles, so each matrix instruction runs with its whole wave.
    for (int t = wave; t < tiles; t += waves) {
        const int row0 = (t / tiles_n) * 16;
        const int col0 = (t % tiles_n) * 16;
        hip_v8f acc = {};
        for (int k0 = 0; k0 < K; k0 += 16) {
            if constexpr (hip_wmma_kind<ElemA>::value == 1) {
                hip_v16h a, b;
                WP_PRAGMA_UNROLL
                for (int e = 0; e < 16; ++e) {
                    a[e] = __builtin_bit_cast(_Float16, a_ptr[(row0 + l16) * sa0 + (k0 + e) * sa1].u);
                    b[e] = __builtin_bit_cast(_Float16, b_ptr[(k0 + e) * sb0 + (col0 + l16) * sb1].u);
                }
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
            } else {
                hip_v16bf a, b;
                WP_PRAGMA_UNROLL
                for (int e = 0; e < 16; ++e) {
                    a[e] = __builtin_bit_cast(__bf16, a_ptr[(row0 + l16) * sa0 + (k0 + e) * sa1].u);
                    b[e] = __builtin_bit_cast(__bf16, b_ptr[(k0 + e) * sb0 + (col0 + l16) * sb1].u);
                }
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
            }
        }
        WP_PRAGMA_UNROLL
        for (int e = 0; e < 8; ++e) {
            const int idx = (row0 + 2 * e + (lane >> 4)) * sc0 + (col0 + l16) * sc1;
            const float v = alpha_f * acc[e];
            if (read_c)
                c_ptr[idx] = ElemC(v + beta_f * float32(c_ptr[idx]));
            else
                c_ptr[idx] = ElemC(v);
        }
    }
}
#endif  // HIP gfx11 wave32

// Register-blocked scalar GEMM with direct pointer arithmetic.
//
// Each thread computes a BM x BN sub-tile of C by iterating over K,
// loading BM values from A and BN values from B per step, and
// accumulating via an outer product into BM*BN registers.
//
// Optimizations:
//   - Direct pointer access with __restrict__ and compile-time strides
//     (bypasses tile_coord / index_from_coord abstraction in the hot loop)
//   - Precomputed row/column offsets outside the K loop
//   - Compile-time boundary elimination when M%BM==0 and N%BN==0
//   - K loop fully unrolled for small K (<=32), improving scheduling and
//     reducing branch overhead; left to compiler for large K to limit I-cache use
//   - Adaptive sub-tile size: 8x4, 4x4, 4x2, 2x2, or 1x1 based on parallelism
template <
    bool Accumulate,
    bool TF32,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul_impl(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    constexpr int M = LayoutC::Shape::dim(0);
    constexpr int N = LayoutC::Shape::dim(1);
    constexpr int K = LayoutA::Shape::dim(1);

    // Compile-time strides for direct pointer arithmetic
    constexpr int sa0 = LayoutA::Stride::dim(0);
    constexpr int sa1 = LayoutA::Stride::dim(1);
    constexpr int sb0 = LayoutB::Stride::dim(0);
    constexpr int sb1 = LayoutB::Stride::dim(1);
    constexpr int sc0 = LayoutC::Stride::dim(0);
    constexpr int sc1 = LayoutC::Stride::dim(1);

    // Use actual storage element types for pointer declarations.
    // A, B, C may have different element types in the backward pass
    // (e.g. adj_C is T_C*, B is T_B*). T is used only for the accumulator.
    using ElemA = typename remove_reference<decltype(A.ptr[0])>::type;
    using ElemB = typename remove_reference<decltype(B.ptr[0])>::type;
    using ElemC = typename remove_reference<decltype(C.ptr[0])>::type;

    // Direct pointer access with __restrict__ to enable compiler optimizations
    const ElemA* __restrict__ a_ptr = A.ptr;
    const ElemB* __restrict__ b_ptr = B.ptr;
    ElemC* __restrict__ c_ptr = C.ptr;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) && WP_TILE_WARP_SIZE == 32
    // Use the matrix instruction path for half or bfloat16 operands of one kind, sizes in whole 16x16 tiles, and a block of whole waves.
    if constexpr (
        hip_wmma_kind<ElemA>::value != 0 && hip_wmma_kind<ElemA>::value == hip_wmma_kind<ElemB>::value && M % 16 == 0
        && N % 16 == 0 && K % 16 == 0 && WP_TILE_BLOCK_DIM % 32 == 0
    ) {
        hip_wmma_matmul<Accumulate, LayoutA, LayoutB, LayoutC>(a_ptr, b_ptr, c_ptr, alpha, beta);
        return;
    }
#endif

    // Choose register sub-tile size to maximize effective throughput, balancing
    // arithmetic intensity (FMAs per shared-memory load) against thread
    // utilization.  Higher-intensity sub-tiles (>= 4x2, intensity >= 1.33) are
    // worth a modest utilization drop because the reduced memory traffic more
    // than compensates; we allow down to 75 % utilization for those.  For
    // low-intensity sub-tiles (2x2 / 1x1) we require full utilization since
    // their throughput relies on parallelism rather than reuse.
    constexpr int min_blocks_full = WP_TILE_BLOCK_DIM;
    constexpr int min_blocks_75 = (WP_TILE_BLOCK_DIM * 3 + 3) / 4;  // ceil(bd*3/4)
    constexpr int blocks_8x4 = ((M + 7) / 8) * ((N + 3) / 4);
    constexpr int blocks_4x4 = ((M + 3) / 4) * ((N + 3) / 4);
    constexpr int blocks_4x2 = ((M + 3) / 4) * ((N + 1) / 2);
    constexpr int blocks_2x2 = ((M + 1) / 2) * ((N + 1) / 2);
    constexpr int BM = (blocks_8x4 >= min_blocks_75) ? 8
        : (blocks_4x4 >= min_blocks_75)              ? 4
        : (blocks_4x2 >= min_blocks_75)              ? 4
        : (blocks_2x2 >= min_blocks_full)            ? 2
                                                     : 1;
    constexpr int BN = (BM == 8) ? 4 : (BM == 4 && blocks_4x4 >= min_blocks_75) ? 4 : (BM == 4) ? 2 : BM;

    // Number of sub-tile blocks covering the output (ceiling division)
    constexpr int blocks_m = (M + BM - 1) / BM;
    constexpr int blocks_n = (N + BN - 1) / BN;
    constexpr int num_blocks = blocks_m * blocks_n;

    // Whether boundary checks can be eliminated at compile time
    constexpr bool aligned_m = (M % BM == 0);
    constexpr bool aligned_n = (N % BN == 0);

    // As in BLAS, C is not read when beta is zero, so non-finite values in C
    // cannot reach the result through beta * C.
    const bool read_c = Accumulate && beta != T(0);

    for (int t = WP_TILE_THREAD_IDX; t < num_blocks; t += WP_TILE_BLOCK_DIM) {
        const int block_i = t / blocks_n;
        const int block_j = t % blocks_n;

        const int base_i = block_i * BM;
        const int base_j = block_j * BN;

        // Precompute base offsets for A rows and B columns (constant across K)
        int a_offsets[BM];
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++)
            a_offsets[si] = (base_i + si) * sa0;

        int b_offsets[BN];
        WP_PRAGMA_UNROLL
        for (int sj = 0; sj < BN; sj++)
            b_offsets[sj] = (base_j + sj) * sb1;

        // Accumulator in registers
        T sum[BM][BN];
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++)
            WP_PRAGMA_UNROLL
        for (int sj = 0; sj < BN; sj++)
            sum[si][sj] = T(0);

        // Reduction along K with register-blocked outer product.
        // For small K (<= 32), fully unroll to eliminate branch overhead and
        // enable better load/FMA scheduling.  For large K, leave unrolling to
        // the compiler to limit I-cache pressure.
        // The if-constexpr duplicates the body so the pragma applies correctly.
        if constexpr (K <= 32) {
            WP_PRAGMA_UNROLL
            for (int k = 0; k < K; k++) {
                const int ka = k * sa1;
                const int kb = k * sb0;

                T a_reg[BM];
                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++) {
                    if constexpr (aligned_m)
                        a_reg[si] = tf32_if<TF32>(T(a_ptr[a_offsets[si] + ka]));
                    else
                        a_reg[si] = (base_i + si < M) ? tf32_if<TF32>(T(a_ptr[a_offsets[si] + ka])) : T(0);
                }

                T b_reg[BN];
                WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++) {
                    if constexpr (aligned_n)
                        b_reg[sj] = tf32_if<TF32>(T(b_ptr[kb + b_offsets[sj]]));
                    else
                        b_reg[sj] = (base_j + sj < N) ? tf32_if<TF32>(T(b_ptr[kb + b_offsets[sj]])) : T(0);
                }

                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++)
                    WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++)
                    sum[si][sj] = muladd<T>(a_reg[si], b_reg[sj], sum[si][sj]);
            }
        } else {
            for (int k = 0; k < K; k++) {
                const int ka = k * sa1;
                const int kb = k * sb0;

                T a_reg[BM];
                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++) {
                    if constexpr (aligned_m)
                        a_reg[si] = tf32_if<TF32>(T(a_ptr[a_offsets[si] + ka]));
                    else
                        a_reg[si] = (base_i + si < M) ? tf32_if<TF32>(T(a_ptr[a_offsets[si] + ka])) : T(0);
                }

                T b_reg[BN];
                WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++) {
                    if constexpr (aligned_n)
                        b_reg[sj] = tf32_if<TF32>(T(b_ptr[kb + b_offsets[sj]]));
                    else
                        b_reg[sj] = (base_j + sj < N) ? tf32_if<TF32>(T(b_ptr[kb + b_offsets[sj]])) : T(0);
                }

                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++)
                    WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++)
                    sum[si][sj] = muladd<T>(a_reg[si], b_reg[sj], sum[si][sj]);
            }
        }

        // Store results with direct pointer arithmetic
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++) {
            WP_PRAGMA_UNROLL
            for (int sj = 0; sj < BN; sj++) {
                if constexpr (aligned_m && aligned_n) {
                    const int idx = (base_i + si) * sc0 + (base_j + sj) * sc1;
                    if (read_c)
                        c_ptr[idx] = ElemC(alpha * sum[si][sj] + beta * T(c_ptr[idx]));
                    else
                        c_ptr[idx] = ElemC(alpha * sum[si][sj]);
                } else {
                    if (base_i + si < M && base_j + sj < N) {
                        const int idx = (base_i + si) * sc0 + (base_j + sj) * sc1;
                        if (read_c)
                            c_ptr[idx] = ElemC(alpha * sum[si][sj] + beta * T(c_ptr[idx]));
                        else
                            c_ptr[idx] = ElemC(alpha * sum[si][sj]);
                    }
                }
            }
        }
    }
}

template <
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    scalar_matmul_impl<Accumulate, false, LayoutA, LayoutB, LayoutC>(A, B, C, alpha, beta);
}

// The same GEMM with TF32 input precision (hip_tf32_t above).
template <
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul_tf32(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    scalar_matmul_impl<Accumulate, true, LayoutA, LayoutB, LayoutC>(A, B, C, alpha, beta);
}

// One scalar GEMM, at TF32 input precision when its marker says so.
template <
    typename Marker,
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul_as(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    if constexpr (wp_is_hip_tf32<Marker>::value)
        scalar_matmul_tf32<Accumulate, LayoutA, LayoutB, LayoutC>(A, B, C, alpha, beta);
    else
        scalar_matmul<Accumulate, LayoutA, LayoutB, LayoutC>(A, B, C, alpha, beta);
}

}  // namespace partitioned_gemm


// tile_matmul: C = alpha * A @ B (does not read from C)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta>
CUDA_CALLABLE TileC& tile_matmul(
    Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, Alpha& alpha, Beta& beta
)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");

    using T = typename TileC::Type;

    T alphaT = T(alpha);
    T betaT = T(beta);

#if !WP_MATHDX_GEMM_LINKED
    // Use TF32 input precision when the dispatch's marker asks for it (hip_tf32_t), and the plain scalar GEMM otherwise.
    partitioned_gemm::scalar_matmul_as<Fwd, false, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
        A.data, B.data, C.data, alphaT, betaT
    );
#else
    // A GEMM the library doesn't link (a null slot or the TF32 marker) takes the header path at the requested precision.
    if constexpr (wp_is_null_func<Fwd>::value || wp_is_hip_tf32<Fwd>::value) {
        partitioned_gemm::scalar_matmul_as<Fwd, false, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
            A.data, B.data, C.data, alphaT, betaT
        );
    } else {
        fun_forward(&alphaT, A.data.ptr, B.data.ptr, &betaT, C.data.ptr);
    }
#endif

    WP_TILE_SYNC();

    return C;
}

// tile_matmul_acc: C = alpha * A @ B + beta * C (accumulates into C, does not read C when beta == 0)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta>
CUDA_CALLABLE TileC& tile_matmul_acc(
    Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, Alpha& alpha, Beta& beta
)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");

    using T = typename TileC::Type;

    T alphaT = T(alpha);
    T betaT = T(beta);

#if !WP_MATHDX_GEMM_LINKED
    // Use TF32 input precision when the dispatch's marker asks for it (hip_tf32_t), and the plain scalar GEMM otherwise.
    partitioned_gemm::scalar_matmul_as<Fwd, true, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
        A.data, B.data, C.data, alphaT, betaT
    );
#else
    // A GEMM the library doesn't link (a null slot or the TF32 marker) takes the header path at the requested precision.
    if constexpr (wp_is_null_func<Fwd>::value || wp_is_hip_tf32<Fwd>::value) {
        partitioned_gemm::scalar_matmul_as<Fwd, true, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
            A.data, B.data, C.data, alphaT, betaT
        );
    } else {
        fun_forward(&alphaT, A.data.ptr, B.data.ptr, &betaT, C.data.ptr);
    }
#endif

    WP_TILE_SYNC();

    return C;
}


// backward for tile_matmul_acc (the wp.tile_matmul(a, b, out) syntax)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
CUDA_CALLABLE void adj_tile_matmul_acc(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !WP_MATHDX_GEMM_LINKED
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    // Backward always accumulates into gradients (beta=1.0)
    // Each gradient's GEMM uses TF32 input precision when its own marker asks for it (hip_tf32_t).
    partitioned_gemm::scalar_matmul_as<
        AdjA, true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul_as<
        AdjB, true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    // A GEMM the library doesn't link (a null slot or the TF32 marker) takes the header path at the requested precision.
    if constexpr (wp_is_null_func<Fwd>::value || wp_is_hip_tf32<Fwd>::value) {
        auto At = tile_transpose(A);
        auto Bt = tile_transpose(B);

        partitioned_gemm::scalar_matmul_as<
            AdjA, true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
            adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
        );
        partitioned_gemm::scalar_matmul_as<
            AdjB, true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
            At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
        );
    } else {
        fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
        fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
    }
#endif

    // The matmuls read several adj_C slots per lane, while beta scaling below
    // modifies each lane's local slot. Make every read complete before any
    // cooperative CPU fiber begins the in-place scaling phase.
    WP_TILE_SYNC();

    if (T_C(beta) != T_C(1.0)) {
        for (int i = WP_TILE_THREAD_IDX; i < TileC::Layout::Size; i += WP_TILE_BLOCK_DIM)
            adj_C.grad(i) *= T_C(beta);
    }

    WP_TILE_SYNC();
}

// backward for tile_matmul (the out = wp.tile_matmul(a, b) syntax)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
CUDA_CALLABLE void adj_tile_matmul(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta,
    TileC& adj_ret
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !WP_MATHDX_GEMM_LINKED
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    // Backward always accumulates into gradients (beta=1.0)
    // Each gradient's GEMM uses TF32 input precision when its own marker asks for it (hip_tf32_t).
    partitioned_gemm::scalar_matmul_as<
        AdjA, true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul_as<
        AdjB, true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    // A GEMM the library doesn't link (a null slot or the TF32 marker) takes the header path at the requested precision.
    if constexpr (wp_is_null_func<Fwd>::value || wp_is_hip_tf32<Fwd>::value) {
        auto At = tile_transpose(A);
        auto Bt = tile_transpose(B);

        partitioned_gemm::scalar_matmul_as<
            AdjA, true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
            adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
        );
        partitioned_gemm::scalar_matmul_as<
            AdjB, true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
            At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
        );
    } else {
        fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
        fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
    }
#endif

    WP_TILE_SYNC();
}

}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
