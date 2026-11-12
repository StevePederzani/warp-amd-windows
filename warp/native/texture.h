// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by AMD (ROCm Warp) and Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.

#pragma once

#include "builtin.h"

namespace wp {

// Data type constants (must match texture.cpp and Python side)
#define WP_TEXTURE_DTYPE_UINT8   0
#define WP_TEXTURE_DTYPE_UINT16  1
#define WP_TEXTURE_DTYPE_FLOAT32 2
#define WP_TEXTURE_DTYPE_INT8    3
#define WP_TEXTURE_DTYPE_INT16   4
#define WP_TEXTURE_DTYPE_FLOAT16 5
#define WP_TEXTURE_DTYPE_UINT32  6
#define WP_TEXTURE_DTYPE_INT32   7

// Filter mode constants
#define WP_TEXTURE_FILTER_CLOSEST 0
#define WP_TEXTURE_FILTER_LINEAR 1

// Address mode constants
#define WP_TEXTURE_ADDRESS_WRAP   0
#define WP_TEXTURE_ADDRESS_CLAMP  1
#define WP_TEXTURE_ADDRESS_MIRROR 2
#define WP_TEXTURE_ADDRESS_BORDER 3

// Maximum number of mipmap levels supported per texture.
// CUDA textures are capped at 16 levels (16384x16384 base resolution).
#define WP_TEXTURE_MAX_MIP_LEVELS 16

// Sentinel value for the ``lod`` argument of ``texture_sample`` meaning
// "no level-of-detail requested". When callers omit ``lod`` the builtin's
// default (``-1.0``) routes sampling through the original non-LOD code path,
// preserving the behavior of single-mip textures.
#define WP_TEXTURE_LOD_DISABLED (-1.0f)

// Helper function to get bytes per channel from dtype
inline int get_texture_bytes_per_channel(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
    case WP_TEXTURE_DTYPE_INT8:
        return 1;
    case WP_TEXTURE_DTYPE_UINT16:
    case WP_TEXTURE_DTYPE_INT16:
    case WP_TEXTURE_DTYPE_FLOAT16:
        return 2;
    case WP_TEXTURE_DTYPE_UINT32:
    case WP_TEXTURE_DTYPE_INT32:
    case WP_TEXTURE_DTYPE_FLOAT32:
        return 4;
    default:
        return 0;
    }
}

// Texture class for CPU or other hardware without texture units.
//
// Stores the full mipmap chain as a single contiguous buffer with per-level pointers.
// For a non-mipmapped texture (``num_mip_levels == 1``), ``data == mip_data[0]``.
struct Texture {

    // Unified constructor: allocates a contiguous buffer holding all mipmap levels
    // and initializes per-level pointers. Per-level shapes come from ``mip_widths``,
    // ``mip_heights`` and ``mip_depths`` (height/depth ignored for lower-dim textures).
    // If ``num_mip_levels == 1`` the texture is non-mipmapped and base dimensions are used.
    Texture(
        int32 ndim,
        int32 num_mip_levels,
        const int32* mip_widths,
        const int32* mip_heights,
        const int32* mip_depths,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 address_mode_u,
        int32 address_mode_v,
        int32 address_mode_w,
        bool use_normalized_coords
    )
        : data(nullptr)
        , width(mip_widths[0])
        , height(ndim > 1 ? mip_heights[0] : 0)
        , depth(ndim > 2 ? mip_depths[0] : 0)
        , num_channels(num_channels)
        , dtype(dtype)
        , filter_mode(filter_mode)
        , mip_filter_mode(mip_filter_mode)
        , num_mip_levels(num_mip_levels)
        , address_mode_u(address_mode_u)
        , address_mode_v(address_mode_v)
        , address_mode_w(address_mode_w)
        , use_normalized_coords(use_normalized_coords)
        , mip_data {}
        , mip_offsets {}
        , mip_widths_arr {}
        , mip_heights_arr {}
        , mip_depths_arr {}
    {
        const int bytes_per_channel = get_texture_bytes_per_channel(dtype);

        size_t total_bytes = 0;
        for (int level = 0; level < num_mip_levels; ++level) {
            mip_widths_arr[level] = mip_widths[level];
            mip_heights_arr[level] = ndim > 1 ? mip_heights[level] : 1;
            mip_depths_arr[level] = ndim > 2 ? mip_depths[level] : 1;

            const size_t level_bytes = size_t(mip_widths_arr[level]) * size_t(mip_heights_arr[level])
                * size_t(mip_depths_arr[level]) * size_t(num_channels) * size_t(bytes_per_channel);

            mip_offsets[level] = total_bytes;
            total_bytes += level_bytes;
        }

        uint8* buffer = new uint8[total_bytes];
        data = buffer;

        for (int level = 0; level < num_mip_levels; ++level) {
            mip_data[level] = buffer + mip_offsets[level];
        }
    }

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    ~Texture() { delete[] static_cast<uint8*>(data); }

    void* data = nullptr;  // base-level pointer, equals mip_data[0]
    int32 width = 0;  // base-level width
    int32 height = 0;  // base-level height
    int32 depth = 0;  // base-level depth
    int32 num_channels = 0;
    int32 dtype = 0;
    int32 filter_mode = 0;
    int32 mip_filter_mode = 0;  // mipmap level filter mode (nearest or linear between levels)
    int32 num_mip_levels = 0;
    int32 address_mode_u = 0;  // Per-axis address mode for U
    int32 address_mode_v = 0;  // Per-axis address mode for V
    int32 address_mode_w = 0;  // Per-axis address mode for W
    bool use_normalized_coords = false;  // If true, coords in [0,1]; if false, in texel space

    // Per-level metadata (owned by the texture; indexed [0, num_mip_levels))
    void* mip_data[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    size_t mip_offsets[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_widths_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_heights_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
    int32 mip_depths_arr[WP_TEXTURE_MAX_MIP_LEVELS] = {};
};

// Texture descriptor passed to kernels
// Contains the CUDA texture object handle (GPU) or pointer to cpu_texture*_data (CPU)
CUDA_CALLABLE inline uint32 texture_metadata(
    int32 num_channels,
    int32 dtype,
    int32 filter_mode,
    int32 mip_filter_mode,
    int32 num_mip_levels,
    int32 use_normalized_coords,
    int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
    int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
    int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
)
{
    return (uint32)num_channels | ((uint32)dtype << 3) | ((uint32)filter_mode << 6) | ((uint32)mip_filter_mode << 7)
        | ((uint32)num_mip_levels << 8) | ((uint32)use_normalized_coords << 13) | ((uint32)address_mode_u << 14)
        | ((uint32)address_mode_v << 16) | ((uint32)address_mode_w << 18);
}

#define WP_TEXTURE_METADATA_ACCESSORS                                                                                  \
    CUDA_CALLABLE inline int32 get_num_channels() const { return (int32)(metadata & 0x7u); }                           \
    CUDA_CALLABLE inline int32 get_dtype() const { return (int32)((metadata >> 3) & 0x7u); }                           \
    CUDA_CALLABLE inline int32 get_filter_mode() const { return (int32)((metadata >> 6) & 0x1u); }                     \
    CUDA_CALLABLE inline int32 get_mip_filter_mode() const { return (int32)((metadata >> 7) & 0x1u); }                 \
    CUDA_CALLABLE inline int32 get_num_mip_levels() const { return (int32)((metadata >> 8) & 0x1fu); }                 \
    CUDA_CALLABLE inline int32 get_use_normalized_coords() const { return (int32)((metadata >> 13) & 0x1u); } \
    CUDA_CALLABLE inline int32 get_address_mode_u() const { return (int32)((metadata >> 14) & 0x3u); } \
    CUDA_CALLABLE inline int32 get_address_mode_v() const { return (int32)((metadata >> 16) & 0x3u); } \
    CUDA_CALLABLE inline int32 get_address_mode_w() const { return (int32)((metadata >> 18) & 0x3u); }

struct texture1d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture1d_t()
        : tex(0)
        , width(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture1d_t(
        uint64 tex,
        int32 width,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

struct texture2d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    int32 height;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture2d_t()
        : tex(0)
        , width(0)
        , height(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture2d_t(
        uint64 tex,
        int32 width,
        int32 height,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , height(height)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

struct texture3d_t {
    uint64 tex;  // CUtexObject handle (GPU) or Texture* (CPU)
    int32 width;
    int32 height;
    int32 depth;
    uint32 metadata;
    WP_TEXTURE_METADATA_ACCESSORS

    CUDA_CALLABLE inline texture3d_t()
        : tex(0)
        , width(0)
        , height(0)
        , depth(0)
        , metadata(texture_metadata(0, 0, 0, 0, 1, 1))
    {
    }

    CUDA_CALLABLE inline texture3d_t(
        uint64 tex,
        int32 width,
        int32 height,
        int32 depth,
        int32 num_channels,
        int32 dtype,
        int32 filter_mode,
        int32 mip_filter_mode,
        int32 num_mip_levels,
        int32 use_normalized_coords,
        int32 address_mode_u = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_v = WP_TEXTURE_ADDRESS_CLAMP,
        int32 address_mode_w = WP_TEXTURE_ADDRESS_CLAMP
    )
        : tex(tex)
        , width(width)
        , height(height)
        , depth(depth)
        , metadata(texture_metadata(
              num_channels,
              dtype,
              filter_mode,
              mip_filter_mode,
              num_mip_levels,
              use_normalized_coords,
              address_mode_u,
              address_mode_v,
              address_mode_w
          ))
    {
    }
};

static_assert(sizeof(texture1d_t) == 16, "texture1d_t metadata must remain packed");
static_assert(sizeof(texture2d_t) == 24, "texture2d_t metadata must remain packed");
static_assert(sizeof(texture3d_t) == 24, "texture3d_t metadata must remain packed");

#undef WP_TEXTURE_METADATA_ACCESSORS

struct cuda_array_desc_t {
    int32 ndim;
    int32 shape[3];
    int32 num_channels;
    int32 dtype;
};

#if defined(__HIP_PLATFORM_AMD__)
// A 3D texture larger than the device's 3D image limit. The Radeon 8060S can't create a 3D array with more than 2048 texels on
// any axis, while CUDA GPUs accept 16384 (test_texture_sample_grad_periodic_coordinates uses 16384 along one axis, and
// _non_power_of_two_plateau uses 8191). Such a texture lives in plain device memory, one mip level after another with x fastest,
// and is sampled in software with CUDA's semantics (texture_cuda_sample, texture_grad_texel). Its texture handle is that memory's
// address with the low bit set. A HIP texture object is an aligned pointer, so the bit tells the two apart.
#define WP_HIP_SOFT_TEXTURE_BIT 1ull
#define WP_HIP_SOFT_LEVEL_ALIGN 256ull

CUDA_CALLABLE inline bool hip_soft_texture(uint64 handle) { return (handle & WP_HIP_SOFT_TEXTURE_BIT) != 0; }

CUDA_CALLABLE inline int hip_soft_texel_bytes(int dtype, int num_channels)
{
    const int channel = (dtype == WP_TEXTURE_DTYPE_UINT8 || dtype == WP_TEXTURE_DTYPE_INT8) ? 1
        : (dtype == WP_TEXTURE_DTYPE_UINT16 || dtype == WP_TEXTURE_DTYPE_INT16 || dtype == WP_TEXTURE_DTYPE_FLOAT16)
        ? 2
        : 4;
    return channel * num_channels;
}

CUDA_CALLABLE inline int hip_soft_dim(int base_dim, int level)
{
    const int dim = base_dim >> level;
    return dim < 1 ? 1 : dim;
}

// Returns where a level starts in the texture's memory, after every level below it, each padded to WP_HIP_SOFT_LEVEL_ALIGN bytes.
CUDA_CALLABLE inline uint64 hip_soft_level_offset(int width, int height, int depth, int texel_bytes, int level)
{
    uint64 offset = 0;
    for (int l = 0; l < level; ++l) {
        const uint64 bytes = uint64(hip_soft_dim(width, l)) * uint64(hip_soft_dim(height, l))
            * uint64(hip_soft_dim(depth, l)) * uint64(texel_bytes);
        offset += (bytes + WP_HIP_SOFT_LEVEL_ALIGN - 1) / WP_HIP_SOFT_LEVEL_ALIGN * WP_HIP_SOFT_LEVEL_ALIGN;
    }
    return offset;
}
#endif

// ============================================================================
// CPU Software Texture Sampling Helpers
// ============================================================================

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

// Reduce periodic coordinates before division or scaling can discard the
// fractional position (or overflow). Unnormalized CPU textures also support
// periodic addressing, unlike CUDA textures.
inline float cpu_texture_normalized_coord(float coord, int normalized, int base_dim, int address_mode)
{
    if (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const float period
            = (normalized ? 1.0f : float(base_dim)) * (address_mode == WP_TEXTURE_ADDRESS_MIRROR ? 2.0f : 1.0f);
        coord = fmodf(coord, period);
        if (coord < 0.0f)
            coord += period;
    }
    return normalized ? coord : coord / float(base_dim);
}

// Convert reduced normalized coordinates to bounded texel coordinates.
// Apply periodic addressing to integer indices after rounding, so CLOSEST
// filtering chooses the same side of a MIRROR boundary as the CUDA sampler.
inline float cpu_apply_address_mode_1d(float coord, int size, int address_mode)
{
    float texel = coord * float(size) - 0.5f;
    if (address_mode == WP_TEXTURE_ADDRESS_CLAMP)
        texel = max(0.0f, min(texel, float(size - 1)));
    else if (address_mode != WP_TEXTURE_ADDRESS_WRAP && address_mode != WP_TEXTURE_ADDRESS_MIRROR)
        // Keep both neighboring indices outside the texture without converting
        // an unbounded float to int, including when finite coord * size overflows.
        texel = max(-2.0f, min(texel, float(size)));
    return texel;
}

// Clamp integer index to valid range
inline int64 cpu_clamp_index(int64 idx, int size) { return (idx < 0) ? 0 : ((idx >= size) ? size - 1 : idx); }

// Apply address mode to an integer texel index
// This is used for neighbor indices in linear filtering to properly handle wrap/mirror
inline int64 cpu_apply_address_mode_index(int64 idx, int size, int address_mode)
{
    switch (address_mode) {
    case WP_TEXTURE_ADDRESS_WRAP: {
        int64 m = idx % size;
        if (m < 0)
            m += size;
        return m;
    }
    case WP_TEXTURE_ADDRESS_CLAMP:
        return cpu_clamp_index(idx, size);
    case WP_TEXTURE_ADDRESS_MIRROR: {
        int64 period = 2 * int64(size);
        int64 m = idx % period;
        if (m < 0)
            m += period;
        if (m >= size)
            return period - m - 1;
        return m;
    }
    case WP_TEXTURE_ADDRESS_BORDER:
    default:
        return idx;  // border handled by fetch bounds checks
    }
}

// Nearest filtering rounds the original coordinate down to a texel index.
// Avoid the linear sampler's subtract/add-half round trip, which can erase a
// small negative coordinate before BORDER addressing. Wide arithmetic also
// preserves its sign through normalization, including float32 subnormals.
inline int64 cpu_nearest_texel_index(float coord, int normalized, int base_dim, int level_dim, int address_mode)
{
    double reduced = double(coord);
    if (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const double period
            = (normalized ? 1.0 : double(base_dim)) * (address_mode == WP_TEXTURE_ADDRESS_MIRROR ? 2.0 : 1.0);
        // Keep negative remainders: adding the period can round them up to its
        // endpoint. Integer addressing below handles either sign directly.
        reduced = fmod(reduced, period);
    }
    double texel = normalized ? reduced * double(level_dim) : reduced * double(level_dim) / double(base_dim);
    if (address_mode != WP_TEXTURE_ADDRESS_WRAP && address_mode != WP_TEXTURE_ADDRESS_MIRROR)
        texel = max(-1.0, min(texel, double(level_dim)));
    return cpu_apply_address_mode_index(int64(floor(texel)), level_dim, address_mode);
}

// Check if index is within bounds (for border mode)
inline bool cpu_in_bounds_1d(int64 x, int w) { return x >= 0 && x < w; }

inline bool cpu_in_bounds_2d(int64 x, int64 y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

inline bool cpu_in_bounds_3d(int64 x, int64 y, int64 z, int w, int h, int d)
{
    return x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d;
}

// Clamp a LOD value to the texture's valid mip-level range.
inline float cpu_clamp_lod(const Texture* tex, float lod)
{
    if (lod < 0.0f)
        return 0.0f;
    float max_lod = (float)(tex->num_mip_levels - 1);
    return lod > max_lod ? max_lod : lod;
}

// Convert IEEE 754 half-precision bits to float (for CPU float16 texture support)
inline float cpu_half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t result;
    if (exp == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            // Denormalized: convert to normalized float
            exp = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            result = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        result = sign | 0x7F800000u | ((uint32_t)mantissa << 13);
    } else {
        result = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mantissa << 13);
    }

    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = result;
    return conv.f;
}

// Decode a raw texel at ``idx`` into a float. Normalize 8- and 16-bit integers;
// convert 32-bit integers numerically and return float types as-is.
inline float cpu_decode_texel(const void* level_data, int dtype, size_t idx)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return ((const uint8_t*)level_data)[idx] / 255.0f;
    case WP_TEXTURE_DTYPE_UINT16:
        return ((const uint16_t*)level_data)[idx] / 65535.0f;
    case WP_TEXTURE_DTYPE_UINT32:
        return float(((const uint32_t*)level_data)[idx]);
    case WP_TEXTURE_DTYPE_INT8: {
        float v = ((const int8_t*)level_data)[idx] / 127.0f;
        return v < -1.0f ? -1.0f : v;
    }
    case WP_TEXTURE_DTYPE_INT16: {
        float v = ((const int16_t*)level_data)[idx] / 32767.0f;
        return v < -1.0f ? -1.0f : v;
    }
    case WP_TEXTURE_DTYPE_INT32:
        return float(((const int32_t*)level_data)[idx]);
    case WP_TEXTURE_DTYPE_FLOAT16:
        return cpu_half_to_float(((const uint16_t*)level_data)[idx]);
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return ((const float*)level_data)[idx];
    }
}

// Fetch a single texel value from the given mip level.
inline float cpu_fetch_texel_1d(const Texture* tex, int level, int64 x, int channel)
{
    const int w = tex->mip_widths_arr[level];
    if (!cpu_in_bounds_1d(x, w) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = size_t(x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_2d(const Texture* tex, int level, int64 x, int64 y, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    if (!cpu_in_bounds_2d(x, y, w, h) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = (size_t(y) * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

inline float cpu_fetch_texel_3d(const Texture* tex, int level, int64 x, int64 y, int64 z, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    const int d = tex->mip_depths_arr[level];
    if (!cpu_in_bounds_3d(x, y, z, w, h, d) || channel < 0 || channel >= tex->num_channels) {
        return 0.0f;
    }
    size_t idx = ((size_t(z) * h + y) * w + x) * tex->num_channels + channel;
    return cpu_decode_texel(tex->mip_data[level], tex->dtype, idx);
}

// Sample a single channel with linear interpolation (1D) at a specific mip level.
inline float cpu_sample_1d_channel_at_level(const Texture* tex, int level, float u, int channel)
{
    const int w = tex->mip_widths_arr[level];

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        return cpu_fetch_texel_1d(tex, level, x, channel);
    } else {
        // Linear interpolation
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        int64 x0 = (int64)floor(tx);
        int64 x1 = x0 + 1;

        float fx = tx - x0;

        // Apply address mode to neighbor indices (properly handles wrap/mirror at edges)
        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);

        float v0 = cpu_fetch_texel_1d(tex, level, x0, channel);
        float v1 = cpu_fetch_texel_1d(tex, level, x1, channel);

        return v0 * (1.0f - fx) + v1 * fx;
    }
}

// Sample a single channel with bilinear interpolation (2D) at a specific mip level.
inline float cpu_sample_2d_channel_at_level(const Texture* tex, int level, float u, float v, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        const int64 y = cpu_nearest_texel_index(v, tex->use_normalized_coords, tex->height, h, tex->address_mode_v);
        return cpu_fetch_texel_2d(tex, level, x, y, channel);
    } else {
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float coord_v = cpu_texture_normalized_coord(v, tex->use_normalized_coords, tex->height, tex->address_mode_v);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);
        int64 x0 = (int64)floor(tx);
        int64 y0 = (int64)floor(ty);
        int64 x1 = x0 + 1;
        int64 y1 = y0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;

        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, h, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, h, tex->address_mode_v);

        float v00 = cpu_fetch_texel_2d(tex, level, x0, y0, channel);
        float v10 = cpu_fetch_texel_2d(tex, level, x1, y0, channel);
        float v01 = cpu_fetch_texel_2d(tex, level, x0, y1, channel);
        float v11 = cpu_fetch_texel_2d(tex, level, x1, y1, channel);

        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        return v0 * (1.0f - fy) + v1 * fy;
    }
}

// Sample a single channel with trilinear interpolation (3D) at a specific mip level.
inline float cpu_sample_3d_channel_at_level(const Texture* tex, int level, float u, float v, float w_coord, int channel)
{
    const int w = tex->mip_widths_arr[level];
    const int h = tex->mip_heights_arr[level];
    const int d = tex->mip_depths_arr[level];

    if (tex->filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        const int64 x = cpu_nearest_texel_index(u, tex->use_normalized_coords, tex->width, w, tex->address_mode_u);
        const int64 y = cpu_nearest_texel_index(v, tex->use_normalized_coords, tex->height, h, tex->address_mode_v);
        const int64 z
            = cpu_nearest_texel_index(w_coord, tex->use_normalized_coords, tex->depth, d, tex->address_mode_w);
        return cpu_fetch_texel_3d(tex, level, x, y, z, channel);
    } else {
        float coord_u = cpu_texture_normalized_coord(u, tex->use_normalized_coords, tex->width, tex->address_mode_u);
        float coord_v = cpu_texture_normalized_coord(v, tex->use_normalized_coords, tex->height, tex->address_mode_v);
        float coord_w
            = cpu_texture_normalized_coord(w_coord, tex->use_normalized_coords, tex->depth, tex->address_mode_w);
        float tx = cpu_apply_address_mode_1d(coord_u, w, tex->address_mode_u);
        float ty = cpu_apply_address_mode_1d(coord_v, h, tex->address_mode_v);
        float tz = cpu_apply_address_mode_1d(coord_w, d, tex->address_mode_w);
        int64 x0 = (int64)floor(tx);
        int64 y0 = (int64)floor(ty);
        int64 z0 = (int64)floor(tz);
        int64 x1 = x0 + 1;
        int64 y1 = y0 + 1;
        int64 z1 = z0 + 1;

        float fx = tx - x0;
        float fy = ty - y0;
        float fz = tz - z0;

        x0 = cpu_apply_address_mode_index(x0, w, tex->address_mode_u);
        x1 = cpu_apply_address_mode_index(x1, w, tex->address_mode_u);
        y0 = cpu_apply_address_mode_index(y0, h, tex->address_mode_v);
        y1 = cpu_apply_address_mode_index(y1, h, tex->address_mode_v);
        z0 = cpu_apply_address_mode_index(z0, d, tex->address_mode_w);
        z1 = cpu_apply_address_mode_index(z1, d, tex->address_mode_w);

        float v000 = cpu_fetch_texel_3d(tex, level, x0, y0, z0, channel);
        float v100 = cpu_fetch_texel_3d(tex, level, x1, y0, z0, channel);
        float v010 = cpu_fetch_texel_3d(tex, level, x0, y1, z0, channel);
        float v110 = cpu_fetch_texel_3d(tex, level, x1, y1, z0, channel);
        float v001 = cpu_fetch_texel_3d(tex, level, x0, y0, z1, channel);
        float v101 = cpu_fetch_texel_3d(tex, level, x1, y0, z1, channel);
        float v011 = cpu_fetch_texel_3d(tex, level, x0, y1, z1, channel);
        float v111 = cpu_fetch_texel_3d(tex, level, x1, y1, z1, channel);

        float v00 = v000 * (1.0f - fx) + v100 * fx;
        float v10 = v010 * (1.0f - fx) + v110 * fx;
        float v01 = v001 * (1.0f - fx) + v101 * fx;
        float v11 = v011 * (1.0f - fx) + v111 * fx;

        float v0 = v00 * (1.0f - fy) + v10 * fy;
        float v1 = v01 * (1.0f - fy) + v11 * fy;

        return v0 * (1.0f - fz) + v1 * fz;
    }
}

// Sample a single channel across mipmap levels using the texture's mip filter mode.
inline float cpu_sample_1d_channel(const Texture* tex, float u, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_1d_channel_at_level(tex, nearest, u, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_1d_channel_at_level(tex, level0, u, channel);
    if (level1 == level0 || fl == 0.0f)
        return v0;
    float v1 = cpu_sample_1d_channel_at_level(tex, level1, u, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

inline float cpu_sample_2d_channel(const Texture* tex, float u, float v, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_2d_channel_at_level(tex, nearest, u, v, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_2d_channel_at_level(tex, level0, u, v, channel);
    if (level1 == level0 || fl == 0.0f)
        return v0;
    float v1 = cpu_sample_2d_channel_at_level(tex, level1, u, v, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

inline float cpu_sample_3d_channel(const Texture* tex, float u, float v, float w, int channel, float lod)
{
    float clamped_lod = cpu_clamp_lod(tex, lod);
    int level0 = (int)floor(clamped_lod);
    if (tex->num_mip_levels <= 1 || tex->mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped_lod + 0.5f);
        if (nearest >= tex->num_mip_levels)
            nearest = tex->num_mip_levels - 1;
        return cpu_sample_3d_channel_at_level(tex, nearest, u, v, w, channel);
    }
    int level1 = level0 + 1;
    if (level1 >= tex->num_mip_levels)
        level1 = tex->num_mip_levels - 1;
    float fl = clamped_lod - (float)level0;
    float v0 = cpu_sample_3d_channel_at_level(tex, level0, u, v, w, channel);
    if (level1 == level0 || fl == 0.0f)
        return v0;
    float v1 = cpu_sample_3d_channel_at_level(tex, level1, u, v, w, channel);
    return v0 * (1.0f - fl) + v1 * fl;
}

#endif  // !defined(__CUDA_ARCH__)

// ============================================================================
// Texture Sampling Functions
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)
// Warp stores texture objects as 64-bit handles (CUDA's cudaTextureObject_t is an integer); HIP's texture fetch
// functions take hipTextureObject_t, a pointer type. These overloads forward a handle to them.
template <typename T> __device__ inline T tex1D(unsigned long long texObj, float x)
{
    return ::tex1D<T>(reinterpret_cast<hipTextureObject_t>(texObj), x);
}
template <typename T> __device__ inline T tex2D(unsigned long long texObj, float x, float y)
{
    return ::tex2D<T>(reinterpret_cast<hipTextureObject_t>(texObj), x, y);
}
template <typename T> __device__ inline T tex3D(unsigned long long texObj, float x, float y, float z)
{
    return ::tex3D<T>(reinterpret_cast<hipTextureObject_t>(texObj), x, y, z);
}
template <typename T> __device__ inline T tex1DLod(unsigned long long texObj, float x, float lod)
{
    return ::tex1DLod<T>(reinterpret_cast<hipTextureObject_t>(texObj), x, lod);
}
template <typename T> __device__ inline T tex2DLod(unsigned long long texObj, float x, float y, float lod)
{
    return ::tex2DLod<T>(reinterpret_cast<hipTextureObject_t>(texObj), x, y, lod);
}
template <typename T> __device__ inline T tex3DLod(unsigned long long texObj, float x, float y, float z, float lod)
{
    return ::tex3DLod<T>(reinterpret_cast<hipTextureObject_t>(texObj), x, y, z, lod);
}
#endif

// TODO: Implement texture fetch functions for Clang CUDA JIT (currently stubbed, texture API is not supported)
#if defined(__clang__) && defined(__CUDA__)
template <typename T> __device__ T tex1D(unsigned long long texObj, float x)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex2D(unsigned long long texObj, float x, float y)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex3D(unsigned long long texObj, float x, float y, float z)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex1DLod(unsigned long long texObj, float x, float lod)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex2DLod(unsigned long long texObj, float x, float y, float lod)
{
    T v {};
    return v;
}
template <typename T> __device__ T tex3DLod(unsigned long long texObj, float x, float y, float z, float lod)
{
    T v {};
    return v;
}
#endif

// CUDA < 13.1 can miscompile vector texture fetches in functions that mix
// return widths on sm_89 and older; texture_cuda_fetch adds a call boundary.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
constexpr bool use_cuda_texture_mixed_width_workaround = (__CUDA_ARCH__ < 900)
    && ((__CUDACC_VER_MAJOR__ < 13) || ((__CUDACC_VER_MAJOR__ == 13) && (__CUDACC_VER_MINOR__ < 1)));
#else
constexpr bool use_cuda_texture_mixed_width_workaround = false;
#endif

// Relies on undocumented behavior: a float fetch returns 32-bit integer texels
// as raw bits (test_texture_sample_32bit_int checks exact values). Per-format
// fetches would need a branch per sample, which serializes fetches in loops.
CUDA_CALLABLE_DEVICE inline float texture_cuda_decode(float bits, bool is_int32, bool is_uint32)
{
    union {
        float f;
        int32 i;
        uint32 u;
    } texel;
    texel.f = bits;
    const float as_int32 = float(texel.i);
    const float as_uint32 = float(texel.u);
    return is_int32 ? as_int32 : (is_uint32 ? as_uint32 : bits);
}

template <typename T> struct texture_cuda_sample_types;

template <> struct texture_cuda_sample_types<float> {
    using float_type = float;

    static CUDA_CALLABLE_DEVICE float decode(float bits, bool is_int32, bool is_uint32)
    {
        return texture_cuda_decode(bits, is_int32, is_uint32);
    }
};

template <> struct texture_cuda_sample_types<vec2f> {
    using float_type = float2;

    static CUDA_CALLABLE_DEVICE vec2f decode(float2 bits, bool is_int32, bool is_uint32)
    {
        return vec2f(
            texture_cuda_decode(bits.x, is_int32, is_uint32), texture_cuda_decode(bits.y, is_int32, is_uint32)
        );
    }
};

template <> struct texture_cuda_sample_types<vec4f> {
    using float_type = float4;

    static CUDA_CALLABLE_DEVICE vec4f decode(float4 bits, bool is_int32, bool is_uint32)
    {
        return vec4f(
            texture_cuda_decode(bits.x, is_int32, is_uint32), texture_cuda_decode(bits.y, is_int32, is_uint32),
            texture_cuda_decode(bits.z, is_int32, is_uint32), texture_cuda_decode(bits.w, is_int32, is_uint32)
        );
    }
};

template <typename NativeType> struct texture_cuda_fetch {
    static CUDA_CALLABLE_DEVICE __noinline__ NativeType sample_base(const texture1d_t& tex, float u)
    {
        return tex1D<NativeType>(tex.tex, u);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ NativeType sample_base(const texture2d_t& tex, float u, float v)
    {
        return tex2D<NativeType>(tex.tex, u, v);
    }

    static CUDA_CALLABLE_DEVICE __noinline__ NativeType sample_base(const texture3d_t& tex, float u, float v, float w)
    {
        return tex3D<NativeType>(tex.tex, u, v, w);
    }

    static CUDA_CALLABLE_DEVICE NativeType sample(const texture1d_t& tex, float lod, float u)
    {
        if constexpr (use_cuda_texture_mixed_width_workaround && sizeof(NativeType) > sizeof(float))
            return (lod < 0.0f) ? sample_base(tex, u) : tex1DLod<NativeType>(tex.tex, u, lod);
        else
            return (lod < 0.0f) ? tex1D<NativeType>(tex.tex, u) : tex1DLod<NativeType>(tex.tex, u, lod);
    }

    static CUDA_CALLABLE_DEVICE NativeType sample(const texture2d_t& tex, float lod, float u, float v)
    {
        if constexpr (use_cuda_texture_mixed_width_workaround && sizeof(NativeType) > sizeof(float))
            return (lod < 0.0f) ? sample_base(tex, u, v) : tex2DLod<NativeType>(tex.tex, u, v, lod);
        else
            return (lod < 0.0f) ? tex2D<NativeType>(tex.tex, u, v) : tex2DLod<NativeType>(tex.tex, u, v, lod);
    }

    static CUDA_CALLABLE_DEVICE NativeType sample(const texture3d_t& tex, float lod, float u, float v, float w)
    {
        if constexpr (use_cuda_texture_mixed_width_workaround && sizeof(NativeType) > sizeof(float))
            return (lod < 0.0f) ? sample_base(tex, u, v, w) : tex3DLod<NativeType>(tex.tex, u, v, w, lod);
        else
            return (lod < 0.0f) ? tex3D<NativeType>(tex.tex, u, v, w) : tex3DLod<NativeType>(tex.tex, u, v, w, lod);
    }
};

#if defined(__HIP_DEVICE_COMPILE__)
// The Radeon's texture unit maps a coordinate far outside the texture (2^23 texels and beyond, such as a large offset) to a
// different texel than CUDA's does, because its wrap, mirror, and clamp run in fixed point. So each coordinate is first brought
// into the range where the two agree, keeping the texel and filter weights the original coordinate selects.
//   - WRAP, normalized, keeps the fractional part (x - floor(x) is exact in floating point).
//   - MIRROR, normalized, keeps the coordinate's place within one period of two.
//   - CLAMP and BORDER (and unnormalized WRAP and MIRROR, which CUDA clamps) keep at most two whole extents past either edge,
//     and at least two texels at any mip level, beyond which clamping reads the edge texel and BORDER reads the border color.
// A coordinate that is not finite is left to the hardware.
CUDA_CALLABLE_DEVICE inline float hip_texture_coord(float c, int mode, int normalized, int size)
{
    if (!isfinite(c))
        return c;
    if (normalized && mode == WP_TEXTURE_ADDRESS_WRAP)
        return c - floorf(c);
    if (normalized && mode == WP_TEXTURE_ADDRESS_MIRROR)
        return c - 2.0f * floorf(c * 0.5f);
    const float extent = normalized ? 1.0f : float(size);
    return fminf(fmaxf(c, -2.0f * extent), 3.0f * extent);
}

CUDA_CALLABLE_DEVICE inline void hip_texture_coords(const texture1d_t& tex, float& u)
{
    u = hip_texture_coord(u, tex.get_address_mode_u(), tex.get_use_normalized_coords(), tex.width);
}

CUDA_CALLABLE_DEVICE inline void hip_texture_coords(const texture2d_t& tex, float& u, float& v)
{
    const int normalized = tex.get_use_normalized_coords();
    u = hip_texture_coord(u, tex.get_address_mode_u(), normalized, tex.width);
    v = hip_texture_coord(v, tex.get_address_mode_v(), normalized, tex.height);
}

CUDA_CALLABLE_DEVICE inline void hip_texture_coords(const texture3d_t& tex, float& u, float& v, float& w)
{
    const int normalized = tex.get_use_normalized_coords();
    u = hip_texture_coord(u, tex.get_address_mode_u(), normalized, tex.width);
    v = hip_texture_coord(v, tex.get_address_mode_v(), normalized, tex.height);
    w = hip_texture_coord(w, tex.get_address_mode_w(), normalized, tex.depth);
}

// Samples a soft 3D texture (hip_soft_texture above) the way CUDA's sampler reads a texture, with the GPU gradient path's own
// cell and addressing (defined below). Each axis's cell and weight come from texture_grad_cell after its periodic reduction
// (unnormalized WRAP and MIRROR act as CLAMP, with eight fractional bits), the neighbors are addressed by
// cuda_texture_grad_index, a texel outside the texture is zero (BORDER's color), and each format is converted the way the image
// load converts it.
CUDA_CALLABLE inline float texture_coord_scale(int use_normalized_coords, int level_dim, int base_dim);
CUDA_CALLABLE inline float texture_grad_periodic_coord(float coord, int use_normalized_coords, int address_mode);
CUDA_CALLABLE inline int
texture_grad_cell(float coord, int normalized, int base_dim, int level_dim, int address_mode, float& fraction);
CUDA_CALLABLE_DEVICE inline int cuda_texture_grad_index(int index, int size, int mode, int normalized);

struct hip_soft_level_t {
    const unsigned char* data;
    int width;
    int height;
    int depth;
    int texel_bytes;
    int dtype;
    int num_channels;
};

CUDA_CALLABLE_DEVICE inline hip_soft_level_t hip_soft_level(const texture3d_t& tex, int level)
{
    hip_soft_level_t soft;
    soft.dtype = tex.get_dtype();
    soft.num_channels = tex.get_num_channels();
    soft.texel_bytes = hip_soft_texel_bytes(soft.dtype, soft.num_channels);
    soft.width = hip_soft_dim(tex.width, level);
    soft.height = hip_soft_dim(tex.height, level);
    soft.depth = hip_soft_dim(tex.depth, level);
    soft.data = reinterpret_cast<const unsigned char*>(tex.tex & ~WP_HIP_SOFT_TEXTURE_BIT)
        + hip_soft_level_offset(tex.width, tex.height, tex.depth, soft.texel_bytes, level);
    return soft;
}

// Returns one channel the way the image load gives it. 8- and 16-bit integers are normalized, half is widened, and a 32-bit
// integer is returned either as its raw bits (what a float fetch of it returns, see texture_cuda_decode) or as its value.
CUDA_CALLABLE_DEVICE inline float hip_soft_channel(const unsigned char* texel, int dtype, int channel, bool raw)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return float(texel[channel]) / 255.0f;
    case WP_TEXTURE_DTYPE_UINT16:
        return float(reinterpret_cast<const unsigned short*>(texel)[channel]) / 65535.0f;
    case WP_TEXTURE_DTYPE_INT8:
        return fmaxf(float(reinterpret_cast<const signed char*>(texel)[channel]) / 127.0f, -1.0f);
    case WP_TEXTURE_DTYPE_INT16:
        return fmaxf(float(reinterpret_cast<const short*>(texel)[channel]) / 32767.0f, -1.0f);
    case WP_TEXTURE_DTYPE_FLOAT16: {
        half h;
        h.u = reinterpret_cast<const unsigned short*>(texel)[channel];
        return half_to_float(h);
    }
    case WP_TEXTURE_DTYPE_UINT32: {
        const unsigned int bits = reinterpret_cast<const unsigned int*>(texel)[channel];
        return raw ? __uint_as_float(bits) : float(bits);
    }
    case WP_TEXTURE_DTYPE_INT32: {
        const int bits = reinterpret_cast<const int*>(texel)[channel];
        return raw ? __int_as_float(bits) : float(bits);
    }
    default:
        return reinterpret_cast<const float*>(texel)[channel];
    }
}

CUDA_CALLABLE_DEVICE inline vec4f hip_soft_texel(const hip_soft_level_t& soft, int x, int y, int z, bool raw)
{
    vec4f value(0.0f);
    if (x < 0 || x >= soft.width || y < 0 || y >= soft.height || z < 0 || z >= soft.depth)
        return value;
    const unsigned char* texel = soft.data
        + ((uint64(z) * uint64(soft.height) + uint64(y)) * uint64(soft.width) + uint64(x)) * uint64(soft.texel_bytes);
    for (int channel = 0; channel < soft.num_channels; ++channel)
        value[channel] = hip_soft_channel(texel, soft.dtype, channel, raw);
    return value;
}

// With CLOSEST filtering, returns the texel the coordinate falls in, addressed the way the sampler addresses it.
CUDA_CALLABLE_DEVICE inline int hip_soft_nearest(float coord, int normalized, int base_dim, int level_dim, int mode)
{
    if (!normalized && (mode == WP_TEXTURE_ADDRESS_WRAP || mode == WP_TEXTURE_ADDRESS_MIRROR))
        mode = WP_TEXTURE_ADDRESS_CLAMP;
    float t = texture_grad_periodic_coord(coord, normalized, mode) * texture_coord_scale(normalized, level_dim, base_dim);
    if (t != t)
        t = 0.0f;
    if (mode == WP_TEXTURE_ADDRESS_CLAMP || mode == WP_TEXTURE_ADDRESS_BORDER)
        t = fmaxf(-1.0f, fminf(t, float(level_dim)));
    return cuda_texture_grad_index(int(floorf(t)), level_dim, mode, normalized);
}

// With LINEAR filtering, returns the cell's lower texel and its weight.
CUDA_CALLABLE_DEVICE inline int
hip_soft_cell(float coord, int normalized, int base_dim, int level_dim, int mode, float& fraction)
{
    coord = texture_grad_periodic_coord(coord, normalized, mode);
    if (coord != coord)
        coord = 0.0f;
    return texture_grad_cell(coord, normalized, base_dim, level_dim, mode, fraction);
}

CUDA_CALLABLE_DEVICE inline vec4f hip_soft_sample_level(const texture3d_t& tex, int level, float u, float v, float w)
{
    const hip_soft_level_t soft = hip_soft_level(tex, level);
    const int normalized = tex.get_use_normalized_coords();
    const int mode_u = tex.get_address_mode_u();
    const int mode_v = tex.get_address_mode_v();
    const int mode_w = tex.get_address_mode_w();
    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_CLOSEST)
        return hip_soft_texel(
            soft, hip_soft_nearest(u, normalized, tex.width, soft.width, mode_u),
            hip_soft_nearest(v, normalized, tex.height, soft.height, mode_v),
            hip_soft_nearest(w, normalized, tex.depth, soft.depth, mode_w), false
        );
    float fx, fy, fz;
    const int x = hip_soft_cell(u, normalized, tex.width, soft.width, mode_u, fx);
    const int y = hip_soft_cell(v, normalized, tex.height, soft.height, mode_v, fy);
    const int z = hip_soft_cell(w, normalized, tex.depth, soft.depth, mode_w, fz);
    const int x0 = cuda_texture_grad_index(x, soft.width, mode_u, normalized);
    const int x1 = cuda_texture_grad_index(x + 1, soft.width, mode_u, normalized);
    const int y0 = cuda_texture_grad_index(y, soft.height, mode_v, normalized);
    const int y1 = cuda_texture_grad_index(y + 1, soft.height, mode_v, normalized);
    const int z0 = cuda_texture_grad_index(z, soft.depth, mode_w, normalized);
    const int z1 = cuda_texture_grad_index(z + 1, soft.depth, mode_w, normalized);
    const vec4f c00 = hip_soft_texel(soft, x0, y0, z0, false) * (1.0f - fx) + hip_soft_texel(soft, x1, y0, z0, false) * fx;
    const vec4f c10 = hip_soft_texel(soft, x0, y1, z0, false) * (1.0f - fx) + hip_soft_texel(soft, x1, y1, z0, false) * fx;
    const vec4f c01 = hip_soft_texel(soft, x0, y0, z1, false) * (1.0f - fx) + hip_soft_texel(soft, x1, y0, z1, false) * fx;
    const vec4f c11 = hip_soft_texel(soft, x0, y1, z1, false) * (1.0f - fx) + hip_soft_texel(soft, x1, y1, z1, false) * fx;
    const vec4f c0 = c00 * (1.0f - fy) + c10 * fy;
    const vec4f c1 = c01 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

template <typename T> CUDA_CALLABLE_DEVICE inline T hip_soft_as(const vec4f& value);
template <> CUDA_CALLABLE_DEVICE inline float hip_soft_as<float>(const vec4f& value) { return value[0]; }
template <> CUDA_CALLABLE_DEVICE inline vec2f hip_soft_as<vec2f>(const vec4f& value) { return vec2f(value[0], value[1]); }
template <> CUDA_CALLABLE_DEVICE inline vec4f hip_soft_as<vec4f>(const vec4f& value) { return value; }

// Samples at a LOD the way CUDA's sampler and the CPU path (cpu_sample_3d_channel) do. A negative LOD reads the base level, a
// CLOSEST mip filter reads the nearest level, and a LINEAR one blends the two levels around the LOD by its fraction.
template <typename T>
CUDA_CALLABLE_DEVICE inline T hip_soft_sample(const texture3d_t& tex, float lod, float u, float v, float w)
{
    const int levels = tex.get_num_mip_levels();
    if (lod < 0.0f || levels <= 1)
        return hip_soft_as<T>(hip_soft_sample_level(tex, 0, u, v, w));
    const float clamped = fminf(lod, float(levels - 1));
    if (tex.get_mip_filter_mode() == WP_TEXTURE_FILTER_CLOSEST)
        return hip_soft_as<T>(hip_soft_sample_level(tex, min(int(floorf(clamped + 0.5f)), levels - 1), u, v, w));
    const int level = int(floorf(clamped));
    const float fraction = clamped - float(level);
    const vec4f lower = hip_soft_sample_level(tex, level, u, v, w);
    if (fraction == 0.0f || level + 1 >= levels)
        return hip_soft_as<T>(lower);
    const vec4f upper = hip_soft_sample_level(tex, level + 1, u, v, w);
    return hip_soft_as<T>(lower * (1.0f - fraction) + upper * fraction);
}

// The soft branch of texture_cuda_sample handles a 3D texture with the soft bit set, and every other texture goes to the sampler.
template <typename T, typename TextureType, typename... Coordinates>
CUDA_CALLABLE_DEVICE inline bool hip_soft_try(const TextureType&, float, T&, Coordinates...)
{
    return false;
}

template <typename T>
CUDA_CALLABLE_DEVICE inline bool hip_soft_try(const texture3d_t& tex, float lod, T& out, float u, float v, float w)
{
    if (!hip_soft_texture(tex.tex))
        return false;
    out = hip_soft_sample<T>(tex, lod, u, v, w);
    return true;
}
#endif

template <typename T, typename TextureType, typename... Coordinates>
CUDA_CALLABLE_DEVICE T texture_cuda_sample(const TextureType& tex, float lod, Coordinates... coords)
{
    using types = texture_cuda_sample_types<T>;

#if defined(__HIP_DEVICE_COMPILE__)
    {
        T soft;
        if (hip_soft_try<T>(tex, lod, soft, coords...))
            return soft;
    }
    hip_texture_coords(tex, coords...);
#endif
    // One float fetch for every format; see texture_cuda_decode().
    const int32 dtype = tex.get_dtype();
#if defined(__HIP_DEVICE_COMPILE__)
    // The Radeon's sampler blends mip levels with a trilinear optimization. On a 16-texel texture with three levels, it reads
    // level 0 alone up to an LOD fraction of 0.25 and level 1 alone from 0.75, with a steep ramp in between (0.3125 gives 0.10,
    // 0.375 gives 0.23, 0.5 gives 0.5, and 0.625 gives 0.77), while CUDA blends by the fraction itself (in
    // test_texture_sample_grad_lod the forward samples did not move from LOD 0 to 0.125). So each level is sampled at a whole LOD,
    // where the optimization has nothing to blend, and the two are blended here by the LOD's own fraction, clamped to the levels
    // the way CUDA and the CPU path (cpu_clamp_lod) clamp it.
    if (lod >= 0.0f && tex.get_num_mip_levels() > 1 && tex.get_mip_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        const float top = float(tex.get_num_mip_levels() - 1);
        const float clamped = fminf(lod, top);
        const float level = floorf(clamped);
        const float fraction = clamped - level;
        const bool is_int32 = dtype == WP_TEXTURE_DTYPE_INT32;
        const bool is_uint32 = dtype == WP_TEXTURE_DTYPE_UINT32;
        const T lower
            = types::decode(texture_cuda_fetch<typename types::float_type>::sample(tex, level, coords...), is_int32, is_uint32);
        if (fraction == 0.0f)
            return lower;
        const T upper = types::decode(
            texture_cuda_fetch<typename types::float_type>::sample(tex, level + 1.0f, coords...), is_int32, is_uint32
        );
        return lerp(lower, upper, fraction);
    }
#endif
    const typename types::float_type bits = texture_cuda_fetch<typename types::float_type>::sample(tex, lod, coords...);
    return types::decode(bits, dtype == WP_TEXTURE_DTYPE_INT32, dtype == WP_TEXTURE_DTYPE_UINT32);
}
#endif

// Per-type sampling helpers; CUDA builds call texture_cuda_sample().
//
// Each ``sample_*`` helper accepts a single ``lod`` argument. When ``lod`` is negative
// (including :c:macro:`WP_TEXTURE_LOD_DISABLED`), the helper dispatches to the original
// non-LOD texture fetch (``tex1D`` / ``tex2D`` / ``tex3D`` on GPU, base mip level on CPU)
// so that callers that never opt into LOD preserve the pre-mipmap behavior exactly.
// Any nonnegative ``lod`` value routes through the mipmap-aware path (``tex1DLod`` / ...).
template <typename T> struct texture_sample_helper;

template <> struct texture_sample_helper<float> {
    static CUDA_CALLABLE float sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<float>(tex, lod, u);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0);
        return cpu_sample_1d_channel(cpu_tex, u, 0, lod);
#endif
    }

    static CUDA_CALLABLE float sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<float>(tex, lod, u, v);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0);
        return cpu_sample_2d_channel(cpu_tex, u, v, 0, lod);
#endif
    }

    static CUDA_CALLABLE float sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<float>(tex, lod, u, v, w);
#else
        if (tex.tex == 0)
            return 0.0f;
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0);
        return cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod);
#endif
    }

    static CUDA_CALLABLE float zero() { return 0.0f; }
};

template <> struct texture_sample_helper<vec2f> {
    static CUDA_CALLABLE vec2f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec2f>(tex, lod, u);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 1)
            );
        return vec2f(cpu_sample_1d_channel(cpu_tex, u, 0, lod), cpu_sample_1d_channel(cpu_tex, u, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec2f>(tex, lod, u, v);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0), cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 1)
            );
        return vec2f(cpu_sample_2d_channel(cpu_tex, u, v, 0, lod), cpu_sample_2d_channel(cpu_tex, u, v, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec2f>(tex, lod, u, v, w);
#else
        if (tex.tex == 0)
            return vec2f(0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec2f(
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 1)
            );
        return vec2f(cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 1, lod));
#endif
    }

    static CUDA_CALLABLE vec2f zero() { return vec2f(0.0f, 0.0f); }
};

template <> struct texture_sample_helper<vec4f> {
    static CUDA_CALLABLE vec4f sample_1d(const texture1d_t& tex, float u, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec4f>(tex, lod, u);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 0), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 1),
                cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 2), cpu_sample_1d_channel_at_level(cpu_tex, 0, u, 3)
            );
        return vec4f(
            cpu_sample_1d_channel(cpu_tex, u, 0, lod), cpu_sample_1d_channel(cpu_tex, u, 1, lod),
            cpu_sample_1d_channel(cpu_tex, u, 2, lod), cpu_sample_1d_channel(cpu_tex, u, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f sample_2d(const texture2d_t& tex, float u, float v, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec4f>(tex, lod, u, v);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 0),
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 1),
                cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 2), cpu_sample_2d_channel_at_level(cpu_tex, 0, u, v, 3)
            );
        return vec4f(
            cpu_sample_2d_channel(cpu_tex, u, v, 0, lod), cpu_sample_2d_channel(cpu_tex, u, v, 1, lod),
            cpu_sample_2d_channel(cpu_tex, u, v, 2, lod), cpu_sample_2d_channel(cpu_tex, u, v, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f sample_3d(const texture3d_t& tex, float u, float v, float w, float lod)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return texture_cuda_sample<vec4f>(tex, lod, u, v, w);
#else
        if (tex.tex == 0)
            return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        const Texture* cpu_tex = (const Texture*)tex.tex;
        if (lod < 0.0f)
            return vec4f(
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 0),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 1),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 2),
                cpu_sample_3d_channel_at_level(cpu_tex, 0, u, v, w, 3)
            );
        return vec4f(
            cpu_sample_3d_channel(cpu_tex, u, v, w, 0, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 1, lod),
            cpu_sample_3d_channel(cpu_tex, u, v, w, 2, lod), cpu_sample_3d_channel(cpu_tex, u, v, w, 3, lod)
        );
#endif
    }

    static CUDA_CALLABLE vec4f zero() { return vec4f(0.0f, 0.0f, 0.0f, 0.0f); }
};

#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
// CUDA 12 can miscompile optimized CUBIN texture sampling on sm_101 and the
// sm_120 family. When active lanes use different handles, a lane may sample
// another lane's texture. The uniform case is unaffected, so keep it inline and
// route only divergent handles through a no-inline call boundary that prevents
// the miscompile.
CUDA_CALLABLE_DEVICE inline bool texture_handle_is_uniform(uint64 handle)
{
    int uniform;
    (void)__match_all_sync(__activemask(), handle, &uniform);
    return uniform != 0;
}

template <typename T>
CUDA_CALLABLE_DEVICE __noinline__ T texture_sample_divergent(const texture2d_t& tex, float u, float v, float lod)
{
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
}

template <typename T>
CUDA_CALLABLE_DEVICE __noinline__ T
texture_sample_divergent(const texture3d_t& tex, float u, float v, float w, float lod)
{
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
}
#endif

// 1D texture sampling with scalar coordinate
template <typename T> CUDA_CALLABLE T texture_sample(const texture1d_t& tex, float u, float lod)
{
    return texture_sample_helper<T>::sample_1d(tex, u, lod);
}

// 2D texture sampling with vec2 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, const vec2f& uv, float lod)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, uv[0], uv[1], lod);
#endif
    return texture_sample_helper<T>::sample_2d(tex, uv[0], uv[1], lod);
}

// 2D texture sampling with separate u, v coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture2d_t& tex, float u, float v, float lod)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, lod);
#endif
    return texture_sample_helper<T>::sample_2d(tex, u, v, lod);
}

// 3D texture sampling with vec3 coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, const vec3f& uvw, float lod)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, uvw[0], uvw[1], uvw[2], lod);
#endif
    return texture_sample_helper<T>::sample_3d(tex, uvw[0], uvw[1], uvw[2], lod);
}

// 3D texture sampling with separate u, v, w coordinates
template <typename T> CUDA_CALLABLE T texture_sample(const texture3d_t& tex, float u, float v, float w, float lod)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex.tex))
        return texture_sample_divergent<T>(tex, u, v, w, lod);
#endif
    return texture_sample_helper<T>::sample_3d(tex, u, v, w, lod);
}

// ============================================================================
// Texture Sampling Adjoints
// ============================================================================
//
// Coordinate gradients interpolate differences of neighboring texels. Subtract
// before interpolation: differencing interpolated samples can lose small slopes
// when the texels share a large offset. CUDA reads exact corners with integer
// coordinates after applying the texture's address modes explicitly.
//
// LOD gradients use sample(level1) - sample(level0), the derivative of the
// linear blend between adjacent mip levels. They are zero when mip filtering is
// closest or the LOD is clamped.
//
// Gradients with respect to the texel data (``adj_tex``) are not implemented.

// Contract each active output channel independently so a zero seed cannot
// turn an overflowing difference in an inactive channel into NaN.
CUDA_CALLABLE inline float texture_grad_dot(float seed, float delta) { return seed == 0.0f ? 0.0f : seed * delta; }

CUDA_CALLABLE inline float texture_grad_dot(const vec2f& seed, const vec2f& delta)
{
    return texture_grad_dot(seed[0], delta[0]) + texture_grad_dot(seed[1], delta[1]);
}

CUDA_CALLABLE inline float texture_grad_dot(const vec4f& seed, const vec4f& delta)
{
    return texture_grad_dot(seed[0], delta[0]) + texture_grad_dot(seed[1], delta[1])
        + texture_grad_dot(seed[2], delta[2]) + texture_grad_dot(seed[3], delta[3]);
}

// Dimension of a mip level (mip chains always use floor-halving)
CUDA_CALLABLE inline int texture_mip_dim(int base_dim, int level)
{
    int d = base_dim >> level;
    return d < 1 ? 1 : d;
}

// Derivative of the raw texel-space coordinate with respect to the sampling
// coordinate: texel = coord * scale - 0.5. Unnormalized coordinates address
// mip levels in base-level texel space, hence the level/base ratio.
CUDA_CALLABLE inline float texture_coord_scale(int use_normalized_coords, int level_dim, int base_dim)
{
    return use_normalized_coords ? (float)level_dim : ((float)level_dim / (float)base_dim);
}

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// Reduce periodic coordinates before scaling to texels so large offsets do not
// round away the half-texel shift or the reconstructed corner coordinates.
CUDA_CALLABLE inline float texture_grad_periodic_coord(float coord, int use_normalized_coords, int address_mode)
{
    if (use_normalized_coords) {
        if (address_mode == WP_TEXTURE_ADDRESS_WRAP)
            return coord - floor(coord);
        if (address_mode == WP_TEXTURE_ADDRESS_MIRROR)
            return coord - 2.0f * floor(coord * 0.5f);
    }
    return coord;
}
#endif

// CPU dimensions can approach INT_MAX; a reflected cell and its upper neighbor
// need wider indices even though every addressed texel lies inside the texture.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
using texture_grad_index = int;
#else
using texture_grad_index = int64;
#endif

// Select an equivalent bounded interpolation cell before converting to int.
// Clamped axes must still select the correct edge for derivatives in other axes.
CUDA_CALLABLE inline texture_grad_index
texture_grad_cell(float coord, int normalized, int base_dim, int level_dim, int address_mode, float& fraction)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // CUDA treats unnormalized WRAP/MIRROR as CLAMP. Normalized periodic
    // coordinates were already reduced by adj_texture_sample().
    if (!normalized && (address_mode == WP_TEXTURE_ADDRESS_WRAP || address_mode == WP_TEXTURE_ADDRESS_MIRROR))
        address_mode = WP_TEXTURE_ADDRESS_CLAMP;
    float t = coord * texture_coord_scale(normalized, level_dim, base_dim) - 0.5f;
#else
    // Use the same normalization as the CPU forward sampler, including the
    // order of division and multiplication for non-power-of-two dimensions.
    coord = cpu_texture_normalized_coord(coord, normalized, base_dim, address_mode);
    float t = coord * float(level_dim) - 0.5f;
#endif
    if (address_mode == WP_TEXTURE_ADDRESS_CLAMP)
        t = max(-1.0f, min(t, float(level_dim - 1)));
    else if (address_mode == WP_TEXTURE_ADDRESS_BORDER)
        t = max(-2.0f, min(t, float(level_dim)));
    const float cell = floor(t);
    fraction = t - cell;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Match the CUDA sampler's eight fractional bits on the other axes.
    fraction = rintf(fraction * 256.0f) / 256.0f;
#endif
    return (texture_grad_index)cell;
}

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
// Read CPU corners directly so coordinate round trips cannot interpolate them.
template <typename T> inline T cpu_texture_grad_texel(const Texture* tex, int level, int64 x, int64 y = 0, int64 z = 0)
{
    x = cpu_apply_address_mode_index(x, tex->mip_widths_arr[level], tex->address_mode_u);
    y = cpu_apply_address_mode_index(y, tex->mip_heights_arr[level], tex->address_mode_v);
    z = cpu_apply_address_mode_index(z, tex->mip_depths_arr[level], tex->address_mode_w);
    if constexpr (sizeof(T) == sizeof(float)) {
        return cpu_fetch_texel_3d(tex, level, x, y, z, 0);
    } else {
        T value;
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            value[channel] = cpu_fetch_texel_3d(tex, level, x, y, z, channel);
        return value;
    }
}

#endif  // !defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// Integer coordinates bypass texture filtering. CUDA arrays, including 1D
// arrays, use the 2D array-fetch instruction; the 1D integer instruction is for
// linear-memory textures. Always fetch four channels to avoid mixed-width
// texture instruction compiler bugs, then select the requested components.
#if defined(__HIP_DEVICE_COMPILE__)
// PTX's tex.level is NVIDIA's own instruction. Here the same fetch is an image load at a mip level (image_load_mip) with integer
// coordinates and no sampler, so there is no filtering or address mode, and four channels are converted by the image's format
// (a 32-bit integer texture returns its raw bits, as CUDA's float fetch does). The first dwords of a HIP texture object are its
// image descriptor (TEXTURE_OBJECT_PARAMETERS_INIT in hip/amd_detail/texture_indirect_functions.h).
CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y)
{
    unsigned int __attribute__((address_space(4)))* image = (unsigned int __attribute__((address_space(4)))*)tex;
    int2 coords{x, y};
    const auto texel = __ockl_image_load_lod_2D(image, get_native_vector(coords), level);
    vec4f value;
    value[0] = texel[0];
    value[1] = texel[1];
    value[2] = texel[2];
    value[3] = texel[3];
    return value;
}

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y, int z)
{
    unsigned int __attribute__((address_space(4)))* image = (unsigned int __attribute__((address_space(4)))*)tex;
    int4 coords{x, y, z, 0};
    const auto texel = __ockl_image_load_lod_3D(image, get_native_vector(coords), level);
    vec4f value;
    value[0] = texel[0];
    value[1] = texel[1];
    value[2] = texel[2];
    value[3] = texel[3];
    return value;
}
#else
CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y)
{
    vec4f value;
    asm("tex.level.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3])
        : "l"(tex), "r"(x), "r"(y), "r"(level));
    return value;
}

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch(uint64 tex, int level, int x, int y, int z)
{
    vec4f value;
    asm("tex.level.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, 0}], %8;"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3])
        : "l"(tex), "r"(x), "r"(y), "r"(z), "r"(level));
    return value;
}
#endif

#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
CUDA_CALLABLE_DEVICE __noinline__ vec4f cuda_texture_grad_fetch_divergent(uint64 tex, int level, int x, int y)
{
    return cuda_texture_grad_fetch(tex, level, x, y);
}

CUDA_CALLABLE_DEVICE __noinline__ vec4f cuda_texture_grad_fetch_divergent(uint64 tex, int level, int x, int y, int z)
{
    return cuda_texture_grad_fetch(tex, level, x, y, z);
}
#endif

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch_protected(uint64 tex, int level, int x, int y)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex))
        return cuda_texture_grad_fetch_divergent(tex, level, x, y);
#endif
    return cuda_texture_grad_fetch(tex, level, x, y);
}

CUDA_CALLABLE_DEVICE inline vec4f cuda_texture_grad_fetch_protected(uint64 tex, int level, int x, int y, int z)
{
#if defined(WP_WORKAROUND_CUDA_TEXTURE_CUBIN)
    if (!texture_handle_is_uniform(tex))
        return cuda_texture_grad_fetch_divergent(tex, level, x, y, z);
#endif
    return cuda_texture_grad_fetch(tex, level, x, y, z);
}

// Integer fetches do not apply periodic or border addressing. CUDA texture
// dimensions are hardware-bounded, so their indices and periods fit in int32.
CUDA_CALLABLE_DEVICE inline int cuda_texture_grad_index(int index, int size, int mode, int normalized)
{
    if (!normalized && (mode == WP_TEXTURE_ADDRESS_WRAP || mode == WP_TEXTURE_ADDRESS_MIRROR))
        mode = WP_TEXTURE_ADDRESS_CLAMP;
    if (mode == WP_TEXTURE_ADDRESS_WRAP) {
        const int m = index % size;
        return m < 0 ? m + size : m;
    }
    if (mode == WP_TEXTURE_ADDRESS_MIRROR) {
        const int period = 2 * size;
        int m = index % period;
        if (m < 0)
            m += period;
        return m < size ? m : period - m - 1;
    }
    if (mode == WP_TEXTURE_ADDRESS_CLAMP)
        return max(0, min(index, size - 1));
    return index;
}

template <typename T> CUDA_CALLABLE_DEVICE inline T texture_grad_components(const vec4f& value)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return value[0];
    } else {
        T result;
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            result[channel] = value[channel];
        return result;
    }
}
#endif

template <typename T> CUDA_CALLABLE inline T texture_grad_texel(const texture1d_t& tex, int level, texture_grad_index x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int width = texture_mip_dim(tex.width, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width)
        return texture_sample_helper<T>::zero();
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, 0));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x);
#endif
}

template <typename T>
CUDA_CALLABLE inline T texture_grad_texel(const texture2d_t& tex, int level, texture_grad_index x, texture_grad_index y)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int width = texture_mip_dim(tex.width, level);
    const int height = texture_mip_dim(tex.height, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    y = cuda_texture_grad_index(y, height, tex.get_address_mode_v(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width || y < 0 || y >= height)
        return texture_sample_helper<T>::zero();
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, y));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x, y);
#endif
}

template <typename T>
CUDA_CALLABLE inline T
texture_grad_texel(const texture3d_t& tex, int level, texture_grad_index x, texture_grad_index y, texture_grad_index z)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int width = texture_mip_dim(tex.width, level);
    const int height = texture_mip_dim(tex.height, level);
    const int depth = texture_mip_dim(tex.depth, level);
    x = cuda_texture_grad_index(x, width, tex.get_address_mode_u(), tex.get_use_normalized_coords());
    y = cuda_texture_grad_index(y, height, tex.get_address_mode_v(), tex.get_use_normalized_coords());
    z = cuda_texture_grad_index(z, depth, tex.get_address_mode_w(), tex.get_use_normalized_coords());
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth)
        return texture_sample_helper<T>::zero();
#if defined(__HIP_DEVICE_COMPILE__)
    // A soft 3D texture's texel, as the image load would return it (hip_soft_texture).
    if (hip_soft_texture(tex.tex))
        return texture_grad_components<T>(hip_soft_texel(hip_soft_level(tex, level), x, y, z, true));
#endif
    return texture_grad_components<T>(cuda_texture_grad_fetch_protected(tex.tex, level, x, y, z));
#else
    return cpu_texture_grad_texel<T>((const Texture*)tex.tex, level, x, y, z);
#endif
}

// Keep the corner reference separate from the interpolated differences so a
// large shared texel offset cannot erase the difference between mip samples.
template <typename T> struct texture_grad_sample {
    T anchor;
    T residual;
};

// Opposite-sign finite texels can overflow centered differences even when the
// interpolated samples and their LOD difference are finite.
template <typename T> CUDA_CALLABLE inline bool texture_grad_lod_isfinite(const T& value)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return isfinite(value);
    } else {
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            if (!isfinite(value[channel]))
                return false;
        return true;
    }
}

// Preserve stable differences in the other channels, including small slopes
// atop large offsets. Forward sampling also preserves genuine NaN inputs.
template <typename T> CUDA_CALLABLE inline T texture_grad_lod_fallback(T value, const T& sampled_difference)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return isfinite(value) ? value : sampled_difference;
    } else {
        for (int channel = 0; channel < sizeof(T) / sizeof(float); ++channel)
            if (!isfinite(value[channel]))
                value[channel] = sampled_difference[channel];
        return value;
    }
}

// The returned sample reuses the fetched corners when an LOD gradient is also
// needed. Ignored results let the compiler remove the residual interpolation.
template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T>
texture_grad_coords_1d(const texture1d_t& tex, float u, int level, float weight, const T& adj_ret, float& adj_u)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);

    float fu;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const T v0 = texture_grad_texel<T>(tex, level, x);
    const T v1 = texture_grad_texel<T>(tex, level, x + 1);
    // At integer LODs the next level is sampled only for the LOD derivative.
    if (weight != 0.0f)
        adj_u += weight * su * texture_grad_dot(adj_ret, v1 - v0);
    return { v0, (v1 - v0) * fu };
}

template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T> texture_grad_coords_2d(
    const texture2d_t& tex, float u, float v, int level, float weight, const T& adj_ret, float& adj_u, float& adj_v
)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const int level_h = texture_mip_dim(tex.height, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);
    const float sv = texture_coord_scale(tex.get_use_normalized_coords(), level_h, tex.height);

    float fu, fv;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const texture_grad_index y
        = texture_grad_cell(v, tex.get_use_normalized_coords(), tex.height, level_h, tex.get_address_mode_v(), fv);
    const T v00 = texture_grad_texel<T>(tex, level, x, y);
    const T v10 = texture_grad_texel<T>(tex, level, x + 1, y);
    const T v01 = texture_grad_texel<T>(tex, level, x, y + 1);
    const T v11 = texture_grad_texel<T>(tex, level, x + 1, y + 1);

    if (weight != 0.0f) {
        adj_u += weight * su * lerp(texture_grad_dot(adj_ret, v10 - v00), texture_grad_dot(adj_ret, v11 - v01), fv);
        adj_v += weight * sv * lerp(texture_grad_dot(adj_ret, v01 - v00), texture_grad_dot(adj_ret, v11 - v10), fu);
    }
    return { v00, lerp((v10 - v00) * fu, lerp(v01 - v00, v11 - v00, fu), fv) };
}

template <typename T>
CUDA_CALLABLE inline texture_grad_sample<T> texture_grad_coords_3d(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    int level,
    float weight,
    const T& adj_ret,
    float& adj_u,
    float& adj_v,
    float& adj_w
)
{
    const int level_w = texture_mip_dim(tex.width, level);
    const int level_h = texture_mip_dim(tex.height, level);
    const int level_d = texture_mip_dim(tex.depth, level);
    const float su = texture_coord_scale(tex.get_use_normalized_coords(), level_w, tex.width);
    const float sv = texture_coord_scale(tex.get_use_normalized_coords(), level_h, tex.height);
    const float sw = texture_coord_scale(tex.get_use_normalized_coords(), level_d, tex.depth);

    float fu, fv, fw;
    const texture_grad_index x
        = texture_grad_cell(u, tex.get_use_normalized_coords(), tex.width, level_w, tex.get_address_mode_u(), fu);
    const texture_grad_index y
        = texture_grad_cell(v, tex.get_use_normalized_coords(), tex.height, level_h, tex.get_address_mode_v(), fv);
    const texture_grad_index z
        = texture_grad_cell(w, tex.get_use_normalized_coords(), tex.depth, level_d, tex.get_address_mode_w(), fw);
    const T v000 = texture_grad_texel<T>(tex, level, x, y, z);
    const T v100 = texture_grad_texel<T>(tex, level, x + 1, y, z);
    const T v010 = texture_grad_texel<T>(tex, level, x, y + 1, z);
    const T v110 = texture_grad_texel<T>(tex, level, x + 1, y + 1, z);
    const T v001 = texture_grad_texel<T>(tex, level, x, y, z + 1);
    const T v101 = texture_grad_texel<T>(tex, level, x + 1, y, z + 1);
    const T v011 = texture_grad_texel<T>(tex, level, x, y + 1, z + 1);
    const T v111 = texture_grad_texel<T>(tex, level, x + 1, y + 1, z + 1);

    if (weight != 0.0f) {
        adj_u += weight * su
            * lerp(lerp(texture_grad_dot(adj_ret, v100 - v000), texture_grad_dot(adj_ret, v110 - v010), fv),
                   lerp(texture_grad_dot(adj_ret, v101 - v001), texture_grad_dot(adj_ret, v111 - v011), fv), fw);
        adj_v += weight * sv
            * lerp(lerp(texture_grad_dot(adj_ret, v010 - v000), texture_grad_dot(adj_ret, v110 - v100), fu),
                   lerp(texture_grad_dot(adj_ret, v011 - v001), texture_grad_dot(adj_ret, v111 - v101), fu), fw);
        adj_w += weight * sw
            * lerp(lerp(texture_grad_dot(adj_ret, v001 - v000), texture_grad_dot(adj_ret, v101 - v100), fu),
                   lerp(texture_grad_dot(adj_ret, v011 - v010), texture_grad_dot(adj_ret, v111 - v110), fu), fv);
    }
    const T lower = lerp((v100 - v000) * fu, lerp(v010 - v000, v110 - v000, fu), fv);
    const T upper = lerp(lerp(v001 - v000, v101 - v000, fu), lerp(v011 - v000, v111 - v000, fu), fv);
    return { v000, lerp(lower, upper, fw) };
}

// Resolve the mip levels and blend weights the forward pass reads for a given
// LOD, mirroring cpu_clamp_lod and the mip filter selection. Returns the
// number of levels contributing spatial gradients (1 or 2).
CUDA_CALLABLE inline int
texture_grad_mip_levels(int num_mip_levels, int mip_filter_mode, float lod, int* levels, float* weights)
{
    const float max_lod = (float)(num_mip_levels - 1);
    float clamped = lod < 0.0f ? 0.0f : (lod > max_lod ? max_lod : lod);

    if (mip_filter_mode == WP_TEXTURE_FILTER_CLOSEST) {
        int nearest = (int)floor(clamped + 0.5f);
        if (nearest >= num_mip_levels)
            nearest = num_mip_levels - 1;
        levels[0] = nearest;
        weights[0] = 1.0f;
        return 1;
    }

    const int level0 = (int)floor(clamped);
    int level1 = level0 + 1;
    if (level1 >= num_mip_levels)
        level1 = num_mip_levels - 1;
    const float fl = clamped - (float)level0;

    levels[0] = level0;
    weights[0] = 1.0f - fl;
    if (level1 == level0 || fl == 0.0f)
        return 1;
    levels[1] = level1;
    weights[1] = fl;
    return 2;
}

// LOD gradient: the mip-linear blend is piecewise linear in the clamped LOD,
// with derivative sample(level1) - sample(level0) between levels and zero in
// the clamped ranges or with a CLOSEST mip filter.
CUDA_CALLABLE inline bool
texture_grad_lod_levels(int num_mip_levels, int mip_filter_mode, float lod, int& level0, int& level1)
{
    if (num_mip_levels <= 1 || mip_filter_mode != WP_TEXTURE_FILTER_LINEAR)
        return false;
    if (lod < 0.0f || lod >= (float)(num_mip_levels - 1))
        return false;
    level0 = (int)floor(lod);
    level1 = level0 + 1;
    return true;
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture1d_t& tex, float u, float lod, texture1d_t& adj_tex, float& adj_u, float& adj_lod, const T& adj_ret
)
{
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0 = texture_grad_coords_1d(tex, u, level0, 1.0f - fl, adj_ret, adj_u);
            const texture_grad_sample<T> sample1 = texture_grad_coords_1d(tex, u, level1, fl, adj_ret, adj_u);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_sample_helper<T>::sample_1d(tex, u, (float)level0);
                const T v1 = texture_sample_helper<T>::sample_1d(tex, u, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_1d(tex, u, 0, 1.0f, adj_ret, adj_u);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_1d(tex, u, levels[i], weights[i], adj_ret, adj_u);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_sample_helper<T>::sample_1d(tex, u, (float)level0);
        const T v1 = texture_sample_helper<T>::sample_1d(tex, u, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    float u,
    float v,
    float lod,
    texture2d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_lod,
    const T& adj_ret
)
{
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
    v = texture_grad_periodic_coord(v, tex.get_use_normalized_coords(), tex.get_address_mode_v());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0
                = texture_grad_coords_2d(tex, u, v, level0, 1.0f - fl, adj_ret, adj_u, adj_v);
            const texture_grad_sample<T> sample1 = texture_grad_coords_2d(tex, u, v, level1, fl, adj_ret, adj_u, adj_v);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_sample<T>(tex, u, v, (float)level0);
                const T v1 = texture_sample<T>(tex, u, v, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_2d(tex, u, v, 0, 1.0f, adj_ret, adj_u, adj_v);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_2d(tex, u, v, levels[i], weights[i], adj_ret, adj_u, adj_v);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_sample<T>(tex, u, v, (float)level0);
        const T v1 = texture_sample<T>(tex, u, v, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    const vec2f& uv,
    float lod,
    texture2d_t& adj_tex,
    vec2f& adj_uv,
    float& adj_lod,
    const T& adj_ret
)
{
    adj_texture_sample(tex, uv[0], uv[1], lod, adj_tex, adj_uv[0], adj_uv[1], adj_lod, adj_ret);
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    float lod,
    texture3d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_w,
    float& adj_lod,
    const T& adj_ret
)
{
    if (tex.tex == 0)
        return;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    u = texture_grad_periodic_coord(u, tex.get_use_normalized_coords(), tex.get_address_mode_u());
    v = texture_grad_periodic_coord(v, tex.get_use_normalized_coords(), tex.get_address_mode_v());
    w = texture_grad_periodic_coord(w, tex.get_use_normalized_coords(), tex.get_address_mode_w());
#endif

    const bool mipmapped = tex.get_num_mip_levels() > 1 && lod >= 0.0f;
    int level0, level1;
    const bool grad_lod = mipmapped
        && texture_grad_lod_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, level0, level1);

    if (tex.get_filter_mode() == WP_TEXTURE_FILTER_LINEAR) {
        if (grad_lod) {
            const float fl = lod - (float)level0;
            const texture_grad_sample<T> sample0
                = texture_grad_coords_3d(tex, u, v, w, level0, 1.0f - fl, adj_ret, adj_u, adj_v, adj_w);
            const texture_grad_sample<T> sample1
                = texture_grad_coords_3d(tex, u, v, w, level1, fl, adj_ret, adj_u, adj_v, adj_w);
            T difference = (sample1.anchor - sample0.anchor) + (sample1.residual - sample0.residual);
            if (!texture_grad_lod_isfinite(difference)) {
                const T v0 = texture_sample<T>(tex, u, v, w, (float)level0);
                const T v1 = texture_sample<T>(tex, u, v, w, (float)level1);
                difference = texture_grad_lod_fallback(difference, v1 - v0);
            }
            adj_lod += texture_grad_dot(adj_ret, difference);
        } else if (!mipmapped) {
            texture_grad_coords_3d(tex, u, v, w, 0, 1.0f, adj_ret, adj_u, adj_v, adj_w);
        } else {
            int levels[2];
            float weights[2];
            const int n
                = texture_grad_mip_levels(tex.get_num_mip_levels(), tex.get_mip_filter_mode(), lod, levels, weights);
            for (int i = 0; i < n; ++i)
                texture_grad_coords_3d(tex, u, v, w, levels[i], weights[i], adj_ret, adj_u, adj_v, adj_w);
        }
    } else if (grad_lod) {
        // CLOSEST spatial filtering needs only the two selected texels.
        const T v0 = texture_sample<T>(tex, u, v, w, (float)level0);
        const T v1 = texture_sample<T>(tex, u, v, w, (float)level1);
        adj_lod += texture_grad_dot(adj_ret, v1 - v0);
    }
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    const vec3f& uvw,
    float lod,
    texture3d_t& adj_tex,
    vec3f& adj_uvw,
    float& adj_lod,
    const T& adj_ret
)
{
    adj_texture_sample(tex, uvw[0], uvw[1], uvw[2], lod, adj_tex, adj_uvw[0], adj_uvw[1], adj_uvw[2], adj_lod, adj_ret);
}

// Type aliases for code generation
using Texture1D = texture1d_t;
using Texture2D = texture2d_t;
using Texture3D = texture3d_t;

// ============================================================================
// Adjoint Support for Texture Types (required when textures are array dtypes)
// Textures are not differentiable, so these are essentially no-ops
// ============================================================================

// 1D Texture operations
CUDA_CALLABLE inline texture1d_t add(const texture1d_t& a, const texture1d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture1d_t& operator+=(texture1d_t& a, const texture1d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture1d_t* p, const texture1d_t& t)
{
    // No-op: textures are not differentiable
}

// 2D Texture operations
CUDA_CALLABLE inline texture2d_t add(const texture2d_t& a, const texture2d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture2d_t& operator+=(texture2d_t& a, const texture2d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture2d_t* p, const texture2d_t& t)
{
    // No-op: textures are not differentiable
}

// 3D Texture operations
CUDA_CALLABLE inline texture3d_t add(const texture3d_t& a, const texture3d_t& b)
{
    // Textures are not addable; return first argument unchanged
    return a;
}

CUDA_CALLABLE inline texture3d_t& operator+=(texture3d_t& a, const texture3d_t& b)
{
    // No-op: textures have no gradients to accumulate
    return a;
}

CUDA_CALLABLE inline void adj_atomic_add(texture3d_t* p, const texture3d_t& t)
{
    // No-op: textures are not differentiable
}

}  // namespace wp
