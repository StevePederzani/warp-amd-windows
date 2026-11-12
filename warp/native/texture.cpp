// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by AMD (ROCm Warp) and Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.

#include "warp.h"

#include "error.h"
#include "texture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

using wp::get_texture_bytes_per_channel;
using wp::Texture;

uint64_t wp_texture_create_host(
    int ndim,
    int num_mip_levels,
    int* mip_widths,
    int* mip_heights,
    int* mip_depths,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** mip_data_ptrs_out
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    for (int i = 0; i < num_mip_levels; i++) {
        if (mip_widths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture width must be a positive integer at mip level %d, got %d", i, mip_widths[i]
            );
            return 0;
        }
        if (ndim > 1 && mip_heights[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture height must be a positive integer at mip level %d, got %d", i, mip_heights[i]
            );
            return 0;
        }
        if (ndim > 2 && mip_depths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture depth must be a positive integer at mip level %d, got %d", i, mip_depths[i]
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    int32_t addr_u = address_modes[0];
    int32_t addr_v = ndim > 1 ? address_modes[1] : 0;
    int32_t addr_w = ndim > 2 ? address_modes[2] : 0;

    Texture* tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)")) Texture(
        ndim, num_mip_levels, mip_widths, mip_heights, mip_depths, num_channels, dtype, filter_mode, mip_filter_mode,
        addr_u, addr_v, addr_w, use_normalized_coords
    );

    if (mip_data_ptrs_out) {
        for (int i = 0; i < num_mip_levels; i++) {
            mip_data_ptrs_out[i] = tex->mip_data[i];
        }
    }

    return reinterpret_cast<uint64_t>(tex);
}

void wp_texture_destroy_host(uint64_t tex_handle)
{
    Texture* tex = (Texture*)tex_handle;
    tex->~Texture();
    wp_free_host(tex);
}

#if WP_ENABLE_CUDA

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================

#include "cuda_util.h"

#if defined(__HIP_PLATFORM_AMD__)
#include <mutex>
#include <unordered_map>
#include <utility>

// hip_soft_texture (texture.h) is a 3D texture larger than the device's 3D image limit, kept in plain device memory. Its array
// handle is its record's address, and each mip level's handle is that address plus one plus the level, an odd address inside the
// record that no allocation returns. Its texture object's handle is the memory's address with the low bit set.
namespace {
struct HipSoftArray {
    void* context;
    int width;
    int height;
    int depth;
    int num_channels;
    int dtype;
    int num_levels;
    int texel_bytes;
    void* data;
};

std::mutex g_soft_mutex;
std::unordered_map<uint64_t, std::pair<HipSoftArray*, int>> g_soft_handles;  // An array's handle and its levels' handles.

bool soft_lookup(uint64_t handle, HipSoftArray** array, int* level)
{
    std::lock_guard<std::mutex> lock(g_soft_mutex);
    const auto found = g_soft_handles.find(handle);
    if (found == g_soft_handles.end())
        return false;
    *array = found->second.first;
    *level = found->second.second;
    return true;
}

// Returns true if a 3D texture fits within the current device's 3D image limits. The context must be current.
bool soft_needed(int ndim, size_t width, size_t height, size_t depth)
{
    if (ndim != 3)
        return false;
    CUdevice device = 0;
    if (!check_cu(cuCtxGetDevice_f(&device)))
        return false;
    int max_width = 0, max_height = 0, max_depth = 0;
    if (!check_cuda(hipDeviceGetAttribute(&max_width, hipDeviceAttributeMaxTexture3DWidth, device))
        || !check_cuda(hipDeviceGetAttribute(&max_height, hipDeviceAttributeMaxTexture3DHeight, device))
        || !check_cuda(hipDeviceGetAttribute(&max_depth, hipDeviceAttributeMaxTexture3DDepth, device)))
        return false;
    return width > size_t(max_width) || height > size_t(max_height) || depth > size_t(max_depth);
}

uint64_t soft_create(void* context, int width, int height, int depth, int num_channels, int dtype, int num_levels)
{
    const int texel_bytes = wp::hip_soft_texel_bytes(dtype, num_channels);
    const uint64_t bytes = wp::hip_soft_level_offset(width, height, depth, texel_bytes, num_levels);
    void* data = wp_alloc_device(context, size_t(bytes), "(native:texture)");
    if (!data)
        return 0;
    HipSoftArray* array = new HipSoftArray { context, width, height, depth, num_channels, dtype, num_levels, texel_bytes, data };
    const uint64_t handle = reinterpret_cast<uint64_t>(array);
    std::lock_guard<std::mutex> lock(g_soft_mutex);
    g_soft_handles[handle] = { array, 0 };
    for (int level = 0; level < num_levels; ++level)
        g_soft_handles[handle + 1 + level] = { array, level };
    return handle;
}

bool soft_destroy(uint64_t handle)
{
    HipSoftArray* array = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_soft_mutex);
        const auto found = g_soft_handles.find(handle);
        if (found == g_soft_handles.end() || reinterpret_cast<uint64_t>(found->second.first) != handle)
            return false;
        array = found->second.first;
        g_soft_handles.erase(handle);
        for (int level = 0; level < array->num_levels; ++level)
            g_soft_handles.erase(handle + 1 + level);
    }
    // Released like any Warp allocation, in stream order and deferred while a capture is in progress.
    wp_free_device(array->context, array->data);
    delete array;
    return true;
}

// Turns a copy's side on a soft array's level into plain device memory, with rows as wide as the level.
void soft_copy_side(int& memory_type, uint64_t& handle, unsigned& pitch, unsigned& height)
{
    if (memory_type != static_cast<int>(CU_MEMORYTYPE_ARRAY))
        return;
    HipSoftArray* array = nullptr;
    int level = 0;
    if (!soft_lookup(handle, &array, &level))
        return;
    memory_type = static_cast<int>(CU_MEMORYTYPE_DEVICE);
    handle = reinterpret_cast<uint64_t>(static_cast<char*>(array->data))
        + wp::hip_soft_level_offset(array->width, array->height, array->depth, array->texel_bytes, level);
    pitch = unsigned(wp::hip_soft_dim(array->width, level) * array->texel_bytes);
    height = unsigned(wp::hip_soft_dim(array->height, level));
}
}  // namespace
#endif

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_UINT32:
        return CU_AD_FORMAT_UNSIGNED_INT32;
    case WP_TEXTURE_DTYPE_INT8:
        return CU_AD_FORMAT_SIGNED_INT8;
    case WP_TEXTURE_DTYPE_INT16:
        return CU_AD_FORMAT_SIGNED_INT16;
    case WP_TEXTURE_DTYPE_INT32:
        return CU_AD_FORMAT_SIGNED_INT32;
    case WP_TEXTURE_DTYPE_FLOAT16:
        return CU_AD_FORMAT_HALF;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
static CUaddress_mode get_cuda_address_mode(int address_mode)
{
    switch (address_mode) {
    case 0:
        return CU_TR_ADDRESS_MODE_WRAP;
    case 1:
        return CU_TR_ADDRESS_MODE_CLAMP;
    case 2:
        return CU_TR_ADDRESS_MODE_MIRROR;
    case 3:
        return CU_TR_ADDRESS_MODE_BORDER;
    default:
        return CU_TR_ADDRESS_MODE_CLAMP;
    }
}

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    ContextGuard guard(context);

    size_t width = shape[0];
    size_t height = ndim > 1 ? shape[1] : 0;
    size_t depth = ndim > 2 ? shape[2] : 0;

#if defined(__HIP_PLATFORM_AMD__)
    if (soft_needed(ndim, width, height, depth)) {
        if (surface_access) {
            wp::set_error_string(
                "Warp HIP error: a %zu x %zu x %zu texture passes this device's 3D image limit, and surface access needs an image",
                width, height, depth
            );
            return 0;
        }
        return soft_create(context, int(width), int(height), int(depth), num_channels, dtype, num_mip_levels);
    }
#endif

    CUarray_format format = get_cuda_format(dtype);

    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = surface_access ? CUDA_ARRAY3D_SURFACE_LDST : 0;

    if (num_mip_levels > 1) {
        CUmipmappedArray mipmap_array = NULL;
        if (!check_cu(cuMipmappedArrayCreate_f(&mipmap_array, &arr_desc, (unsigned int)num_mip_levels))) {
            return 0;
        }
        return reinterpret_cast<uint64_t>(mipmap_array);
    }

    CUarray cuda_array = NULL;
    if (!check_cu(cuArray3DCreate_f(&cuda_array, &arr_desc))) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped)
{
#if defined(__HIP_PLATFORM_AMD__)
    if (soft_destroy(array_handle))
        return;
#endif
    release_or_defer(is_mipmapped ? ReleaseKind::MipmappedArray : ReleaseKind::Array, context, array_handle);
}

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    if (!mipmap_array_handle) {
        wp::set_error_string("Warp error: Null mipmapped array handle");
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__)
    {
        HipSoftArray* array = nullptr;
        int base_level = 0;
        if (soft_lookup(mipmap_array_handle, &array, &base_level)) {
            if (level < 0 || level >= array->num_levels) {
                wp::set_error_string("Warp error: mip level %d is outside the texture's %d levels", level, array->num_levels);
                return 0;
            }
            return reinterpret_cast<uint64_t>(array) + 1 + uint64_t(level);
        }
    }
#endif

    ContextGuard guard(context);

    CUarray level_array = NULL;
    if (!check_cu(
            cuMipmappedArrayGetLevel_f(&level_array, (CUmipmappedArray)mipmap_array_handle, (unsigned int)level)
        )) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(level_array);
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    // Warp's Python side numbers a copy's memory types the way CUDA does (texture.py MemoryType, with host 1, device 2, and
    // array 3), but HIP's enum differs. hipMemoryTypeArray is 10 and 3 is managed memory, so every copy into a texture array was
    // refused with "Invalid destination memory type 3". Each type is read in CUDA's numbering and named in HIP's.
    auto hip_memory_type = [](int t) -> int {
        return t == 1 ? static_cast<int>(CU_MEMORYTYPE_HOST)
            : t == 2  ? static_cast<int>(CU_MEMORYTYPE_DEVICE)
            : t == 3  ? static_cast<int>(CU_MEMORYTYPE_ARRAY)
                      : t;
    };
    dst_memory_type = hip_memory_type(dst_memory_type);
    src_memory_type = hip_memory_type(src_memory_type);
    // A soft 3D texture's level is plain device memory (hip_soft_texture in texture.h).
    soft_copy_side(dst_memory_type, dst_handle, dst_pitch, dst_height);
    soft_copy_side(src_memory_type, src_handle, src_pitch, src_height);
#endif

    CUstream cuda_stream = static_cast<CUstream>(stream);
    WP_HIP_ISSUE(cuda_stream);

    CUDA_MEMCPY3D copy_params = {};
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    copy_params.Depth = depth;

    copy_params.dstMemoryType = static_cast<CUmemorytype>(dst_memory_type);
    if (dst_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_DEVICE) {
        // CUDA: CUdeviceptr is `unsigned long long` (static_cast OK).
        // HIP:  CUdeviceptr is `void*` (only reinterpret_cast is valid).
#if defined(__HIP_PLATFORM_AMD__)
        copy_params.dstDevice = reinterpret_cast<CUdeviceptr>(dst_handle);
#else
        copy_params.dstDevice = static_cast<CUdeviceptr>(dst_handle);
#endif
    } else if (dst_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
    copy_params.dstPitch = dst_pitch;
    copy_params.dstHeight = dst_height;

    copy_params.srcMemoryType = static_cast<CUmemorytype>(src_memory_type);
    if (src_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.srcHost = reinterpret_cast<void*>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_DEVICE) {
#if defined(__HIP_PLATFORM_AMD__)
        copy_params.srcDevice = reinterpret_cast<CUdeviceptr>(src_handle);
#else
        copy_params.srcDevice = static_cast<CUdeviceptr>(src_handle);
#endif
    } else if (src_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
    copy_params.srcPitch = src_pitch;
    copy_params.srcHeight = src_height;

#if defined(__HIP_PLATFORM_AMD__)
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t capture_graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t capture_dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(
            cuda_stream, &capture_status, nullptr, &capture_graph, &capture_deps, &capture_dep_count
        )))
        return false;
    if (capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        // HIP's stream capture records no node for a 3D copy that touches an array. The copy runs once, at capture time, and the
        // relaunched graph never copies again (a texture's copy_from and copy_to captured as 0 nodes in
        // test_texture2d/3d_cuda_array_copy_api_graph_capture). While the stream is capturing, the copy is added to the capture's
        // graph as a memcpy node after the capture's frontier, and the frontier moves to it, which is what CUDA's capture records.
        if (!capture_graph) {
            wp::set_error_string("Warp HIP error: a capturing stream reported no graph for a texture copy");
            return false;
        }
        hipGraphNode_t node = NULL;
        if (!check_cuda(hipDrvGraphAddMemcpyNode(
                &node, capture_graph, capture_deps, capture_dep_count, &copy_params, static_cast<hipCtx_t>(context)
            )))
            return false;
        return check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, &node, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    }
    // CUDA stages a pageable host buffer before an async copy returns, so the caller can free it right away, and Warp's texture
    // upload passes a temporary array. HIP may read the host buffer after the call returns, and a texture uploaded from host
    // memory occasionally came out as garbage (test_texture_sample_32bit_int and test_texture1d_linear_filter, each of which
    // passed alone). So a copy with a host side waits for its stream here.
    if (!check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream)))
        return false;
    if (dst_memory_type == CU_MEMORYTYPE_HOST || src_memory_type == CU_MEMORYTYPE_HOST)
        return check_cu(cuStreamSynchronize_f(cuda_stream));
    return true;
#else
    return check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream));
#endif
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    if (!array_handle || !desc_out) {
        wp::set_error_string("Warp error: NULL array handle");
        return false;
    }

#if defined(__HIP_PLATFORM_AMD__)
    {
        HipSoftArray* array = nullptr;
        int level = 0;
        if (soft_lookup(array_handle, &array, &level)) {
            desc_out->ndim = 3;
            desc_out->shape[0] = wp::hip_soft_dim(array->width, level);
            desc_out->shape[1] = wp::hip_soft_dim(array->height, level);
            desc_out->shape[2] = wp::hip_soft_dim(array->depth, level);
            desc_out->num_channels = array->num_channels;
            desc_out->dtype = array->dtype;
            return true;
        }
    }
#endif

    ContextGuard guard(context);

    CUarray cuda_array = reinterpret_cast<CUarray>(array_handle);

    CUDA_ARRAY3D_DESCRIPTOR desc;
    if (!check_cu(cuArray3DGetDescriptor_f(&desc, cuda_array)))
        return false;

    switch (desc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT8;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT16;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT32;
        break;
    case CU_AD_FORMAT_SIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT8;
        break;
    case CU_AD_FORMAT_SIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT16;
        break;
    case CU_AD_FORMAT_SIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT32;
        break;
    case CU_AD_FORMAT_HALF:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT16;
        break;
    case CU_AD_FORMAT_FLOAT:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT32;
        break;
    default:
        wp::set_error_string("Warp error: Unsupported texture format");
        return false;
    }

    desc_out->shape[0] = int32_t(desc.Width);
    desc_out->shape[1] = int32_t(desc.Height);
    desc_out->shape[2] = int32_t(desc.Depth);

    if (desc.Depth == 0) {
        if (desc.Height == 0)
            desc_out->ndim = 1;
        else
            desc_out->ndim = 2;
    } else {
        desc_out->ndim = 3;
    }

    // TODO: desc.Flags?

    desc_out->num_channels = int32_t(desc.NumChannels);

    return true;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    if (!array_handle) {
        wp::set_error_string("Null texture array handle");
        return 0;
    }

    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS, num_mip_levels
        );
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__)
    {
        // A soft 3D texture is sampled from its memory, and its filter and address modes travel in the kernel's texture3d_t.
        HipSoftArray* array = nullptr;
        int level = 0;
        if (soft_lookup(array_handle, &array, &level)) {
            if (ndim != 3 || num_mip_levels != array->num_levels) {
                wp::set_error_string(
                    "Warp HIP error: a texture of %d dimensions and %d levels over a 3D texture's memory of %d levels", ndim,
                    num_mip_levels, array->num_levels
                );
                return 0;
            }
            return reinterpret_cast<uint64_t>(array->data) | WP_HIP_SOFT_TEXTURE_BIT;
        }
    }
#endif

    ContextGuard guard(context);

    CUDA_RESOURCE_DESC res_desc = {};
    if (num_mip_levels > 1) {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array_handle);
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array_handle);
    }
    res_desc.flags = 0;

    CUDA_TEXTURE_DESC tex_desc = {};

    for (int i = 0; i < 3; i++) {
        if (i < ndim)
            tex_desc.addressMode[i] = get_cuda_address_mode(address_modes[i]);
        else
            tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
    }

    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = (num_mip_levels > 1) ? (float)(num_mip_levels - 1) : 0.0f;

    CUtexObject tex_object = 0;
    check_cu(cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr));

    // CUDA: CUtexObject is `unsigned long long`. HIP: CUtexObject is `__hip_texture*`.
#if defined(__HIP_PLATFORM_AMD__)
    return reinterpret_cast<uint64_t>(tex_object);
#else
    return static_cast<uint64_t>(tex_object);
#endif
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle)
{
#if defined(__HIP_PLATFORM_AMD__)
    // A soft 3D texture's handle is its array's memory, which is released with the array.
    if (wp::hip_soft_texture(tex_handle))
        return;
#endif
    release_or_defer(ReleaseKind::TextureObject, context, tex_handle);
}

uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
#if defined(__HIP_PLATFORM_AMD__)
    {
        HipSoftArray* array = nullptr;
        int level = 0;
        if (soft_lookup(array_handle, &array, &level)) {
            wp::set_error_string("Warp HIP error: a 3D texture past this device's 3D image limit has no surface");
            return 0;
        }
    }
#endif
    ContextGuard guard(context);

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = reinterpret_cast<cudaArray_t>(array_handle);

    cudaSurfaceObject_t surface = 0;
    check_cuda(cudaCreateSurfaceObject(&surface, &desc));

    // CUDA: cudaSurfaceObject_t is `unsigned long long`. HIP: it is `__hip_surface*`.
#if defined(__HIP_PLATFORM_AMD__)
    return reinterpret_cast<uint64_t>(surface);
#else
    return static_cast<uint64_t>(surface);
#endif
}

void wp_surface_object_destroy_device(void* context, uint64_t surface_handle)
{
    release_or_defer(ReleaseKind::SurfaceObject, context, surface_handle);
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped) { }

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle) { }

WP_API uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

WP_API void wp_surface_object_destroy_device(void* context, uint64_t surface_handle) { }

#endif  // WP_ENABLE_CUDA
