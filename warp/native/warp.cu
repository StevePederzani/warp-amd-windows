// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by AMD (ROCm Warp) and Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.

#include "warp.h"

#include "alloc_tracker.h"
#include "apic.h"
#include "apic_internal.h"
#include "cuda_util.h"
#include "error.h"
#include "scan.h"
#include "sort.h"

#include <chrono>
#include <cstdlib>
#include <fstream>

#if !defined(__HIP_PLATFORM_AMD__)
#include <nvPTXCompiler.h>
#include <nvrtc.h>
#if WP_ENABLE_MATHDX
#include <libcublasdx.h>
#include <libcufftdx.h>
#include <libcusolverdx.h>
#include <libmathdx.h>
#include <nvJitLink.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
// Kernel printf on HIP uses a print buffer that is attached to each module and flushed at every synchronization.
#include "hip_print_host.h"
#endif
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
// Sharing GPU memory between processes through NT handles. These are defined further down, next to the other IPC helpers.
static bool hip_ipc_available(int device);
#endif

#define check_any(result) (check_generic(result, __FILE__, __LINE__))
#define check_nvrtc(code) (check_nvrtc_result(code, __FILE__, __LINE__))
#if !defined(__HIP_PLATFORM_AMD__)
#define check_nvptx(code) (check_nvptx_result(code, __FILE__, __LINE__))
#else
#define check_nvptx(code) (false)
#endif
#if WP_ENABLE_MATHDX && !defined(__HIP_PLATFORM_AMD__)
#define check_nvjitlink(handle, code) (check_nvjitlink_result(handle, code, __FILE__, __LINE__))
#else
#define check_nvjitlink(handle, code) (false)
#endif
#define check_cufftdx(code) (check_cufftdx_result(code, __FILE__, __LINE__))
#define check_cublasdx(code) (check_cublasdx_result(code, __FILE__, __LINE__))
#define check_cusolver(code) (check_cusolver_result(code, __FILE__, __LINE__))
#define CHECK_ANY(code) \
{ \
    do { \
        bool out = (check_any(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUFFTDX(code) \
{ \
    do { \
        bool out = (check_cufftdx(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUBLASDX(code) \
{ \
    do { \
        bool out = (check_cublasdx(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}
#define CHECK_CUSOLVER(code) \
{ \
    do { \
        bool out = (check_cusolver(code)); \
        if(!out) { \
            return out; \
        } \
    } while(0); \
}

bool check_nvrtc_result(nvrtcResult result, const char* file, int line)
{
    if (result == NVRTC_SUCCESS)
        return true;

    const char* error_string = nvrtcGetErrorString(result);
    fprintf(stderr, "Warp NVRTC compilation error %u: %s (%s:%d)\n", unsigned(result), error_string, file, line);
    return false;
}

#if !defined(__HIP_PLATFORM_AMD__)
bool check_nvptx_result(nvPTXCompileResult result, const char* file, int line)
{
    if (result == NVPTXCOMPILE_SUCCESS)
        return true;

    const char* error_string;
    switch (result) {
    case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
        error_string = "Invalid compiler handle";
        break;
    case NVPTXCOMPILE_ERROR_INVALID_INPUT:
        error_string = "Invalid input";
        break;
    case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
        error_string = "Compilation failure";
        break;
    case NVPTXCOMPILE_ERROR_INTERNAL:
        error_string = "Internal error";
        break;
    case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
        error_string = "Out of memory";
        break;
    case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
        error_string = "Incomplete compiler invocation";
        break;
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
        error_string = "Unsupported PTX version";
        break;
    default:
        error_string = "Unknown error";
        break;
    }

    fprintf(stderr, "Warp PTX compilation error %u: %s (%s:%d)\n", unsigned(result), error_string, file, line);
    return false;
}
#endif

bool check_generic(int result, const char* file, int line)
{
    if (!result) {
        fprintf(stderr, "Error %d on %s:%d\n", (int)result, file, line);
        return false;
    } else {
        return true;
    }
}

struct DeviceInfo {
    static constexpr int kNameLen = 128;
    static constexpr int kArchLen = 32;

    CUdevice device = -1;
    CUuuid uuid = { 0 };
    int ordinal = -1;
    int pci_domain_id = -1;
    int pci_bus_id = -1;
    int pci_device_id = -1;
    char name[kNameLen] = "";
    char arch_str[kArchLen] = "";
    int arch = 0;
    int is_uva = 0;
    int pageable_memory_access = 0;
    int direct_managed_mem_access_from_host = 0;
    int host_native_atomic_supported = 0;
    int managed_memory = 0;
    int concurrent_managed_access = 0;
    int is_mempool_supported = 0;
    int sm_count = 0;
    int is_ipc_supported = -1;
    int max_smem_bytes = 0;
    // Set when managed memory is emulated with mapped host memory, on an APU whose driver has no managed allocations
    // (see wp_alloc_device_managed).
    int hip_managed_on_host = 0;
#if defined(__HIP_PLATFORM_AMD__)
    // A logical device (WARP_HIP_DEVICES=n shows each card as n): a share of one card's memory, with a context of its own and a
    // memory pool of its own, its kernels running on the whole card. `device` is still the card, so every call that asks the card
    // about itself asks the right one; `ordinal` is Warp's number for this share. Zero for a whole card.
    int hip_part = 0;                  // this share's place among its card's, from 1
    int hip_parts = 0;                 // how many shares its card was split into
    size_t hip_memory_budget = 0;      // the bytes it may hold at once
    hipMemPool_t hip_pool = nullptr;   // its own pool, made with its context
#endif
    CUcontext primary_context = NULL;
};

struct ContextInfo {
    DeviceInfo* device_info = NULL;

    // the current stream, managed from Python (see wp_cuda_context_set_stream() and wp_cuda_context_get_stream())
    CUstream stream = NULL;

    // conditional graph node support, loaded on demand if the driver supports it (CUDA 12.4+)
    CUmodule conditional_module = NULL;
};

// Extra resources tied to a graph, freed after the graph is released by CUDA.
// Used with the on_graph_destroy() callback.
struct GraphDestroyCallbackInfo {
    void* context = NULL;  // graph CUDA context
    std::vector<void*> unfreed_allocs;  // graph allocations not freed by the graph
    std::vector<FreeInfo> tmp_allocs;  // temporary allocations owned by the graph (e.g., staged array fill values)
};

// Information for a graph allocation.
// If the allocation is not freed in the graph, it has a shared ownership:
// - The graph instance allocates/maps the memory on each launch, even if the user reference is released.
// - The user reference must remain valid even if the graph is destroyed.
// The memory will be freed once the user reference is released and the graph is destroyed.
struct GraphAllocInfo {
    uint64_t capture_id = 0;
    void* context = NULL;
    cudaGraphNode_t node = NULL;  // corresponding mem alloc node in the graph
    bool ref_exists = false;  // whether user reference still exists
    bool graph_destroyed = false;  // whether graph instance was destroyed
    bool is_static = false;  // Allocated outside stream order during a capture (HIP on Windows), so it has no node and is freed with cudaFree.
    size_t size = 0;  // Size in bytes, counted as graph memory by wp_cuda_device_get_graph_mem_current.
};

static std::unordered_map<CUfunction, std::string> g_kernel_names;

#if defined(__HIP_PLATFORM_AMD__)
// On an APU, Warp's CPU arrays come from pinned arenas. They are defined below, next to the managed-memory registry.
bool hip_host_arena_active();
void* hip_host_arena_alloc(size_t s);
bool hip_host_arena_free(void* ptr);
#endif

#if defined(__HIP_PLATFORM_AMD__)
// The number of registers the GPU reserves for each thread of a kernel. HIP's own count (hipFuncGetAttribute with NUM_REGS) is
// the number the code uses, which is 0 for an empty kernel, but the GPU always reserves at least one granule, and an occupancy
// calculation needs the reserved number, which is what NVIDIA's count reports. The kernel descriptor in the code object
// (`<name>.kd`, 64 bytes) records it in COMPUTE_PGM_RSRC1 as a granulated VGPR count at byte 48. A granule is 8 registers in
// 32-wide waves on RDNA and 4 in 64-wide waves, and bit 10 of kernel_code_properties (byte 56) says which applies. See AMDGPU's
// documentation of the kernel descriptor. The table is filled once per module when it loads, and a kernel finds its entry by name.
static std::unordered_map<CUmodule, std::unordered_map<std::string, int>> g_hip_module_vgprs;
static std::unordered_map<CUfunction, std::pair<CUmodule, int>> g_hip_kernel_vgprs;

static std::unordered_map<std::string, int> hip_reserved_vgprs(const char* image, size_t size)
{
    std::unordered_map<std::string, int> out;
    const unsigned char* b = (const unsigned char*)image;
    auto fits = [&](uint64_t at, uint64_t n) { return at <= size && n <= size - at; };
    auto u16 = [&](uint64_t at) { uint16_t v; memcpy(&v, b + at, 2); return (uint64_t)v; };
    auto u32 = [&](uint64_t at) { uint32_t v; memcpy(&v, b + at, 4); return (uint64_t)v; };
    auto u64 = [&](uint64_t at) { uint64_t v; memcpy(&v, b + at, 8); return v; };

    // The AMDGPU compiler writes code objects as 64-bit little-endian ELF files.
    if (!image || size < 64 || memcmp(b, "\x7f" "ELF", 4) != 0 || b[4] != 2 || b[5] != 1)
        return out;
    const uint64_t shoff = u64(0x28), shentsize = u16(0x3A), shnum = u16(0x3C);
    if (shentsize < 0x40 || !fits(shoff, shentsize * shnum))
        return out;
    auto section = [&](uint64_t k) { return shoff + k * shentsize; };

    for (uint64_t k = 0; k < shnum; ++k) {
        const uint64_t sh = section(k);
        const uint64_t type = u32(sh + 4);
        if (type != 2 && type != 11)  // SHT_SYMTAB, SHT_DYNSYM
            continue;
        const uint64_t sym_at = u64(sh + 0x18), sym_size = u64(sh + 0x20), link = u32(sh + 0x28);
        if (link >= shnum || !fits(sym_at, sym_size))
            continue;
        const uint64_t str_at = u64(section(link) + 0x18), str_size = u64(section(link) + 0x20);
        if (!fits(str_at, str_size))
            continue;
        for (uint64_t p = sym_at; p + 24 <= sym_at + sym_size; p += 24) {
            const uint64_t name_at = u32(p), shndx = u16(p + 6), value = u64(p + 8);
            if (!name_at || name_at >= str_size || shndx == 0 || shndx >= shnum)
                continue;
            const char* name = (const char*)b + str_at + name_at;
            const size_t name_len = strnlen(name, (size_t)(str_size - name_at));
            if (name_len < 4 || memcmp(name + name_len - 3, ".kd", 3) != 0)
                continue;
            // Map the symbol's address to its file offset through the section that contains it.
            const uint64_t holder = section(shndx);
            if (value < u64(holder + 0x10))
                continue;
            const uint64_t at = u64(holder + 0x18) + (value - u64(holder + 0x10));
            if (!fits(at, 64))
                continue;
            const uint64_t rsrc1 = u32(at + 48), code_props = u16(at + 56);
            const int granule = (code_props & (1u << 10)) ? 8 : 4;
            out[std::string(name, name_len - 3)] = (int)((rsrc1 & 0x3F) + 1) * granule;
        }
    }
    return out;
}

static void hip_forget_module_vgprs(CUmodule module)
{
    g_hip_module_vgprs.erase(module);
    for (auto it = g_hip_kernel_vgprs.begin(); it != g_hip_kernel_vgprs.end();)
        it = (it->second.first == module) ? g_hip_kernel_vgprs.erase(it) : std::next(it);
}
#endif  // defined(__HIP_PLATFORM_AMD__)

// cached info for all devices, indexed by ordinal
static std::vector<DeviceInfo> g_devices;

// maps CUdevice to DeviceInfo
static std::map<CUdevice, DeviceInfo*> g_device_map;

// cached info for all known contexts
static std::map<CUcontext, ContextInfo> g_contexts;
#if defined(__HIP_PLATFORM_AMD__)
// A logical device's own context (a card's second or later share) to its Warp ordinal (wp_cuda_device_get_primary_context).
static std::map<CUcontext, int> g_hip_context_part;
// Peer access and memory-pool access Warp turned on between two shares of one card, keyed (target, peer). Their memory is one card's
// and reachable either way; the state is the one Warp's API set, as between two GPUs, where both start off.
static std::map<std::pair<int, int>, int> g_hip_peer_on;
static std::map<std::pair<int, int>, int> g_hip_pool_access_on;
static bool hip_same_card(int a, int b)
{
    return a >= 0 && b >= 0 && a < int(g_devices.size()) && b < int(g_devices.size()) && a != b
        && g_devices[a].device == g_devices[b].device;
}

// A share's own memory pool on its card, made on first use, which keeps what it frees (as Warp keeps the default pool's); nullptr for
// a whole card or when HIP refuses the pool.
static hipMemPool_t hip_share_pool(DeviceInfo& d)
{
    if (d.hip_parts <= 1)
        return nullptr;
    if (!d.hip_pool) {
        hipMemPoolProps props = {};
        props.allocType = hipMemAllocationTypePinned;
        props.location.type = hipMemLocationTypeDevice;
        props.location.id = d.device;
        if (hipMemPoolCreate(&d.hip_pool, &props) == hipSuccess) {
            uint64_t keep = UINT64_MAX;
            ignore_cuda_error(hipMemPoolSetAttribute(d.hip_pool, hipMemPoolAttrReleaseThreshold, &keep));
        } else {
            ignore_cuda_error(hipGetLastError());
            d.hip_pool = nullptr;
        }
    }
    return d.hip_pool;
}

// The pool a Warp device allocates from: a share's own, or its card's default pool, asked by the card's number.
static cudaError_t hip_device_pool(cudaMemPool_t* pool, int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size())) {
        if (hipMemPool_t own = hip_share_pool(g_devices[ordinal])) {
            *pool = own;
            return cudaSuccess;
        }
        return cudaDeviceGetDefaultMemPool(pool, g_devices[ordinal].device);
    }
    return cudaDeviceGetDefaultMemPool(pool, ordinal);
}
#define WP_DEVICE_POOL(pool, ordinal) hip_device_pool(pool, ordinal)
#else
#define WP_DEVICE_POOL(pool, ordinal) cudaDeviceGetDefaultMemPool(pool, ordinal)
#endif

// cached info for all known streams (including registered external streams)
static std::unordered_map<CUstream, StreamInfo> g_streams;

// Ongoing graph captures registered using wp.capture_begin().
// This maps the capture id to the stream where capture was started.
// See wp_cuda_graph_begin_capture(), wp_cuda_graph_end_capture(), and wp_free_device_async().
static std::unordered_map<uint64_t, CaptureInfo*> g_captures;

#if defined(__HIP_PLATFORM_AMD__)
// HIP on Windows (PAL) collects a stream's commands into a batch and only submits the batch at a synchronization, at a query, when
// the batch is full, or at a graph launch. CUDA submits each command as soon as it is issued. Because of this, work issued to one
// stream and then left alone while other streams worked could stay unsubmitted. In test_event_external, the zero fill on the
// device stream only ran at the final synchronize, after ten launches of two graphs on two other streams, and it wiped their sums.
// Calling hipStreamQuery right after the fill returned 600 because the fill had not been submitted, and flushing before the
// launches made the test pass. Submitting every command (ROCclr's GPU_FLUSH_ON_EXECUTION=1) costs about 80 us per submission on
// WDDM, so a tiny launch takes 87 us instead of a few and a graph launch takes 1.9 ms. Instead, the stream that was last issued
// to is submitted when work is issued to a different stream. That matches CUDA's submission order across streams, costs one
// submission per switch, and costs nothing while a program stays on one stream. Nothing is submitted while a capture is open,
// because a query would end the capture.
static void* const g_hip_no_stream = reinterpret_cast<void*>(~uintptr_t(0));
static std::atomic<void*> g_hip_last_issue { g_hip_no_stream };
static std::mutex g_hip_issue_mutex;

// Submit a stream's pending batch now. A stream with an open capture is left alone.
static bool hip_external_capture_on(void* context);  // Defined below, next to release_or_defer.

static void hip_submit(void* stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status) != hipSuccess) {
        ignore_cuda_error(cudaGetLastError());
        return;
    }
    if (status != hipStreamCaptureStatusNone)
        return;
    // Also skip it while a capture that Warp was not told about is running on the context's current stream, such as a framework's
    // capture in global mode. A stream query there is an unsafe call. It ended PyTorch's capture around a Warp radix sort, because
    // the sort's copy switched streams and the switch submitted the side stream that held the sort's scratch memory.
    if (hip_external_capture_on(NULL))
        return;
    // The result is either success or 600 (not ready). Neither is an error, so it is not kept as the last error.
    (void)hipStreamQuery(static_cast<hipStream_t>(stream));
}

// Make sure a batch that ends with an event record gets submitted. CLR's hipStreamQuery only asks the queue to drain when the
// stream's last command has no hardware event yet (rocclr Event::notifyCmdQueue says "If HW event was assigned, then notification
// can be ignored"), and an event record gets its hardware event as soon as it is recorded. So a query right after an event record
// left the batch unsubmitted until the next synchronization, and the event then reported the synchronization's time. In
// test_stream_priority_timings, the high stream read the low stream's end time in two runs of three, and the start events of two
// streams disagreed by 11 ms in 4 runs of 6. Issuing a 4-byte fill on the stream after the event and then querying submitted the
// batch every time (6 of 6, 3 us on the host), because the fill has no hardware event yet. The fill writes to a scratch word on
// the device that nothing reads.
static std::mutex g_hip_flush_word_mutex;
static std::unordered_map<int, void*> g_hip_flush_word;  // Holds one scratch word per device.

static void hip_submit_after_event(void* stream)
{
    if (!stream || hip_external_capture_on(NULL))
        return;
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status) != hipSuccess) {
        ignore_cuda_error(cudaGetLastError());
        return;
    }
    if (status != hipStreamCaptureStatusNone)
        return;
    int device = 0;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(cudaGetLastError());
        return;
    }
    void* word = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hip_flush_word_mutex);
        auto it = g_hip_flush_word.find(device);
        if (it != g_hip_flush_word.end()) {
            word = it->second;
        } else {
            int current = 0;
            (void)hipGetDevice(&current);
            if (current != device)
                (void)hipSetDevice(device);
            if (hipMalloc(&word, 256) != hipSuccess) {
                ignore_cuda_error(cudaGetLastError());
                word = nullptr;
            }
            if (current != device)
                (void)hipSetDevice(current);
            g_hip_flush_word[device] = word;  // Null if the allocation failed, in which case only the query below is used.
        }
    }
    if (word && hipMemsetAsync(word, 0, 4, static_cast<hipStream_t>(stream)) != hipSuccess)
        ignore_cuda_error(cudaGetLastError());
    (void)hipStreamQuery(static_cast<hipStream_t>(stream));
}

// Set once this process exports or opens an IPC handle. After that, every command is submitted as soon as it is issued
// (hip_after_issue), the way CUDA does it. In test_ipc_multiprocess_write, the parent doubled its array and spawned a child that
// doubled it again, and because the parent's launch was still held in its stream's batch, the two raced on the same memory and
// produced 84 instead of 168.
static std::atomic<bool> g_hip_shared_work { false };

void hip_after_issue(void* stream)
{
    if (stream && g_hip_shared_work.load(std::memory_order_relaxed))
        hip_submit(stream);
}

void hip_issue(void* stream)
{
    if (g_hip_last_issue.load(std::memory_order_relaxed) == stream)
        return;
    std::lock_guard<std::mutex> lock(g_hip_issue_mutex);
    void* const last = g_hip_last_issue.exchange(stream, std::memory_order_relaxed);
    if (last == stream || last == g_hip_no_stream || !g_captures.empty())
        return;
    hip_submit(last);
}

void hip_forget_stream(void* stream)
{
    std::lock_guard<std::mutex> lock(g_hip_issue_mutex);
    void* expected = stream;
    g_hip_last_issue.compare_exchange_strong(expected, g_hip_no_stream, std::memory_order_relaxed);
}

// Stream priorities, implemented in software. HIP accepts a stream priority, but the Radeon 8060S under Windows never acts on it.
// Two ALU-bound kernels on streams with priority -1 and priority 0 finish together (147.8 and 147.9 ms), and so do two copies
// (24.13 and 24.31 ms). So the priorities are enforced here, on the GPU.
//   - Work issued to a high-priority stream (priority -1) is wrapped by two one-thread kernels. The first adds one to a counter in
//     device memory and the second subtracts one, so the counter holds the amount of high-priority work running right now. It
//     counts work only once it starts and never while it waits, so a high-priority stream that waits on a lower stream's work never
//     holds that work back. The wrapper is submitted together with its work, because HIP on Windows batches commands and an
//     opening kernel submitted alone would hold up the lower streams.
//   - Work issued to any other stream on that device, while the last work of a high-priority stream is still in flight (checked
//     with an event query on the host), first waits on the GPU behind a one-thread gate kernel. The gate waits until the counter
//     has stayed at zero for about a millisecond, because the gap between two pieces of one stream's work is not idleness. Traces
//     from the gate on the GPU's clock showed 256 us between a high stream's decrement and its next increment, since each
//     submitted batch starts late on WDDM, and a 48 us quiet window let the low-priority work through. The gate gives up after
//     about a second no matter what the counter says, so it can never hang the GPU.
//   - Nothing is wrapped or gated while a stream is being captured into a graph, or while no high-priority stream exists.
// Priorities are clamped to CUDA's two levels, 0 and -1 (see wp_cuda_stream_create).
namespace {
const char* const k_hip_priority_source = R"(
// WARP_HIP_PRIORITY_TRACE=1: each bracket and gate notes when it ran and what it saw, on the GPU's steady clock (wall_clock64), into a
// host-mapped buffer printed at the next synchronization: [0] the count of notes, then (time, kind << 32 | value) pairs
extern "C" __device__ unsigned long long* wp_prio_trace = 0;

__device__ inline void wp_prio_note(unsigned int kind, unsigned int value)
{
    unsigned long long* t = wp_prio_trace;
    if (!t)
        return;
    const unsigned long long slot = __hip_atomic_fetch_add(t, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (slot < 4096ull) {
        t[1 + 2 * slot] = wall_clock64();
        t[2 + 2 * slot] = ((unsigned long long)kind << 32) | value;
    }
}

extern "C" __global__ void wp_priority_count(unsigned int* count, int delta)
{
    const unsigned int now = __hip_atomic_fetch_add(count, (unsigned int)delta, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + (unsigned int)delta;
    if (delta)  // (0 is the dispatch made as a high-priority stream is made)
        wp_prio_note(delta > 0 ? 1u : 2u, now);
}

extern "C" __global__ void wp_priority_gate(const unsigned int* count)
{
    wp_prio_note(3u, __hip_atomic_load(count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
    unsigned int quiet = 0, polls = 0;
    for (; quiet < 700u && polls < 400000u; ++polls) {  // (700 polls: about a millisecond of quiet)
        if (__hip_atomic_load(count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u)
            ++quiet;
        else
            quiet = 0;
        __builtin_amdgcn_s_sleep(64);
    }
    wp_prio_note(4u, polls);
}

// before each piece of a chunked lower-priority copy: waits while high-priority work runs, goes the moment none does
extern "C" __global__ void wp_priority_gate_strict(const unsigned int* count)
{
    wp_prio_note(5u, __hip_atomic_load(count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
    unsigned int polls = 0;
    for (; polls < 400000u; ++polls) {
        if (__hip_atomic_load(count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u)
            break;
        __builtin_amdgcn_s_sleep(64);
    }
    wp_prio_note(6u, polls);
}

// A lower-priority copy as one launch whose blocks yield to high-priority work, as CUDA schedules a high-priority stream's blocks
// ahead of a running copy's remaining blocks: before each 64 KiB piece it copies, a block waits while the counter says high-priority
// work runs, and goes the moment none does (giving up, as the gates do, after about a second). A copy of n16 16-byte words; each of a
// block's 256 threads moves 16 of a piece's 4096 words, all loads before all stores.
extern "C" __global__ void __launch_bounds__(256) wp_priority_copy(uint4* dst, const uint4* src, unsigned long long n16,
                                                                    const unsigned int* count)
{
    const unsigned long long per_piece = 4096ull;
    const unsigned long long pieces = (n16 + per_piece - 1ull) / per_piece;
    for (unsigned long long piece = blockIdx.x; piece < pieces; piece += gridDim.x) {
        if (threadIdx.x == 0u) {
            unsigned int polls = 0;
            while (__hip_atomic_load(count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 0u && polls < 400000u) {
                __builtin_amdgcn_s_sleep(64);
                ++polls;
            }
            if (polls)
                wp_prio_note(7u, polls);
        }
        __syncthreads();
        const unsigned long long base = piece * per_piece + threadIdx.x;
        uint4 v[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const unsigned long long i = base + (unsigned long long)k * 256ull;
            if (i < n16)
                v[k] = src[i];
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const unsigned long long i = base + (unsigned long long)k * 256ull;
            if (i < n16)
                dst[i] = v[k];
        }
    }
}
)";

struct HipPriorityCard {
    bool ready = false;
    bool failed = false;
    hipModule_t module = nullptr;
    hipFunction_t count = nullptr;
    hipFunction_t gate = nullptr;
    hipFunction_t gate_strict = nullptr;
    hipFunction_t copy = nullptr;                          // wp_priority_copy: a lower-priority copy that yields, in one launch.
    unsigned int copy_blocks = 0;                          // Its grid: four blocks a compute unit.
    unsigned int* running = nullptr;                       // Count of high-priority work running now, on the GPU.
    std::unordered_map<void*, hipEvent_t> last_high_work;  // Latest work of each high-priority stream, finished or not.
    unsigned long long* trace = nullptr;                   // Host-mapped trace records the kernels write when WARP_HIP_PRIORITY_TRACE is set.
    double ticks_per_ms = 0.0;                             // Rate of the GPU's steady clock.
};

std::mutex g_hip_priority_mutex;
std::unordered_map<void*, int> g_hip_stream_priority;  // Priority of each stream Warp created with a priority below 0.
std::unordered_map<void*, int> g_hip_high_stream_device;  // Device of each of those streams.
std::unordered_map<int, int> g_hip_high_streams_on;  // Number of those streams on each device.
std::unordered_map<int, HipPriorityCard> g_hip_priority_cards;
std::atomic<int> g_hip_high_streams { 0 };

bool hip_stream_capturing(void* stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return true;  // If the status is unknown, treat the stream as capturing so nothing is added to it.
    }
    return status != hipStreamCaptureStatusNone;
}

bool hip_priority_trace();  // Defined below.

// Creates the wrapper and gate kernels and the counter for a GPU the first time a priority is enforced on it. Call it with
// g_hip_priority_mutex held.
HipPriorityCard* hip_priority_card(int device)
{
    HipPriorityCard& card = g_hip_priority_cards[device];
    if (card.ready)
        return &card;
    if (card.failed)
        return nullptr;
    card.failed = true;  // Stays true unless every step below succeeds.
    char arch[64] = "";
    hipDeviceProp_t prop = {};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return nullptr;
    }
    snprintf(arch, sizeof(arch), "--offload-arch=%s", prop.gcnArchName);
    // Use the processor name alone. The compiler infers the features that follow ':' in a target ID.
    if (char* colon = strchr(arch + 15, ':'))
        *colon = '\0';
    hiprtcProgram prog = nullptr;
    if (hiprtcCreateProgram(&prog, k_hip_priority_source, "wp_priority.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS)
        return nullptr;
    const char* options[] = { arch, "-O3" };
    std::vector<char> code;
    if (hiprtcCompileProgram(prog, 2, options) == HIPRTC_SUCCESS) {
        size_t size = 0;
        if (hiprtcGetCodeSize(prog, &size) == HIPRTC_SUCCESS && size) {
            code.resize(size);
            if (hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS)
                code.clear();
        }
    }
    hiprtcDestroyProgram(&prog);
    if (code.empty()) {
        fprintf(stderr, "Warp HIP warning: stream priorities could not be kept (their kernels did not compile)\n");
        return nullptr;
    }
    if (hipModuleLoadData(&card.module, code.data()) != hipSuccess
        || hipModuleGetFunction(&card.count, card.module, "wp_priority_count") != hipSuccess
        || hipModuleGetFunction(&card.gate, card.module, "wp_priority_gate") != hipSuccess
        || hipModuleGetFunction(&card.gate_strict, card.module, "wp_priority_gate_strict") != hipSuccess
        || hipModuleGetFunction(&card.copy, card.module, "wp_priority_copy") != hipSuccess
        || hipMalloc((void**)&card.running, sizeof(unsigned int)) != hipSuccess
        || hipMemset(card.running, 0, sizeof(unsigned int)) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        fprintf(stderr, "Warp HIP warning: stream priorities could not be kept (their kernels did not load)\n");
        return nullptr;
    }
    if (hip_priority_trace()) {
        // Trace records go to a host-mapped buffer, and its device address is written into the module's wp_prio_trace.
        void* host = nullptr;
        void* dev_ptr = nullptr;
        hipDeviceptr_t global = 0;
        size_t global_size = 0;
        int khz = 0;
        if (hipHostMalloc(&host, sizeof(unsigned long long) * (1 + 2 * 4096), hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess
            && hipHostGetDevicePointer(&dev_ptr, host, 0) == hipSuccess
            && hipModuleGetGlobal(&global, &global_size, card.module, "wp_prio_trace") == hipSuccess
            && hipMemcpyHtoD(global, &dev_ptr, sizeof(dev_ptr)) == hipSuccess) {
            memset(host, 0, sizeof(unsigned long long) * (1 + 2 * 4096));
            card.trace = static_cast<unsigned long long*>(host);
            if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) == hipSuccess && khz > 0)
                card.ticks_per_ms = double(khz);
        } else {
            ignore_cuda_error(hipGetLastError());
            fprintf(stderr, "[priority] the kernels' trace could not be made\n");
        }
    }
    card.copy_blocks = 4u * (unsigned int)(prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 16);
    card.ready = true;
    card.failed = false;
    return &card;
}

bool hip_launch_one(hipFunction_t f, void* stream, void** params)
{
    return hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, static_cast<hipStream_t>(stream), params, nullptr) == hipSuccess;
}

// With WARP_HIP_PRIORITY_TRACE=1, every wrapper and gate decision is written to stderr, showing what was done with each piece of work.
bool hip_priority_trace()
{
    static const bool on = [] {
        const char* v = getenv("WARP_HIP_PRIORITY_TRACE");
        return v && v[0] == '1';
    }();
    return on;
}
}  // namespace

bool hip_priority_before(void* stream)
{
    if (g_hip_high_streams.load(std::memory_order_relaxed) == 0 || !stream) {
        if (hip_priority_trace())
            fprintf(stderr, "[priority] stream %p: %s\n", stream, stream ? "no high-priority stream exists" : "the null stream");
        return false;
    }
    if (hip_stream_capturing(stream)) {
        if (hip_priority_trace())
            fprintf(stderr, "[priority] stream %p: capturing (nothing added)\n", stream);
        return false;
    }
    int device = 0;
    const hipError_t got_device = hipStreamGetDevice(static_cast<hipStream_t>(stream), &device);
    if (got_device != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        if (hip_priority_trace())
            fprintf(stderr, "[priority] stream %p: hipStreamGetDevice failed (%d)\n", stream, int(got_device));
        return false;
    }
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    auto prio = g_hip_stream_priority.find(stream);
    const bool high = prio != g_hip_stream_priority.end() && prio->second < 0;
    HipPriorityCard* card = hip_priority_card(device);
    if (!card) {
        if (hip_priority_trace())
            fprintf(stderr, "[priority] stream %p: no priority kernels on device %d\n", stream, device);
        return false;
    }
    if (high) {
        int delta = 1;
        void* params[] = { &card->running, &delta };
        if (!hip_launch_one(card->count, stream, params)) {
            const hipError_t e = hipGetLastError();
            if (hip_priority_trace())
                fprintf(stderr, "[priority] high stream %p: the +1 bracket failed to launch (%d)\n", stream, int(e));
            return false;
        }
        if (hip_priority_trace())
            fprintf(stderr, "[priority] high stream %p: +1\n", stream);
        return true;
    }
    // A lower-priority stream is gated only while a high-priority stream on this device has work in flight.
    bool in_flight = false;
    int queried = 0;
    hipError_t last_query = hipSuccess;
    for (auto& entry : card->last_high_work) {
        const hipError_t q = hipEventQuery(entry.second);
        ++queried;
        last_query = q;
        if (q == hipErrorNotReady)
            in_flight = true;
        ignore_cuda_error(hipGetLastError());
        if (in_flight)
            break;
    }
    bool gated = false;
    hipError_t gate_error = hipSuccess;
    if (in_flight) {
        void* params[] = { &card->running };
        gated = hip_launch_one(card->gate, stream, params);
        if (!gated)
            gate_error = hipGetLastError();
    }
    if (hip_priority_trace())
        fprintf(stderr, "[priority] stream %p: %d high stream(s) known, %d queried (last %d), %s%s\n", stream,
                int(card->last_high_work.size()), queried, int(last_query), in_flight ? "high work in flight: " : "no high work in flight",
                in_flight ? (gated ? "gated" : "the gate failed to launch") : "");
    (void)gate_error;
    return false;
}

void hip_priority_after(void* stream)
{
    int device = 0;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return;
    }
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    HipPriorityCard* card = hip_priority_card(device);
    if (!card)
        return;
    int delta = -1;
    void* params[] = { &card->running, &delta };
    if (!hip_launch_one(card->count, stream, params)) {
        const hipError_t e = hipGetLastError();
        if (hip_priority_trace())
            fprintf(stderr, "[priority] high stream %p: the -1 bracket failed to launch (%d)\n", stream, int(e));
        return;
    }
    if (hip_priority_trace())
        fprintf(stderr, "[priority] high stream %p: -1, its last work recorded\n", stream);
    hipEvent_t& done = card->last_high_work[stream];
    if (!done && hipEventCreateWithFlags(&done, hipEventDisableTiming) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        done = nullptr;
    }
    if (done && hipEventRecord(done, static_cast<hipStream_t>(stream)) != hipSuccess)
        ignore_cuda_error(hipGetLastError());
    // Submit the wrapper and its work to the GPU together now. HIP on Windows holds a stream's commands until something submits
    // them, and a query alone leaves a batch that ends in an event unsubmitted (see hip_submit_after_event).
    hip_submit_after_event(stream);
}

// Prints the trace records the wrapper and gate kernels wrote since the last synchronization, in GPU time order
// (WARP_HIP_PRIORITY_TRACE).
static void hip_priority_dump()
{
    if (!hip_priority_trace())
        return;
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    static const char* const names[] = { "?", "count +1, now", "count -1, now", "gate starts, count is", "gate ends after polls",
                                         "strict gate starts, count is", "strict gate ends after polls",
                                         "a yielding copy's block waited, polls" };
    for (auto& kv : g_hip_priority_cards) {
        HipPriorityCard& card = kv.second;
        if (!card.trace)
            continue;
        unsigned long long n = __atomic_load_n(&card.trace[0], __ATOMIC_ACQUIRE);
        if (!n)
            continue;
        if (n > 4096ull)
            n = 4096ull;
        std::vector<std::pair<unsigned long long, unsigned long long>> notes;
        for (unsigned long long i = 0; i < n; ++i)
            notes.emplace_back(card.trace[1 + 2 * i], card.trace[2 + 2 * i]);
        std::sort(notes.begin(), notes.end());
        const double per_ms = card.ticks_per_ms > 0.0 ? card.ticks_per_ms : 100000.0;
        for (const auto& note : notes) {
            const unsigned kind = unsigned(note.second >> 32);
            fprintf(stderr, "[priority:card] %10.3f ms  %s %u\n", double(note.first - notes.front().first) / per_ms,
                    names[kind < 8 ? kind : 0], unsigned(note.second & 0xffffffffu));
        }
        __atomic_store_n(&card.trace[0], 0ull, __ATOMIC_RELEASE);
    }
}

// A copy on a lower-priority stream yields to high-priority work, as it does on CUDA. On CUDA, a copy within one device runs as
// thread blocks, and a high-priority stream's blocks are scheduled ahead of the low copy's remaining blocks, whichever was issued
// first. Here a lower-priority copy within the device runs as one kernel whose blocks wait, before each 64 KiB they copy, while any
// high-priority work runs (wp_priority_copy); any other lower-priority copy is split into pieces, and each piece waits behind a
// strict gate (wp_priority_gate_strict, which waits while any high-priority work runs and continues as soon as none does).
// High-priority work that starts after the copy began still runs ahead of the rest of the copy. Every such copy on a device with a
// high-priority stream yields this way, even if no high-priority work is in flight when it is issued. test_stream_priority_timings
// issues its first low copy before the high stream's first copy, and when that copy ran in one piece it ran alongside both high
// copies (25.44 and 25.42 ms). A block or gate that finds no high-priority work continues immediately.
static const size_t k_hip_priority_chunk = size_t(8) << 20;

// Returns true if a copy issued to this stream should yield to high-priority work, which is the case for a lower-priority stream
// on a device that has a high-priority stream.
static bool hip_priority_copy_yields(void* stream)
{
    if (g_hip_high_streams.load(std::memory_order_relaxed) == 0 || !stream || hip_stream_capturing(stream))
        return false;
    int device = 0;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return false;
    }
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    auto prio = g_hip_stream_priority.find(stream);
    if (prio != g_hip_stream_priority.end() && prio->second < 0)
        return false;
    auto highs = g_hip_high_streams_on.find(device);
    if (highs == g_hip_high_streams_on.end() || highs->second <= 0)
        return false;
    HipPriorityCard* card = hip_priority_card(device);
    return card && card->gate_strict;
}

// Issues a strict gate before the next piece of a split copy.
static void hip_priority_gate_strict(void* stream)
{
    int device = 0;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return;
    }
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    HipPriorityCard* card = hip_priority_card(device);
    if (!card || !card->gate_strict)
        return;
    void* params[] = { &card->running };
    if (!hip_launch_one(card->gate_strict, stream, params))
        ignore_cuda_error(hipGetLastError());
}

// Issues a copy within the device on a lower-priority stream as one yielding kernel (wp_priority_copy), the 16-byte words through
// the kernel and any tail through the runtime, both on the stream. Returns 1 when the copy was issued, 0 when issuing it failed, and
// -1, having issued nothing, when the kernel cannot take the copy (an unaligned address, no kernel on the device), and the caller
// copies in gated pieces instead.
// Why one launch: split into gated pieces, a 512 MiB copy is 64 gates and 64 copies, about 128 calls into HIP. The host spent
// milliseconds issuing them, and high-priority work issued after the copy could not reach the GPU until they were all issued, so
// the low copy ran alone meanwhile: traced, 3.4 ms before the high stream's first copy began and 4.5 ms between its two copies,
// and the high stream finished second in 3 of 10 runs of test_stream_priority_timings' copies with nothing else on the card.
static int hip_priority_copy_kernel(void* dest, const void* src, size_t n, CUstream stream)
{
    if (((reinterpret_cast<uintptr_t>(dest) | reinterpret_cast<uintptr_t>(src)) & 15u) != 0 || n < 16)
        return -1;
    int device = 0;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return -1;
    }
    hipFunction_t copy = nullptr;
    unsigned int blocks = 0;
    unsigned int* running = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
        HipPriorityCard* card = hip_priority_card(device);
        if (!card || !card->copy || !card->copy_blocks)
            return -1;
        copy = card->copy;
        blocks = card->copy_blocks;
        running = card->running;
    }
    unsigned long long n16 = n / 16;
    const unsigned long long pieces = (n16 + 4095ull) / 4096ull;
    const unsigned int grid = static_cast<unsigned int>(std::min<unsigned long long>(blocks, pieces));
    void* d = dest;
    const void* s = src;
    void* params[] = { &d, &s, &n16, &running };
    if (hip_priority_trace())
        fprintf(stderr, "[priority] stream %p: a %zu-byte copy that yields to high-priority work, as one kernel of %u blocks\n",
                (void*)stream, n, grid);
    if (hipModuleLaunchKernel(copy, grid, 1, 1, 256, 1, 1, 0, static_cast<hipStream_t>(stream), params, nullptr) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return -1;
    }
    const size_t done = size_t(n16) * 16;
    if (done < n) {
        // (the bytes past the last whole word, after the kernel on the same stream)
        if (!check_cuda(cudaMemcpyAsync(static_cast<char*>(dest) + done, static_cast<const char*>(src) + done, n - done,
                                        cudaMemcpyDeviceToDevice, stream)))
            return 0;
    }
    return 1;
}

// Copies on a lower-priority stream of a device with a high-priority stream so that the copy yields to high-priority work: a copy
// within the device as one yielding kernel, any other in pieces behind strict gates; in one piece otherwise.
static bool hip_priority_copy(void* dest, const void* src, size_t n, cudaMemcpyKind kind, CUstream stream)
{
    if (n <= k_hip_priority_chunk || !hip_priority_copy_yields(stream))
        return check_cuda(cudaMemcpyAsync(dest, src, n, kind, stream));
    if (kind == cudaMemcpyDeviceToDevice) {
        const int issued = hip_priority_copy_kernel(dest, src, n, stream);
        if (issued >= 0)
            return issued == 1;
    }
    if (hip_priority_trace())
        fprintf(stderr, "[priority] stream %p: a %zu-byte copy that yields to high-priority work, in %zu pieces\n", (void*)stream, n,
                (n + k_hip_priority_chunk - 1) / k_hip_priority_chunk);
    for (size_t off = 0; off < n; off += k_hip_priority_chunk) {
        hip_priority_gate_strict(stream);
        const size_t len = std::min(k_hip_priority_chunk, n - off);
        if (!check_cuda(cudaMemcpyAsync(static_cast<char*>(dest) + off, static_cast<const char*>(src) + off, len, kind, stream)))
            return false;
    }
    return true;
}

// Registers a stream Warp created with a priority (wp_cuda_stream_create) and unregisters it when the stream is destroyed
// (wp_cuda_stream_destroy).
static void hip_priority_stream_made(void* stream, int priority)
{
    if (priority >= 0)
        return;
    int device = -1;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &device) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        device = -1;
    }
    hipFunction_t count = nullptr, copy = nullptr, gate_strict = nullptr;
    unsigned int* running = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
        g_hip_stream_priority[stream] = priority;
        g_hip_high_stream_device[stream] = device;
        ++g_hip_high_streams_on[device];
        g_hip_high_streams.fetch_add(1, std::memory_order_relaxed);
        if (device >= 0) {
            if (HipPriorityCard* card = hip_priority_card(device)) {
                count = card->count;
                copy = card->copy;
                gate_strict = card->gate_strict;
                running = card->running;
            }
        }
    }
    // The stream is made ready as it is made, as CUDA's is: the priority kernels are compiled (about 120 ms, once a device) and each
    // is dispatched once here, with nothing to do, before the stream is handed back. Made on first use instead, inside the work, both
    // costs fell inside test_stream_priority_timings' timed window: cold, both streams took 133-146 ms for copies that take 14-26.
    if (count && copy && gate_strict) {
        hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
        (void)hipThreadExchangeStreamCaptureMode(&mode);
        int zero = 0;
        void* count_params[] = { &running, &zero };
        unsigned long long none = 0;
        void* no_dst = nullptr;
        const void* no_src = nullptr;
        void* copy_params[] = { &no_dst, &no_src, &none, &running };
        void* gate_params[] = { &running };
        const hipStream_t s = static_cast<hipStream_t>(stream);
        const bool ok = hipModuleLaunchKernel(count, 1, 1, 1, 1, 1, 1, 0, s, count_params, nullptr) == hipSuccess
            && hipModuleLaunchKernel(copy, 1, 1, 1, 256, 1, 1, 0, s, copy_params, nullptr) == hipSuccess
            && hipModuleLaunchKernel(gate_strict, 1, 1, 1, 1, 1, 1, 0, s, gate_params, nullptr) == hipSuccess
            && hipStreamSynchronize(s) == hipSuccess;
        if (!ok)
            ignore_cuda_error(hipGetLastError());
        (void)hipThreadExchangeStreamCaptureMode(&mode);
    }
}

static void hip_priority_stream_gone(void* stream)
{
    std::lock_guard<std::mutex> lock(g_hip_priority_mutex);
    if (g_hip_stream_priority.erase(stream))
        g_hip_high_streams.fetch_sub(1, std::memory_order_relaxed);
    auto on = g_hip_high_stream_device.find(stream);
    if (on != g_hip_high_stream_device.end()) {
        auto highs = g_hip_high_streams_on.find(on->second);
        if (highs != g_hip_high_streams_on.end() && --highs->second <= 0)
            g_hip_high_streams_on.erase(highs);
        g_hip_high_stream_device.erase(on);
    }
    for (auto& card : g_hip_priority_cards) {
        auto it = card.second.last_high_work.find(stream);
        if (it != card.second.last_high_work.end()) {
            if (it->second)
                ignore_cuda_error(hipEventDestroy(it->second));
            card.second.last_high_work.erase(it);
        }
    }
}
#endif

// Memory allocated during graph capture requires special handling.
// See wp_alloc_device_async() and wp_free_device_async().
static std::unordered_map<void*, GraphAllocInfo> g_graph_allocs;

// Allocations of captures Warp doesn't track, e.g. conditional bodies; their graph may never run.
static std::unordered_set<void*> g_untracked_graph_allocs;

#if defined(__HIP_PLATFORM_AMD__)
// Track allocation stream for each mempool allocation so that
// wp_free_device_async can create proper inter-stream dependencies.
// On CUDA, cudaFreeAsync(ptr, NULL) provides this implicitly via
// the legacy null stream; HIP requires explicit event dependencies.
// A deferred release can run on a garbage-collecting thread, which 1.19's deferred list allows, so the map has its own lock.
static std::unordered_map<void*, CUstream> g_alloc_streams;
static std::mutex g_alloc_streams_mutex;
#endif

// Releases that would invalidate a graph capture, queued until all captures end.
struct ReleaseInfo {
    ReleaseKind kind;
    void* context;  // NULL for releases that are not tied to a context
    uint64_t handle;
};
static std::vector<ReleaseInfo> g_deferred_releases;
// Garbage collection can release resources from any Python thread.
static std::mutex g_deferred_releases_mutex;

// Data from on_graph_destroy() callbacks that run on a different thread.
static std::vector<GraphDestroyCallbackInfo*> g_deferred_graph_destroy_list;
static std::mutex g_graph_destroy_mutex;


void wp_cuda_set_context_restore_policy(bool always_restore) { ContextGuard::always_restore = always_restore; }

int wp_cuda_get_context_restore_policy() { return int(ContextGuard::always_restore); }

#if defined(__HIP_PLATFORM_AMD__)
// WARP_HIP_DEVICES=n: each card is shown as n logical devices (DeviceInfo::hip_part). Each holds an equal share of the card's memory
// and runs its kernels on the whole card, since HIP on Windows holds a stream to no part of it (wp_cuda_stream_create). The card's
// own record stays the template: everything a logical device asks of HIP about itself it asks of its card (`device`), and Warp
// numbers the shares from 0 (`ordinal`). The device map points each card at its first share; a share's own context says which
// share it is (hip_context_part).
static void hip_split_devices()
{
    const char* env = getenv("WARP_HIP_DEVICES");
    const int n = env ? atoi(env) : 1;
    if (n <= 1 || g_devices.empty())
        return;
    std::vector<DeviceInfo> parts;
    parts.reserve(g_devices.size() * size_t(n));
    for (const DeviceInfo& card : g_devices) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        int prev = 0;
        ignore_cuda_error(hipGetDevice(&prev));
        if (hipSetDevice(card.device) == hipSuccess)
            ignore_cuda_error(hipMemGetInfo(&free_bytes, &total_bytes));
        else
            ignore_cuda_error(hipGetLastError());
        ignore_cuda_error(hipSetDevice(prev));
        // (each share reports all of the card's multiprocessors, since its kernels run on all of them; the shares divide the memory)
        const int shares = n;
        for (int k = 0; k < shares; ++k) {
            DeviceInfo d = card;
            d.ordinal = int(parts.size());
            d.hip_part = k + 1;
            d.hip_parts = shares;
            d.hip_memory_budget = total_bytes / size_t(shares);
            char base[DeviceInfo::kNameLen];
            snprintf(base, sizeof(base), "%s", card.name);
            snprintf(d.name, DeviceInfo::kNameLen, "%s (part %d of %d)", base, k + 1, shares);
            d.primary_context = NULL;
            parts.push_back(d);
        }
    }
    g_devices = std::move(parts);
    g_device_map.clear();
    for (DeviceInfo& d : g_devices)
        if (d.hip_part <= 1)
            g_device_map[d.device] = &d;
}
#endif

int cuda_init()
{
    if (!init_cuda_driver())
        return -1;

    int device_count = 0;
    if (check_cu(cuDeviceGetCount_f(&device_count))) {
        g_devices.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            CUdevice device;
            if (check_cu(cuDeviceGet_f(&device, i))) {
                // query device info
                g_devices[i].device = device;
                g_devices[i].ordinal = i;
                check_cu(cuDeviceGetName_f(g_devices[i].name, DeviceInfo::kNameLen, device));
                check_cu(cuDeviceGetUuid_f(&g_devices[i].uuid, device));
                check_cu(
                    cuDeviceGetAttribute_f(&g_devices[i].pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device)
                );
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
                check_cu(
                    cuDeviceGetAttribute_f(&g_devices[i].pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device)
                );
                check_cu(cuDeviceGetAttribute_f(&g_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].pageable_memory_access, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device
                ));
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].direct_managed_mem_access_from_host,
                    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device
                ));
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].host_native_atomic_supported, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device
                ));
                check_cu(
                    cuDeviceGetAttribute_f(&g_devices[i].managed_memory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device)
                );
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].concurrent_managed_access, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device
                ));
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].is_mempool_supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device
                ));
#if defined(__HIP_PLATFORM_AMD__)
                // Managed memory on an APU. HIP under Windows has no managed allocations (hipDeviceAttributeManagedMemory is 0 on
                // the Radeon 8060S), but on an integrated GPU the GPU and the CPU share one physical memory, so host memory mapped
                // into the GPU's address space is a single pointer that both can read and write, with nothing to migrate. Such a
                // device offers managed memory built that way (wp_alloc_device_managed) and reports it the way CUDA under Windows
                // reports its own. The memory is managed and the CPU can read it directly, but not concurrently with the GPU, so the
                // CPU reads after a synchronization, as Warp's checked mode requires.
                {
                    int integrated = 0, can_map = 0;
                    ignore_cuda_error(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device));
                    ignore_cuda_error(hipDeviceGetAttribute(&can_map, hipDeviceAttributeCanMapHostMemory, device));
                    if (!g_devices[i].managed_memory && integrated && can_map && g_devices[i].is_uva) {
                        g_devices[i].managed_memory = 1;
                        g_devices[i].direct_managed_mem_access_from_host = 1;
                        g_devices[i].concurrent_managed_access = 0;
                        g_devices[i].hip_managed_on_host = 1;
                    }
                    // Warp's CPU arrays come from pinned arenas (hip_host_arena_alloc), so the GPU can access them at their own addresses.
                    if (integrated && can_map && g_devices[i].is_uva && hip_host_arena_active())
                        g_devices[i].pageable_memory_access = 1;
                }
#endif
                check_cu(
                    cuDeviceGetAttribute_f(&g_devices[i].sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
                );
#if !defined(__HIP_PLATFORM_AMD__) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
                int device_attribute_integrated = 0;
                check_cu(cuDeviceGetAttribute_f(&device_attribute_integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
                if (device_attribute_integrated == 0) {
                    check_cu(cuDeviceGetAttribute_f(
                        &g_devices[i].is_ipc_supported, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device
                    ));
                } else {
                    // integrated devices do not support CUDA IPC
                    g_devices[i].is_ipc_supported = 0;
                }
#else
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
                // IPC works through NT handles from a shareable memory pool (hip_ipc_alloc), if the driver can create one.
                g_devices[i].is_ipc_supported = hip_ipc_available(i) ? 1 : 0;
#elif defined(__HIP_PLATFORM_AMD__)
                g_devices[i].is_ipc_supported = 0;
                int prev_device = -1;
                if (hipGetDevice(&prev_device) == hipSuccess) {
                    if (hipSetDevice(i) == hipSuccess) {
                        void* ipc_ptr = nullptr;
                        if (hipMalloc(&ipc_ptr, 4) == hipSuccess) {
                            CUipcMemHandle mem_handle;
                            if (cuIpcGetMemHandle_f(&mem_handle, (CUdeviceptr)ipc_ptr) == CUDA_SUCCESS) {
                                g_devices[i].is_ipc_supported = 1;
                            }
                            ignore_cuda_error(hipFree(ipc_ptr));
                        }
                        ignore_cuda_error(hipSetDevice(prev_device));
                    }
                }
#else
                g_devices[i].is_ipc_supported = 0;
#endif
#endif
                check_cu(cuDeviceGetAttribute_f(
                    &g_devices[i].max_smem_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device
                ));
#if !defined(__HIP_PLATFORM_AMD__)
                int major = 0;
                int minor = 0;
                check_cu(cuDeviceGetAttribute_f(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                check_cu(cuDeviceGetAttribute_f(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                g_devices[i].arch = 10 * major + minor;
#ifdef CUDA_VERSION
#if CUDA_VERSION < 13000
                if (g_devices[i].arch == 110) {
                    g_devices[i].arch = 101;  // Thor SM change
                }
#endif
#endif
#else
                g_devices[i].arch = 0;
#endif
#if defined(__HIP_PLATFORM_AMD__)
                cudaDeviceProp props;
                if (check_cuda(cudaGetDeviceProperties(&props, i))) {
                    snprintf(g_devices[i].arch_str, DeviceInfo::kArchLen, "%s", props.gcnArchName);
                }
#else
                snprintf(g_devices[i].arch_str, DeviceInfo::kArchLen, "sm_%d", g_devices[i].arch);
#endif
                g_device_map[device] = &g_devices[i];
            } else {
                return -1;
            }
        }
    } else {
        return -1;
    }
#if defined(__HIP_PLATFORM_AMD__)
    hip_split_devices();
#endif

    // initialize default timing state
    static CudaTimingState default_timing_state(0, NULL);
    g_cuda_timing_state = &default_timing_state;

#if defined(__HIP_PLATFORM_AMD__)
    // Create the kernel print buffer now. When it was created lazily at a module's first load, it could be created inside the
    // capture that loaded the module, where host allocation is refused, and test_torch's test_warp_graph_warp_stream lost its
    // capture.
    if (device_count > 0)
        hip_print_init();
#endif

    return 0;
}


CUcontext get_current_context()
{
    CUcontext ctx;
    if (check_cu(cuCtxGetCurrent_f(&ctx)))
        return ctx;
    else
        return NULL;
}

static inline CUstream get_current_stream(void* context = NULL)
{
    return static_cast<CUstream>(wp_cuda_context_get_stream(context));
}

static ContextInfo* get_context_info(CUcontext ctx)
{
    if (!ctx) {
        ctx = get_current_context();
        if (!ctx)
            return NULL;
    }

    auto it = g_contexts.find(ctx);
    if (it != g_contexts.end()) {
        return &it->second;
    } else {
        // previously unseen context, add the info
        ContextGuard guard(ctx, true);

        CUdevice device;
        if (check_cu(cuCtxGetDevice_f(&device))) {
            DeviceInfo* device_info = g_device_map[device];
#if defined(__HIP_PLATFORM_AMD__)
            // (a card's later share: its context says which, hip_split_devices)
            auto part = g_hip_context_part.find(ctx);
            if (part != g_hip_context_part.end())
                device_info = &g_devices[part->second];
#endif

            // Work around a CUDA driver bug observed with Linux driver 535.54.03: cudaFreeAsync() could crash when
            // directly freeing a graph allocation on an uninitialized default stream. Prime the stream's allocator
            // bookkeeping with an ordinary asynchronous allocation and free.
#if !defined(__HIP_PLATFORM_AMD__)
            if (device_info->is_mempool_supported) {
                void* dummy = NULL;
                check_cuda(cudaMallocAsync(&dummy, 1, NULL));
                check_cuda(cudaFreeAsync(dummy, NULL));
            }
#endif

            ContextInfo context_info;
            context_info.device_info = device_info;
            auto result = g_contexts.insert(std::make_pair(ctx, context_info));
            return &result.first->second;
        }
    }

    return NULL;
}

static inline ContextInfo* get_context_info(void* context) { return get_context_info(static_cast<CUcontext>(context)); }

StreamInfo* get_stream_info(CUstream stream)
{
    auto it = g_streams.find(stream);
    if (it != g_streams.end())
        return &it->second;
    else
        return NULL;
}

CaptureInfo* get_capture_info(CUstream stream)
{
    if (!g_captures.empty() && wp_cuda_stream_is_capturing(stream)) {
        uint64_t capture_id = get_capture_id(stream);
        auto capture_iter = g_captures.find(capture_id);
        if (capture_iter != g_captures.end())
            return capture_iter->second;
    }
    return NULL;
}

CaptureInfo* find_capture_info(uint64_t capture_id)
{
    // Find the registered capture that owns the given capture id, either directly
    // (top-level capture) or as the parent of a child graph capture currently being
    // recorded. While a child body graph is captured, the parent capture's main
    // stream also reports the child's capture id (capture ids are stable across
    // pause/resume), which identifies the parent from any participating stream,
    // including forked streams that carry no capture info of their own.
    for (const auto& capture_iter : g_captures) {
        CaptureInfo* capture = capture_iter.second;
        if (capture && (capture->id == capture_id || get_capture_id(capture->stream) == capture_id))
            return capture;
    }
    return NULL;
}

static inline bool is_context_capturing(CUcontext context)
{
    if (g_captures.empty())
        return false;

    for (const auto& capture_iter : g_captures) {
        CaptureInfo* capture = capture_iter.second;
        if (capture && capture->context == context)
            return true;
    }

    return false;
}

// helper function to copy a value to device memory in a graph-friendly way
#if defined(__HIP_PLATFORM_AMD__)
// Warp relies on CUDA's behavior for copies between the device and pageable (not page-locked) host memory. A device-to-host copy
// into pageable memory has completed when cudaMemcpyAsync returns, since array.numpy() reads the host buffer right after it, and a
// host-to-device copy from pageable memory has consumed its source by then, so a temporary NumPy buffer can be released. HIP does
// not promise either for pageable memory, so on HIP such a copy is followed by a synchronization of its stream. Copies to or from
// pinned memory stay asynchronous, as on CUDA, and nothing is synchronized while the stream is being captured into a graph.
static bool hip_is_pageable_host_memory(const void* host_ptr)
{
    // Warp's CPU arrays from the pinned arenas (WARP_HIP_PINNED_CPU) are pinned memory to HIP but pageable memory to Warp's
    // Python side (array.pinned is False). wp.copy reads a staging buffer right after copying into it and frees one right after
    // copying out of it, and array.numpy() reads its host copy immediately. So these arrays keep the completed copies of pageable
    // memory, and only arrays allocated as pinned (wp_alloc_pinned, outside the arenas) copy asynchronously, as on CUDA.
    if (wp_hip_host_reachable(host_ptr))
        return true;
    hipPointerAttribute_t attr = {};
    if (hipPointerGetAttributes(&attr, host_ptr) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());  // Unknown pointers report an error because they are plain pageable memory.
        return true;
    }
    return attr.type != hipMemoryTypeHost && attr.type != hipMemoryTypeManaged && attr.type != hipMemoryTypeUnified;
}

static bool hip_complete_pageable_copy(CUstream stream, const void* host_ptr)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &status) == hipSuccess && status != hipStreamCaptureStatusNone)
        return true;
    if (!hip_is_pageable_host_memory(host_ptr))
        return true;
    const bool ok = check_cuda(hipStreamSynchronize(stream));
    // A blocking copy also flushes kernel prints on CUDA, and reports a kernel that trapped (hip_print.h trap).
    if (const int trapped = hip_print_flush())
        return check_cuda(static_cast<hipError_t>(trapped));
    return ok;
}
#endif

static bool capturable_tmp_alloc(void* context, const void* data, size_t size, void** devptr_ret, bool* free_devptr_ret)
{
    ContextGuard guard(context);

    CUstream stream = get_current_stream();
    CaptureInfo* capture_info = get_capture_info(stream);
    int device_ordinal = wp_cuda_context_get_device_ordinal(context);
    void* devptr = NULL;
    bool free_devptr = true;

    if (capture_info) {
        // ongoing graph capture - need to stage the fill value so that it persists with the graph
        //
        // Prefer pausing the capture (HIP/ROCm, and CUDA 12.4+) so the staging allocation and copy
        // happen outside the graph. Capturing the device allocation as a graph memory-allocation
        // node fails to instantiate on secondary devices (ordinal != 0) on HIP/ROCm (CUDA error
        // 801, "operation not supported"); staging outside the capture avoids creating such a node.
        bool staged_via_pause = false;
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION >= 12040
#if defined(__HIP_PLATFORM_AMD__)
        const bool can_pause_capture = true;
#else
        const bool can_pause_capture = wp_cuda_driver_version() >= 12040;
#endif
        if (can_pause_capture) {
            // pause the capture so that the alloc/memcpy won't be captured
            void* graph = NULL;
            if (!wp_cuda_graph_pause_capture(WP_CURRENT_CONTEXT, stream, &graph))
                return false;

            // copy value to device memory
            devptr = wp_alloc_device(WP_CURRENT_CONTEXT, size);
            if (!devptr) {
                fprintf(
                    stderr, "Warp error: Failed to allocate %llu bytes on device 'cuda:%d' (in function %s)\n",
                    (unsigned long long)size, device_ordinal, __FUNCTION__
                );
                return false;
            }
            if (!check_cuda(cudaMemcpyAsync(devptr, data, size, cudaMemcpyHostToDevice, stream)))
                return false;
#if defined(__HIP_PLATFORM_AMD__)
            if (!hip_complete_pageable_copy(stream, data))  // The capture is paused here.
                return false;
#endif

            // graph takes ownership of the value storage
            FreeInfo free_info;
            free_info.context = context ? context : get_current_context();
            free_info.ptr = devptr;
            free_info.is_async = wp_cuda_device_is_mempool_supported(device_ordinal);

            // allocation will be freed when graph is destroyed
            capture_info->tmp_allocs.push_back(free_info);

            // resume the capture
            if (!wp_cuda_graph_resume_capture(WP_CURRENT_CONTEXT, stream, graph))
                return false;

            free_devptr = false;  // memory is owned by the graph, doesn't need to be freed
            staged_via_pause = true;
        }
#endif
        if (!staged_via_pause)
        {
            // can't pause/resume the capture, so stage in CPU memory
            void* hostptr = wp_alloc_host(size);
            if (!hostptr) {
                fprintf(
                    stderr, "Warp error: Failed to allocate %llu bytes on device 'cpu' (in function %s)\n",
                    (unsigned long long)size, __FUNCTION__
                );
                return false;
            }
            memcpy(hostptr, data, size);

            // the device allocation and h2d copy will be captured in the graph
            devptr = wp_alloc_device(WP_CURRENT_CONTEXT, size);
            if (!devptr) {
                fprintf(
                    stderr, "Warp error: Failed to allocate %llu bytes on device 'cuda:%d' (in function %s)\n",
                    (unsigned long long)size, device_ordinal, __FUNCTION__
                );
                return false;
            }
            if (!check_cuda(cudaMemcpyAsync(devptr, hostptr, size, cudaMemcpyHostToDevice, stream)))
                return false;

            // graph takes ownership of the value storage
            FreeInfo free_info;
            free_info.context = NULL;
            free_info.ptr = hostptr;
            free_info.is_async = false;

            // allocation will be freed when graph is destroyed
            capture_info->tmp_allocs.push_back(free_info);
        }
    } else {
        // not capturing, copy the value to device memory
        devptr = wp_alloc_device(WP_CURRENT_CONTEXT, size);
        if (!devptr) {
            fprintf(
                stderr, "Warp error: Failed to allocate %llu bytes on device 'cuda:%d' (in function %s)\n",
                (unsigned long long)size, device_ordinal, __FUNCTION__
            );
            return false;
        }
        if (!check_cuda(cudaMemcpyAsync(devptr, data, size, cudaMemcpyHostToDevice, stream)))
            return false;
#if defined(__HIP_PLATFORM_AMD__)
        if (!hip_complete_pageable_copy(stream, data))
            return false;
#endif
    }

    *devptr_ret = devptr;
    *free_devptr_ret = free_devptr;

    return true;
}

// With WARP_HIP_END_TRACE=1, each step of ending a capture and of the deferred actions after it is printed as it is reached.
// It is off by default.
static bool end_trace_on() { static const bool on = getenv("WARP_HIP_END_TRACE") != NULL; return on; }
#define END_TRACE(what) do { if (end_trace_on()) { fprintf(stderr, "[hip-end] %s\n", what); fflush(stderr); } } while (0)

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
static void forget_standins(cudaGraph_t graph);  // Stand-ins for graph memory nodes, defined below before wp_alloc_device_async.
#endif

#if defined(__HIP_PLATFORM_AMD__)
// A stream-ordered free on the context's own stream, ordered after the work on the stream the memory was allocated on (HIP).
// CUDA's cudaFreeAsync(ptr, NULL) is ordered against every stream through the legacy null stream, but HIP's is not, so the free
// waits on an event recorded on the allocation's stream. The resource's context is current here (release_now). In 1.19 every
// deferred and immediate release goes through release_now, so this ordering only needs to live here.
static cudaError_t hip_free_async_ordered(void* ptr)
{
    ContextInfo* ci = get_context_info(static_cast<CUcontext>(NULL));
    CUstream free_stream = ci ? ci->stream : NULL;
    CUstream alloc_stream = NULL;
    {
        std::lock_guard<std::mutex> lock(g_alloc_streams_mutex);
        auto it = g_alloc_streams.find(ptr);
        if (it != g_alloc_streams.end()) {
            alloc_stream = it->second;
            g_alloc_streams.erase(it);
        }
    }
    if (alloc_stream != free_stream && alloc_stream != NULL) {
        StreamInfo* alloc_si = get_stream_info(alloc_stream);
        if (alloc_si && alloc_si->cached_event) {
            check_cu(cuEventRecord_f(alloc_si->cached_event, alloc_stream));
            check_cu(cuStreamWaitEvent_f(free_stream, alloc_si->cached_event, CU_EVENT_WAIT_DEFAULT));
        }
    }
    return cudaFreeAsync(ptr, free_stream);
}
#endif

#if defined(__HIP_PLATFORM_AMD__)
// Managed allocations made as mapped host memory (DeviceInfo::hip_managed_on_host), by start address and size. A pointer into
// one, such as an array view or a wrapped pointer, is then recognized as managed (wp_cuda_pointer_get_memory_kind), and each
// allocation is freed the way it was made.
static std::mutex g_hip_managed_mutex;
static std::map<uintptr_t, size_t> g_hip_managed;

static void hip_managed_remember(void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(g_hip_managed_mutex);
    g_hip_managed[(uintptr_t)ptr] = size;
}

// Returns true if ptr lies inside a managed allocation made as mapped host memory.
static bool hip_managed_contains(const void* ptr)
{
    std::lock_guard<std::mutex> lock(g_hip_managed_mutex);
    auto it = g_hip_managed.upper_bound((uintptr_t)ptr);
    if (it == g_hip_managed.begin())
        return false;
    --it;
    return (uintptr_t)ptr < it->first + it->second;
}

// Forgets the allocation that starts at ptr and returns true if there was one.
static bool hip_managed_forget(void* ptr)
{
    std::lock_guard<std::mutex> lock(g_hip_managed_mutex);
    return g_hip_managed.erase((uintptr_t)ptr) > 0;
}

// Stream-ordered allocations (cudaMallocAsync, from a memory pool), by start address and size. HIP answers no pointer query with the
// pool an allocation came from (CUDA's CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE is refused as not supported), so a pointer into one, such
// as a wrapped pointer, was reported as plain device memory; the port remembers them and reports them as pool memory
// (wp_cuda_pointer_get_memory_kind), as CUDA's driver does.
static std::mutex g_hip_pooled_mutex;
static std::map<uintptr_t, size_t> g_hip_pooled;

static void hip_pooled_remember(void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(g_hip_pooled_mutex);
    g_hip_pooled[(uintptr_t)ptr] = size;
}

static bool hip_pooled_contains(const void* ptr)
{
    std::lock_guard<std::mutex> lock(g_hip_pooled_mutex);
    auto it = g_hip_pooled.upper_bound((uintptr_t)ptr);
    if (it == g_hip_pooled.begin())
        return false;
    --it;
    return (uintptr_t)ptr < it->first + it->second;
}

static void hip_pooled_forget(void* ptr)
{
    std::lock_guard<std::mutex> lock(g_hip_pooled_mutex);
    g_hip_pooled.erase((uintptr_t)ptr);
}

// Keeps Warp's CPU arrays within reach of the GPU on an APU. On an integrated GPU, the GPU and the CPU cores share one memory, and
// a kernel can read and write memory from hipHostMalloc (mapped) at its own host address. Registered malloc memory does not work
// that way because it gets a different address, and HIP under Windows has no pageable memory access. One hipHostMalloc call costs
// about 10 ms even for 4 KiB, so Warp's CPU allocator takes its arrays from pinned arenas instead. Each arena is a 64 MiB chunk
// pinned once and split into blocks of power-of-two size classes from 256 B to 32 MiB, with a free list for each class, and an
// array larger than the largest class gets a pinned allocation of its own. The GPU can then reach every Warp CPU array at its own
// address, with no copies between CPU and GPU arrays, and the device reports pageable memory access. Memory that Warp did not
// allocate is refused by the launch check (hip_host_reachable). It is on by default on an integrated GPU and WARP_HIP_PINNED_CPU=0
// turns it off. Measured on the Radeon 8060S: a CPU kernel over 256 MiB ran in 33.5 ms on the arenas and 31.8 ms on ordinary memory,
// 2,000 small arrays allocated and freed in 8.9 ms against 10.5, and a 256 MiB copy to the GPU took 5.1 ms against 23.2. A pinned
// allocation that fails falls back to ordinary memory (wp_alloc_host), so a CPU array is never refused for it.
namespace {
struct HipHostArena {
    std::mutex mutex;
    bool decided = false;
    bool active = false;
    static constexpr size_t CHUNK = size_t(64) << 20;
    static constexpr int MIN_CLASS = 8;  // 256 B
    static constexpr int MAX_CLASS = 25;  // 32 MiB
    std::vector<void*> free_blocks[MAX_CLASS + 1];
    std::unordered_map<void*, int> owner;  // Size class of each block handed out, or -1 for a separate allocation.
    std::map<uintptr_t, size_t> ranges;  // Every pinned range (chunks and separate allocations), for interior-pointer lookups.
};
HipHostArena g_hip_arena;
}  // namespace

extern "C" int wp_hip_host_sanitized();  // warp.cpp

static bool hip_arena_decide()
{
    const char* env = getenv("WARP_HIP_PINNED_CPU");
    if (env && env[0] == '0')
        return false;
    // An AddressSanitizer build takes CPU arrays from ASan's heap, whose redzones catch an overflow past an array, as upstream's
    // sanitizer build does; WARP_HIP_PINNED_CPU=1 turns the arenas on there all the same.
    if (wp_hip_host_sanitized() && !(env && env[0] == '1'))
        return false;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
        ignore_cuda_error(hipGetLastError());
        return false;
    }
    int integrated = 0, can_map = 0;
    ignore_cuda_error(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, 0));
    ignore_cuda_error(hipDeviceGetAttribute(&can_map, hipDeviceAttributeCanMapHostMemory, 0));
    return integrated && can_map;
}

static bool hip_arena_active_locked()
{
    if (!g_hip_arena.decided) {
        g_hip_arena.active = hip_arena_decide();
        g_hip_arena.decided = true;
    }
    return g_hip_arena.active;
}

bool hip_host_arena_active()
{
    std::lock_guard<std::mutex> lock(g_hip_arena.mutex);
    return hip_arena_active_locked();
}

// Pinned memory is taken and given back with this thread's capture mode relaxed. A CPU array is made or freed inside a capture
// (warp.fem's temporaries do it in the FEM APIC example), and in a thread-local or global capture HIP refuses hipHostMalloc and
// hipHostFree there, as CUDA refuses cudaMallocHost, and the refusal invalidates the capture: its next launch reported error 901. A
// CPU array from malloc never touched the capture, and one from the arenas must not either.
static hipError_t hip_host_malloc_outside_capture(void** p, size_t s)
{
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    const hipError_t e = hipHostMalloc(p, s, hipHostMallocMapped);
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    return e;
}

static hipError_t hip_host_free_outside_capture(void* p)
{
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    const hipError_t e = hipHostFree(p);
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    return e;
}

// Allocates CPU memory from the arenas. Returns nullptr when the arenas are off or pinned memory is refused, and the caller then
// falls back to malloc.
void* hip_host_arena_alloc(size_t s)
{
    std::lock_guard<std::mutex> lock(g_hip_arena.mutex);
    if (!hip_arena_active_locked() || s == 0)
        return nullptr;
    int c = HipHostArena::MIN_CLASS;
    while (c <= HipHostArena::MAX_CLASS && (size_t(1) << c) < s)
        ++c;
    if (c > HipHostArena::MAX_CLASS) {
        void* p = nullptr;
        if (hip_host_malloc_outside_capture(&p, s) != hipSuccess || !p) {
            ignore_cuda_error(hipGetLastError());
            return nullptr;
        }
        g_hip_arena.owner[p] = -1;
        g_hip_arena.ranges[(uintptr_t)p] = s;
        return p;
    }
    auto& free_list = g_hip_arena.free_blocks[c];
    if (free_list.empty()) {
        void* chunk = nullptr;
        if (hip_host_malloc_outside_capture(&chunk, HipHostArena::CHUNK) != hipSuccess || !chunk) {
            ignore_cuda_error(hipGetLastError());
            return nullptr;
        }
        g_hip_arena.ranges[(uintptr_t)chunk] = HipHostArena::CHUNK;
        const size_t block = size_t(1) << c;
        free_list.reserve(free_list.size() + HipHostArena::CHUNK / block);
        for (size_t off = HipHostArena::CHUNK; off >= block; off -= block)
            free_list.push_back((char*)chunk + off - block);  // The lowest block is handed out first.
    }
    void* p = free_list.back();
    free_list.pop_back();
    g_hip_arena.owner[p] = c;
    return p;
}

// Takes ptr back and returns true if it came from the arenas. Returns false for anything else, which the caller frees the way it
// was allocated.
bool hip_host_arena_free(void* ptr)
{
    if (!ptr)
        return false;
    std::lock_guard<std::mutex> lock(g_hip_arena.mutex);
    auto it = g_hip_arena.owner.find(ptr);
    if (it == g_hip_arena.owner.end())
        return false;
    const int c = it->second;
    g_hip_arena.owner.erase(it);
    if (c < 0) {
        g_hip_arena.ranges.erase((uintptr_t)ptr);
        ignore_cuda_error(hip_host_free_outside_capture(ptr));
    } else {
        g_hip_arena.free_blocks[c].push_back(ptr);
    }
    return true;
}

// Returns true if ptr lies in the arenas' memory, which the GPU reaches at its own address. A pinned Warp array is already
// accepted before this is asked, by the launch check in context.py (`value.pinned and device.is_uva`).
int wp_hip_host_reachable(const void* ptr)
{
    if (!ptr)
        return 0;
    std::lock_guard<std::mutex> lock(g_hip_arena.mutex);
    auto it = g_hip_arena.ranges.upper_bound((uintptr_t)ptr);
    if (it == g_hip_arena.ranges.begin())
        return 0;
    --it;
    return (uintptr_t)ptr < it->first + it->second ? 1 : 0;
}
#endif

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
// Sharing GPU memory between processes on Windows. CUDA's IPC (cudaIpcGetMemHandle on a cudaMalloc allocation) only exists on
// Linux, and hipIpcOpenMemHandle hangs in a second process here. Windows shares GPU memory between processes through NT handles,
// and HIP can use them through a memory pool created with hipMemHandleTypeWin32. The pool's handle is duplicated into the other
// process and imported there, and then a pointer from the pool is exported and imported. Tested between two processes on the
// 8060S, each wrote 4 MiB and the other read it back exactly, while the VMM route returns error 801. So on Windows, Warp's default
// allocator, which is the one IPC uses (wp.ScopedMempool(device, False)), takes its memory from one such pool per device, outside
// any graph capture. Warp's 64-byte IPC handle names a small record in the exporting process (`Local\WarpHipIpcMem-<pid>-<n>`, a
// file mapping kept until the memory is freed) that holds the pool's handle in that process and the pointer's export data. The
// importer reads the record, duplicates the pool's handle out of the exporting process (same user, PROCESS_DUP_HANDLE), and
// imports the pool once and then the pointer. Events shared between processes need nothing extra, because HIP's own IPC events
// work here.
namespace {
struct HipIpcPool {
    bool tried = false;
    hipMemPool_t pool = nullptr;
    HANDLE nt = nullptr;  // The pool's NT handle in this process.
};
struct HipIpcRecord {
    char magic[8];
    unsigned pid;
    int device;
    unsigned long long pool_handle;  // The pool's handle value in the exporting process.
    hipMemPoolPtrExportData data;
};
struct HipIpcName {  // Warp's 64-byte IPC handle.
    char magic[4];
    unsigned version;
    unsigned pid;
    int device;
    unsigned long long seq;
    char unused[40];
};
static_assert(sizeof(HipIpcName) == 64, "Warp's IPC handle is 64 bytes");

std::mutex g_hip_ipc_mutex;
std::unordered_map<int, HipIpcPool> g_hip_ipc_pools;          // By device ordinal.
std::unordered_map<void*, int> g_hip_ipc_allocs;              // Default-allocator memory taken from a pool, with its device.
std::unordered_map<void*, std::vector<HANDLE>> g_hip_ipc_exports;  // Named records of an exported pointer, kept while it lives.
std::map<std::pair<unsigned, unsigned long long>, hipMemPool_t> g_hip_ipc_imported_pools;  // Keyed by (pid, pool handle in that process).
std::unordered_set<void*> g_hip_ipc_imported;                 // Pointers opened from another process.
unsigned long long g_hip_ipc_seq = 0;

void hip_ipc_record_name(char* out, size_t n, unsigned pid, unsigned long long seq)
{
    snprintf(out, n, "Local\\WarpHipIpcMem-%u-%llu", pid, seq);
}

// Returns the device's shareable pool, creating it the first time (call with g_hip_ipc_mutex held). Returns nullptr when the
// driver cannot create one.
HipIpcPool* hip_ipc_pool_locked(int device)
{
    HipIpcPool& p = g_hip_ipc_pools[device];
    if (p.tried)
        return p.pool ? &p : nullptr;
    p.tried = true;
    hipMemPoolProps props = {};
    props.allocType = hipMemAllocationTypePinned;
    props.handleTypes = hipMemHandleTypeWin32;
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = device;
    hipMemPool_t pool = nullptr;
    if (hipMemPoolCreate(&pool, &props) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        return nullptr;
    }
    HANDLE nt = nullptr;
    if (hipMemPoolExportToShareableHandle(&nt, pool, hipMemHandleTypeWin32, 0) != hipSuccess || !nt) {
        ignore_cuda_error(hipGetLastError());
        ignore_cuda_error(hipMemPoolDestroy(pool));
        return nullptr;
    }
    // Keep freed memory in the pool, as the device's default pool does for Warp.
    uint64_t keep = UINT64_MAX;
    ignore_cuda_error(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &keep));
    p.pool = pool;
    p.nt = nt;
    return &p;
}
}  // namespace

// Returns true if the device can share its default-allocator memory with another process. This is decided when Warp starts.
static bool hip_ipc_available(int device)
{
    std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
    return hip_ipc_pool_locked(device) != nullptr;
}

// Allocates default-allocator memory from the device's shareable pool, or returns nullptr so the caller allocates the usual way.
static void* hip_ipc_alloc(int device, size_t size)
{
    if (!g_captures.empty() || size == 0)
        return nullptr;  // The allocation's stream synchronization must never run into a capture.
    hipMemPool_t pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
        HipIpcPool* p = hip_ipc_pool_locked(device);
        if (!p)
            return nullptr;
        pool = p->pool;
    }
    // Allocate on a separate stream and synchronize it, so the memory is ready for every stream when this returns, like cudaMalloc.
    static thread_local hipStream_t s = nullptr;
    if (!s && hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        s = nullptr;
        return nullptr;
    }
    void* ptr = nullptr;
    if (hipMallocFromPoolAsync(&ptr, size, pool, s) != hipSuccess || hipStreamSynchronize(s) != hipSuccess || !ptr) {
        ignore_cuda_error(hipGetLastError());
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
    g_hip_ipc_allocs[ptr] = device;
    return ptr;
}

// Closes the named records of memory that is being freed. Returns true if the memory came from another process, in which case
// hipFree releases the import.
static void hip_ipc_forget(void* ptr)
{
    std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
    g_hip_ipc_allocs.erase(ptr);
    g_hip_ipc_imported.erase(ptr);
    auto it = g_hip_ipc_exports.find(ptr);
    if (it != g_hip_ipc_exports.end()) {
        for (HANDLE h : it->second)
            CloseHandle(h);
        g_hip_ipc_exports.erase(it);
    }
}

// Writes Warp's 64-byte handle for default-allocator memory. Returns false when the memory is not shareable.
static bool hip_ipc_export(void* ptr, char* out)
{
    memset(out, 0, 64);
    int device = -1;
    {
        std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
        auto it = g_hip_ipc_allocs.find(ptr);
        if (it == g_hip_ipc_allocs.end()) {
            wp::set_error_string(
                "Warp HIP error: this memory cannot be shared with another process (only the default allocator's memory, allocated "
                "outside a graph capture, is shareable on Windows)"
            );
            return false;
        }
        device = it->second;
    }
    HipIpcRecord rec = {};
    memcpy(rec.magic, "WHIPMEM1", 8);
    rec.pid = unsigned(GetCurrentProcessId());
    rec.device = device;
    if (hipMemPoolExportPointer(&rec.data, ptr) != hipSuccess) {
        ignore_cuda_error(hipGetLastError());
        wp::set_error_string("Warp HIP error: the memory pool would not export this pointer");
        return false;
    }
    std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
    HipIpcPool* p = hip_ipc_pool_locked(device);
    if (!p)
        return false;
    rec.pool_handle = (unsigned long long)(uintptr_t)p->nt;
    const unsigned long long seq = ++g_hip_ipc_seq;
    char name[96];
    hip_ipc_record_name(name, sizeof(name), rec.pid, seq);
    HANDLE map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, DWORD(sizeof(rec)), name);
    if (!map) {
        wp::set_error_string("Warp HIP error: the IPC record could not be made (CreateFileMapping %lu)", GetLastError());
        return false;
    }
    void* view = MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, sizeof(rec));
    if (!view) {
        CloseHandle(map);
        wp::set_error_string("Warp HIP error: the IPC record could not be written (MapViewOfFile %lu)", GetLastError());
        return false;
    }
    memcpy(view, &rec, sizeof(rec));
    UnmapViewOfFile(view);
    g_hip_ipc_exports[ptr].push_back(map);
    g_hip_shared_work.store(true, std::memory_order_relaxed);  // From now on, every command is submitted as it is issued.
    HipIpcName h = {};
    memcpy(h.magic, "WHIP", 4);
    h.version = 1;
    h.pid = rec.pid;
    h.device = device;
    h.seq = seq;
    memcpy(out, &h, sizeof(h));
    return true;
}

static bool hip_ipc_is_ours(const char* handle) { return memcmp(handle, "WHIP", 4) == 0; }

// Opens memory exported by another process. Returns nullptr on failure, with the reason in the error string.
static void* hip_ipc_import(const char* handle)
{
    HipIpcName h;
    memcpy(&h, handle, sizeof(h));
    char name[96];
    hip_ipc_record_name(name, sizeof(name), h.pid, h.seq);
    HANDLE map = OpenFileMappingA(FILE_MAP_READ, FALSE, name);
    if (!map) {
        wp::set_error_string("Warp HIP error: the IPC handle names no record (its process ended, or freed the memory)");
        return nullptr;
    }
    HipIpcRecord rec = {};
    const void* view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, sizeof(rec));
    if (view) {
        memcpy(&rec, view, sizeof(rec));
        UnmapViewOfFile(view);
    }
    CloseHandle(map);
    if (!view || memcmp(rec.magic, "WHIPMEM1", 8) != 0) {
        wp::set_error_string("Warp HIP error: the IPC record could not be read");
        return nullptr;
    }
    hipMemPool_t pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
        const auto key = std::make_pair(rec.pid, rec.pool_handle);
        auto it = g_hip_ipc_imported_pools.find(key);
        if (it != g_hip_ipc_imported_pools.end()) {
            pool = it->second;
        } else {
            HANDLE source = OpenProcess(PROCESS_DUP_HANDLE, FALSE, rec.pid);
            HANDLE local = nullptr;
            const BOOL dup = source
                && DuplicateHandle(source, (HANDLE)(uintptr_t)rec.pool_handle, GetCurrentProcess(), &local, 0, FALSE,
                                   DUPLICATE_SAME_ACCESS);
            if (source)
                CloseHandle(source);
            if (!dup || !local) {
                wp::set_error_string("Warp HIP error: the exporting process's memory pool could not be reached (%lu)", GetLastError());
                return nullptr;
            }
            if (hipMemPoolImportFromShareableHandle(&pool, (void*)local, hipMemHandleTypeWin32, 0) != hipSuccess || !pool) {
                ignore_cuda_error(hipGetLastError());
                CloseHandle(local);
                wp::set_error_string("Warp HIP error: the exporting process's memory pool could not be imported");
                return nullptr;
            }
            g_hip_ipc_imported_pools[key] = pool;  // The duplicated handle stays open along with the pool.
        }
    }
    void* ptr = nullptr;
    if (hipMemPoolImportPointer(&ptr, pool, &rec.data) != hipSuccess || !ptr) {
        ignore_cuda_error(hipGetLastError());
        wp::set_error_string("Warp HIP error: the exported pointer could not be imported");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
    g_hip_ipc_imported.insert(ptr);
    g_hip_shared_work.store(true, std::memory_order_relaxed);
    return ptr;
}
#endif

// The resource's context, if any, must be current.
static bool release_now(ReleaseKind kind, uint64_t handle)
{
    switch (kind) {
    case ReleaseKind::DeviceFree:
#if defined(__HIP_PLATFORM_AMD__)
        {
            // Stream-ordered memory is freed the default way, and so is memory allocated outside stream order during a capture
            // (is_static).
            std::lock_guard<std::mutex> lock(g_alloc_streams_mutex);
            g_alloc_streams.erase((void*)handle);
        }
        // Managed memory made as mapped host memory is freed the way it was allocated.
        if (hip_managed_forget((void*)handle))
            return check_cuda(hipHostFree((void*)handle));
#if defined(_WIN32)
        // Shareable memory has its IPC records closed. Memory opened from another process is also released by hipFree.
        hip_ipc_forget((void*)handle);
#endif
#endif
        return check_cuda(cudaFree((void*)handle));
    case ReleaseKind::DeviceFreeAsync:
        // postponed until a synchronization point, so preceding work on the pointer completes
#if defined(__HIP_PLATFORM_AMD__)
        return check_cuda(hip_free_async_ordered((void*)handle));
#else
        return check_cuda(cudaFreeAsync((void*)handle, NULL));
#endif
    case ReleaseKind::DeviceFreeAsyncUnmapped: {
#if defined(__HIP_PLATFORM_AMD__)
        cudaError_t res = hip_free_async_ordered((void*)handle);
#else
        cudaError_t res = cudaFreeAsync((void*)handle, NULL);
#endif
        if (res == cudaErrorInvalidValue) {
            // The graph was never launched, so the memory isn't mapped. Clear the error.
            ignore_cuda_error(cudaGetLastError());
            return true;
        }
        return check_cuda(res);
    }
    case ReleaseKind::PinnedFree:
        // errors were never reported here; keep it that way
        ignore_cuda_error(cudaFreeHost((void*)handle));
        return true;
    case ReleaseKind::Module:
#if defined(__HIP_PLATFORM_AMD__)
        hip_forget_module_vgprs((CUmodule)handle);
#endif
        return check_cu(cuModuleUnload_f((CUmodule)handle));
    case ReleaseKind::Graph:
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
        forget_standins((cudaGraph_t)handle);
#endif
        return check_cuda(cudaGraphDestroy((cudaGraph_t)handle));
    case ReleaseKind::GraphExec:
        return check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)handle));
    case ReleaseKind::TextureObject:
        return check_cu(cuTexObjectDestroy_f((CUtexObject)handle));
    case ReleaseKind::SurfaceObject:
        return check_cuda(cudaDestroySurfaceObject((cudaSurfaceObject_t)handle));
    case ReleaseKind::Array:
        return check_cu(cuArrayDestroy_f((CUarray)handle));
    case ReleaseKind::MipmappedArray:
        return check_cu(cuMipmappedArrayDestroy_f((CUmipmappedArray)handle));
    }
    return false;
}

// Queue unconditionally, e.g. while tearing down a failed capture.
static void defer_release(ReleaseKind kind, void* context, uint64_t handle)
{
    if (kind == ReleaseKind::PinnedFree || kind == ReleaseKind::Graph)
        context = NULL;
    else if (!context)
        context = get_current_context();

    std::lock_guard<std::mutex> lock(g_deferred_releases_mutex);
    g_deferred_releases.push_back({ kind, context, handle });
}

// Returns true if the context's current stream is recording a capture that Warp was not told about, such as a framework's own
// capture like PyTorch's torch.cuda.graph. A release must not run then either, because its event wait between streams ended
// test_torch's unregistered capture.
static bool hip_external_capture_on(void* context)
{
#if defined(__HIP_PLATFORM_AMD__)
    return g_captures.empty() && wp_cuda_stream_is_capturing(get_current_stream(context));
#else
    (void)context;
    return false;
#endif
}

bool release_or_defer(ReleaseKind kind, void* context, uint64_t handle)
{
    if (!handle)
        return true;

    // Conservative: the release may only be unsafe for some captures, depending on device and mode.
    if (!g_captures.empty() || hip_external_capture_on(context)) {
        defer_release(kind, context, handle);
        return true;
    }

    ContextGuard guard(context);
    return release_now(kind, handle);
}

static int release_deferred(void* context = NULL)
{
    if (!g_captures.empty() || hip_external_capture_on(context))
        return 0;

    // release outside the lock, since frees can block on device synchronization
    std::vector<ReleaseInfo> pending;
    {
        std::lock_guard<std::mutex> lock(g_deferred_releases_mutex);
        std::vector<ReleaseInfo> kept;
        for (const ReleaseInfo& info : g_deferred_releases) {
            if (!context || !info.context || info.context == context)
                pending.push_back(info);
            else
                kept.push_back(info);
        }
        g_deferred_releases.swap(kept);
    }

    for (const ReleaseInfo& info : pending) {
        // restore, since deferred work must not change the current device
        ContextGuard guard(info.context, true);
        release_now(info.kind, info.handle);
    }

    return int(pending.size());
}

// The driver releases a context's resources with it, except stream-ordered allocations,
// which belong to the device's memory pool, so free those from another context of the device.
static void discard_deferred_releases(void* context, void* device_context)
{
    std::lock_guard<std::mutex> lock(g_deferred_releases_mutex);
    std::vector<ReleaseInfo> kept;
    for (ReleaseInfo info : g_deferred_releases) {
        if (info.context == context) {
            bool stream_ordered
                = info.kind == ReleaseKind::DeviceFreeAsync || info.kind == ReleaseKind::DeviceFreeAsyncUnmapped;
#if defined(__HIP_PLATFORM_AMD__)
            // A second context on a device is a virtual one (see cuda_util.cpp) and holds nothing of its own on the GPU. Everything
            // it allocated, loaded, or created belongs to the device. CUDA frees a destroyed context's resources along with it, so
            // Warp drops their pending releases, but here nothing else would free them, so they all move to the device's context.
            const bool carried = device_context != NULL && context != device_context;
            if (!carried && (!stream_ordered || !device_context))
                continue;
#else
            if (!stream_ordered || !device_context)
                continue;
#endif
            info.context = device_context;
        }
        kept.push_back(info);
    }
    g_deferred_releases.swap(kept);
}

static int process_deferred_graph_destroy_callbacks(void* context = NULL)
{
    int num_freed = 0;

    // Like the deferred frees and graph destroys above, a destroyed graph's memory waits until no capture is in progress. Its
    // temporaries allocated outside stream order (HIP on Windows, is_static in wp_alloc_device_async) are freed with a synchronous
    // cudaFree, which never returns inside another stream's capture. A graph destroyed without a synchronize, whose callback arrived
    // while the next capture was open, hung a test at its third capture.
    if (!g_captures.empty())
        return 0;

    // The destroyed graphs are taken off the list under its lock, and their memory is freed after the lock is released. Freeing can
    // make the runtime run another destroyed graph's callback (on_graph_destroy) on this same thread, which HIP on Windows does, and
    // that callback takes this lock. When the lock was held across the frees, the thread waited on itself forever. A callback that
    // arrives in the meantime is handled on the next pass.
    std::vector<GraphDestroyCallbackInfo*> todo;
    {
        std::lock_guard<std::mutex> lock(g_graph_destroy_mutex);
        for (auto it = g_deferred_graph_destroy_list.begin(); it != g_deferred_graph_destroy_list.end(); /*noop*/) {
            if ((*it)->context == context || !context) {
                todo.push_back(*it);
                it = g_deferred_graph_destroy_list.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (GraphDestroyCallbackInfo* graph_info : todo) {
        {
            // handle unfreed graph allocations (may have outstanding user references)
            for (void* ptr : graph_info->unfreed_allocs) {
                auto alloc_iter = g_graph_allocs.find(ptr);
                if (alloc_iter != g_graph_allocs.end()) {
                    // unlink this allocation from the destroyed graph
                    // and free it if no user reference remains
                    GraphAllocInfo& alloc_info = alloc_iter->second;
                    alloc_info.graph_destroyed = true;
                    if (!alloc_info.ref_exists) {
                        // Multi-GPU graphs own allocations of other contexts than their own.
                        // Restore, since deferred work must not change the current device.
                        ContextGuard guard(alloc_info.context, true);
                        wp_free_device_async(alloc_info.context, ptr);
                    }
                }
            }

            // handle temporary allocations owned by the graph (no user references)
            for (const FreeInfo& tmp_info : graph_info->tmp_allocs) {
                if (tmp_info.context) {
                    // GPU alloc; restore the context as above
                    ContextGuard guard(tmp_info.context, true);
                    if (tmp_info.is_async) {
                        wp_free_device_async(tmp_info.context, tmp_info.ptr);
                    } else {
                        wp_free_device_default(tmp_info.context, tmp_info.ptr);
                    }
                } else {
                    // CPU alloc
                    wp_free_host(tmp_info.ptr);
                }
            }

            ++num_freed;
            delete graph_info;
        }
    }

    return num_freed;
}

static int run_deferred_actions(void* context = NULL)
{
    int num_actions = 0;
    END_TRACE("deferred: releases");
    num_actions += release_deferred(context);
    END_TRACE("deferred: destroy callbacks");
    num_actions += process_deferred_graph_destroy_callbacks(context);
    END_TRACE("deferred: done");
    return num_actions;
}

// Callback used when a graph is destroyed.
// NOTE: this runs on an internal CUDA thread and requires synchronization.
static void CUDART_CB on_graph_destroy(void* user_data)
{
    if (user_data) {
        std::lock_guard<std::mutex> lock(g_graph_destroy_mutex);
        g_deferred_graph_destroy_list.push_back(static_cast<GraphDestroyCallbackInfo*>(user_data));
    }
}

static inline const char* get_cuda_kernel_name(void* kernel)
{
    CUfunction cuda_func = static_cast<CUfunction>(kernel);
    auto name_iter = g_kernel_names.find((CUfunction)cuda_func);
    if (name_iter != g_kernel_names.end())
        return name_iter->second.c_str();
    else
        return "unknown_kernel";
}

template <typename HaystackIter, typename NeedleIter>
static bool
contains_any(HaystackIter haystack_begin, HaystackIter haystack_end, NeedleIter needle_begin, NeedleIter needle_end)
{
    for (auto it = needle_begin; it != needle_end; ++it) {
        if (std::find(haystack_begin, haystack_end, *it) != haystack_end) {
            return true;
        }
    }
    return false;
}


void* wp_alloc_pinned(size_t s, const char* tag)
{
    void* ptr = NULL;
    check_cuda(cudaMallocHost(&ptr, s));
    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_PINNED, -1, tag);
    return ptr;
}

void wp_free_pinned(void* ptr)
{
    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_free(ptr);
    release_or_defer(ReleaseKind::PinnedFree, NULL, (uint64_t)ptr);
}

void* wp_alloc_device(void* context, size_t s, const char* tag)
{
    int ordinal = wp_cuda_context_get_device_ordinal(context);

    if (wp_cuda_device_is_mempool_supported(ordinal))
        return wp_alloc_device_async(context, s, WP_CURRENT_STREAM, tag);
    else
        return wp_alloc_device_default(context, s, tag);
}

void wp_free_device(void* context, void* ptr)
{
    int ordinal = wp_cuda_context_get_device_ordinal(context);

    // use stream-ordered allocator if available
    if (wp_cuda_device_is_mempool_supported(ordinal))
        wp_free_device_async(context, ptr);
    else
        wp_free_device_default(context, ptr);
}

void* wp_alloc_device_default(void* context, size_t s, const char* tag)
{
    ContextGuard guard(context);

    void* ptr = NULL;
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    // Use shareable memory when the driver offers it (hip_ipc_alloc), and the usual allocation otherwise. Never use it while this
    // context's stream is recording a capture Warp was not told about, such as a framework's, because the pool's stream
    // synchronization must not run into that capture.
    if (!wp_cuda_stream_is_capturing(get_current_stream(context))) {
        // With no context given and none current (CUDA's empty context stack), use the current device, which is where cudaMalloc allocates.
        int ordinal = wp_cuda_context_get_device_ordinal(context ? context : get_current_context());
        if (ordinal < 0)
            ignore_cuda_error(hipGetDevice(&ordinal));
        if (ordinal >= 0)
            ptr = hip_ipc_alloc(ordinal, s);
    }
    if (!ptr)
#endif
    check_cuda(cudaMalloc(&ptr, s));

    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);
    return ptr;
}

void wp_free_device_default(void* context, void* ptr)
{
    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_free(ptr);

    release_or_defer(ReleaseKind::DeviceFree, context, (uint64_t)ptr);
}

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
// Emulated graph memory nodes (HIP on Windows). A capture there cannot hold an alloc or free node (see wp_alloc_device_async
// below), so each one is replaced by an empty node placed exactly where it would go. An alloc goes at its stream's frontier, and a
// free goes after every leaf that descends from its alloc, with every stream of the capture that had reached those leaves
// continuing from it. graph_standin_allocs and graph_standin_frees record which pointer each stand-in node represents, so the
// graph's order of use and release reads the same way as CUDA's (graph_alloc_query, wp_cuda_graph_insert_free_node). The memory
// itself belongs to the graph until the graph is destroyed.

// Adds an empty node at a capturing stream's frontier, and the stream continues from it. Returns NULL when the stream is not
// capturing.
static cudaGraphNode_t standin_at_frontier(CUstream stream)
{
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t graph = NULL;
    const cudaGraphNode_t* deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(stream, &status, nullptr, &graph, &deps, &dep_count)) || !graph
        || status != CU_STREAM_CAPTURE_STATUS_ACTIVE)
        return NULL;
    cudaGraphNode_t node = NULL;
    if (!check_cuda(cudaGraphAddEmptyNode(&node, graph, deps, dep_count)))
        return NULL;
    if (!check_cu(cuStreamUpdateCaptureDependencies_f(stream, &node, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES)))
        return NULL;
    return node;
}

// Stands in for a free. An empty node is placed after every leaf that descends from the alloc's stand-in, and each stream of the
// same capture whose frontier held one of those leaves continues from it, keeping its other dependencies. This mirrors how a
// MemFreeNode is placed for CUDA below.
static cudaGraphNode_t standin_free(cudaGraph_t graph, cudaGraphNode_t alloc_node, void* ptr)
{
    std::vector<cudaGraphNode_t> leaves;
    if (!graph || !alloc_node || !get_dependent_leaf_nodes(alloc_node, leaves))
        return NULL;
    cudaGraphNode_t node = NULL;
    if (!check_cuda(cudaGraphAddEmptyNode(&node, graph, leaves.data(), leaves.size())))
        return NULL;
    graph_standin_frees()[static_cast<void*>(node)] = ptr;
    for (const auto& kv : g_streams) {
        cudaStream_t other = kv.first;
        CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
        cudaGraph_t other_graph = NULL;
        const cudaGraphNode_t* deps = NULL;
        size_t dep_count = 0;
        if (!check_cu(cuStreamGetCaptureInfo_f(other, &status, nullptr, &other_graph, &deps, &dep_count)))
            continue;
        if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE || other_graph != graph
            || !contains_any(deps, deps + dep_count, leaves.begin(), leaves.end()))
            continue;
        std::vector<cudaGraphNode_t> next { node };
        for (size_t i = 0; i < dep_count; i++)
            if (std::find(leaves.begin(), leaves.end(), deps[i]) == leaves.end())
                next.push_back(deps[i]);
        check_cu(cuStreamUpdateCaptureDependencies_f(other, next.data(), next.size(), CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    }
    return node;
}

// Forgets a destroyed graph's stand-ins, because a node handle can be reused for a node of another graph.
static void forget_standins(cudaGraph_t graph)
{
    if (!graph || (graph_standin_allocs().empty() && graph_standin_frees().empty()))
        return;
    size_t n = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, NULL, &n)) || !n)
        return;
    std::vector<cudaGraphNode_t> nodes(n);
    if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &n)))
        return;
    for (cudaGraphNode_t node : nodes) {
        graph_standin_allocs().erase(static_cast<void*>(node));
        graph_standin_frees().erase(static_cast<void*>(node));
    }
}
#endif

void* wp_alloc_device_async(void* context, size_t s, void* stream_, const char* tag)
{
    // stream-ordered allocations don't rely on the current context,
    // but we set the context here for consistent behaviour
    ContextGuard guard(context);

    CUstream stream;
    if (stream_ != WP_CURRENT_STREAM)
        stream = static_cast<CUstream>(stream_);
    else
        stream = get_current_stream(context);

    void* ptr = NULL;

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    // HIP on Windows does not support allocation in a capture. ROCm's documentation says its stream-ordered allocator "is
    // implemented on Linux and is under development on Microsoft Windows", and hipMallocAsync in a capturing stream fails there
    // with "operation not permitted when stream is capturing". So the graph cannot hold an allocation node, and every capture that
    // allocates (Warp's deterministic reductions, array scans, and temporaries) failed. Here the memory is allocated outside stream
    // order with hipMalloc, with the calling thread's capture mode relaxed for that one call, the same way Warp's own pause and
    // resume work. The memory is owned like a graph allocation. It is kept while the graph or its user holds it and freed when
    // neither does (wp_free_device_async, is_static). Every replay of the graph uses the same memory, which is safe because a
    // graph's launches are serialized on their stream.
    static const bool alloc_debug = getenv("WARP_HIP_ALLOC_DEBUG") != NULL;
    if (alloc_debug) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        ignore_cuda_error(cudaStreamIsCapturing(stream, &st));
        fprintf(stderr, "[hip-alloc] %zu B on stream %p: capturing %d, id %llu, captures %zu:", s, (void*)stream, int(st),
                (unsigned long long)(st != cudaStreamCaptureStatusNone ? get_capture_id(stream) : 0), g_captures.size());
        for (const auto& kv : g_captures)
            fprintf(stderr, " [id %llu stream %p now %llu]", (unsigned long long)kv.first, (void*)kv.second->stream,
                    (unsigned long long)get_capture_id(kv.second->stream));
        // The context's current stream, which belongs to a foreign capture when one is running, and its status (1 active, 2 invalidated).
        cudaStreamCaptureStatus cur = cudaStreamCaptureStatusNone;
        CUstream current = get_current_stream(context);
        ignore_cuda_error(cudaStreamIsCapturing(current, &cur));
        fprintf(stderr, " | current stream %p status %d, foreign capture beside %d\n", (void*)current, int(cur),
                int(hip_external_capture_on(context)));
    }
    {
        // Find the Warp capture this stream records into by the stream itself. On HIP, a capture resumed after a pause is a new
        // capture sequence with a new id (BeginCaptureToGraph), while Warp keeps it under its first id (capture->id), which is the id
        // its end and its graph's allocations are counted by. Looking it up by the stream's current id alone missed it.
        CaptureInfo* capture = NULL;
        // Handle a capture that Warp was not told about, such as PyTorch's torch.cuda.graph around a Warp radix sort, either on this
        // stream or on the context's current stream while this one is a side stream. The sort's scratch memory is allocated on one of
        // them with its capture mode relaxed, and HIP still refuses that stream-ordered allocation with error 900, which ended
        // PyTorch's capture. The memory is allocated the same way, outside stream order, with an empty node where its alloc node would
        // go on a capturing stream. It belongs to a graph that Warp never sees end, so the memory stays for that graph's replays.
        // wp_free_device_async keeps it, because its capture is unknown and its graph is never reported destroyed.
        const bool foreign_here = wp_cuda_stream_is_capturing(stream) && !find_capture_info(get_capture_id(stream));
        const bool foreign_beside = !foreign_here && !wp_cuda_stream_is_capturing(stream) && hip_external_capture_on(context);
        if (wp_cuda_stream_is_capturing(stream) || foreign_beside) {
            capture = foreign_beside ? NULL : find_capture_info(get_capture_id(stream));
            if (!capture) {
                cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
                check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
                check_cuda(cudaMalloc(&ptr, s));
                check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
                if (alloc_debug) {
                    cudaStreamCaptureStatus cur = cudaStreamCaptureStatusNone;
                    ignore_cuda_error(cudaStreamIsCapturing(get_current_stream(context), &cur));
                    fprintf(stderr, "[hip-alloc]   made outside the stream order for a foreign capture (%s): %p; current stream status %d\n",
                            foreign_beside ? "beside it" : "on it", ptr, int(cur));
                }
                if (ptr) {
                    GraphAllocInfo alloc_info;
                    alloc_info.capture_id = foreign_beside ? get_capture_id(get_current_stream(context)) : get_capture_id(stream);
                    alloc_info.context = context ? context : get_current_context();
                    alloc_info.ref_exists = true;
                    alloc_info.graph_destroyed = false;
                    alloc_info.is_static = true;
                    alloc_info.size = s;
                    alloc_info.node = foreign_beside ? NULL : standin_at_frontier(stream);
                    if (alloc_info.node)
                        graph_standin_allocs()[static_cast<void*>(alloc_info.node)] = ptr;
                    g_graph_allocs[ptr] = alloc_info;
                    if (g_alloc_tracker.enabled)
                        g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);
                }
                return ptr;
            }
        } else if (!g_captures.empty()) {
            // Handle a stream that is not capturing while a Warp capture on the same context runs on another stream. HIP refuses a
            // stream-ordered allocation then (error 900 in the capture's global mode), where CUDA allows it. A deterministic launch
            // asked for its counters on the context's own stream while the capture ran on the device's stream. The memory belongs to
            // the capture, as if it had been allocated in it, and is kept as part of its graph.
            CUcontext want = static_cast<CUcontext>(context ? context : get_current_context());
            for (const auto& kv : g_captures) {
                if (kv.second && kv.second->context == want) {
                    capture = kv.second;
                    break;
                }
            }
        }
        if (capture) {
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
            check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
            check_cuda(cudaMalloc(&ptr, s));
            check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
            if (ptr) {
                GraphAllocInfo alloc_info;
                alloc_info.capture_id = capture->id;
                alloc_info.context = context ? context : get_current_context();
                alloc_info.ref_exists = true;
                alloc_info.graph_destroyed = false;
                alloc_info.is_static = true;
                alloc_info.size = s;
                // On a capturing stream, an empty node goes where the alloc node would be and the stream continues from it, so
                // whatever uses this memory in the graph comes after it (standin_at_frontier).
                if (wp_cuda_stream_is_capturing(stream)) {
                    alloc_info.node = standin_at_frontier(stream);
                    if (alloc_info.node)
                        graph_standin_allocs()[static_cast<void*>(alloc_info.node)] = ptr;
                }
                g_graph_allocs[ptr] = alloc_info;
                if (g_alloc_tracker.enabled)
                    g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);
            }
            return ptr;
        }
    }
#endif

#if defined(__HIP_PLATFORM_AMD__)
    // A logical device allocates from a pool of its own on its card, held to its share of the card's memory, as a separate GPU holds
    // its own (hip_split_devices). Past its budget an allocation fails as an out-of-memory GPU's does.
    ContextInfo* alloc_context = get_context_info(context);
    DeviceInfo* d = alloc_context ? alloc_context->device_info : NULL;
    if (d && d->hip_parts > 1) {
        if (hip_share_pool(*d)) {
            uint64_t used = 0;
            if (hipMemPoolGetAttribute(d->hip_pool, hipMemPoolAttrUsedMemCurrent, &used) != hipSuccess)
                ignore_cuda_error(hipGetLastError());
            if (d->hip_memory_budget && used + s > d->hip_memory_budget) {
                wp::set_error_string("Warp CUDA error 2: out of memory (%s holds %zu bytes of its %zu)", d->name, size_t(used),
                                     d->hip_memory_budget);
                return NULL;
            }
            check_cuda(hipMallocFromPoolAsync(&ptr, s, d->hip_pool, static_cast<hipStream_t>(stream)));
        } else {
            check_cuda(cudaMallocAsync(&ptr, s, stream));
        }
    } else
#endif
        check_cuda(cudaMallocAsync(&ptr, s, stream));

    if (ptr) {
        // if the stream is capturing, the allocation requires special handling
        if (wp_cuda_stream_is_capturing(stream)) {
            // check if this is a known capture
            uint64_t capture_id = get_capture_id(stream);
            auto capture_iter = g_captures.find(capture_id);
            if (capture_iter != g_captures.end()) {
                // remember graph allocation details
                GraphAllocInfo alloc_info;
                alloc_info.capture_id = capture_id;
                alloc_info.context = context ? context : get_current_context();
                alloc_info.ref_exists = true;  // user reference created and returned here
                alloc_info.graph_destroyed = false;  // graph not destroyed yet
                alloc_info.size = s;

                // find the MemAllocNode that was just added
                std::vector<cudaGraphNode_t> deps;
                if (get_capture_dependencies(stream, deps)) {
                    for (cudaGraphNode_t node : deps) {
                        CUgraphNodeType node_type;
                        if (check_cu(cuGraphNodeGetType_f(node, &node_type))) {
                            if (node_type == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
                                cudaMemAllocNodeParams params;
                                if (check_cuda(cudaGraphMemAllocNodeGetParams(node, &params))) {
                                    if (params.dptr == ptr) {
                                        alloc_info.node = node;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                // Warn if the node is not found. This is unlikely and it's not a critical error,
                // but we must also handle this situation in wp_free_device_async().
                if (!alloc_info.node) {
                    fprintf(stderr, "Warp warning: %s: failed to find memory allocation node\n", __FUNCTION__);
                }

                g_graph_allocs[ptr] = alloc_info;
            } else {
                g_untracked_graph_allocs.insert(ptr);
            }
        }
#if defined(__HIP_PLATFORM_AMD__)
        else {
            // Track the allocation stream for cross-stream free safety
            {
                std::lock_guard<std::mutex> lock(g_alloc_streams_mutex);
                g_alloc_streams[ptr] = stream;
            }
            hip_pooled_remember(ptr, s);  // (pool memory, which no HIP query names: wp_cuda_pointer_get_memory_kind)
        }
#endif
    }

    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);

    return ptr;
}

void* wp_alloc_device_managed(void* context, size_t s, const char* tag)
{
    ContextGuard guard(context);

    ContextInfo* context_info = get_context_info(context);
    if (!context_info || !context_info->device_info)
        return NULL;

    DeviceInfo* device_info = context_info->device_info;
    if (!device_info->managed_memory)
        return NULL;

    void* ptr = NULL;

#if defined(__HIP_PLATFORM_AMD__)
    if (device_info->hip_managed_on_host) {
        // Coherent host memory mapped into the GPU. On an APU it is the same physical memory under one pointer (UVA).
        if (!check_cuda(hipHostMalloc(&ptr, s, hipHostMallocMapped | hipHostMallocCoherent)))
            return NULL;
        hip_managed_remember(ptr, s);
        if (g_alloc_tracker.enabled && ptr)
            g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);
        return ptr;
    }
#endif

    if (!check_cuda(cudaMallocManaged(&ptr, s, cudaMemAttachGlobal)))
        return NULL;

    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_alloc(ptr, s, ALLOC_KIND_DEVICE, wp_cuda_context_get_device_ordinal(context), tag);

    return ptr;
}

void wp_free_device_async(void* context, void* ptr, void** dbg_node_ret)
{
#if defined(__HIP_PLATFORM_AMD__)
    // AMD's driver frees whatever pointer it is given in stream order without checking it. On the Radeon 8060S under Windows, a
    // host address returns hipSuccess, while CUDA's driver returns "invalid argument" (test_capture_release,
    // test_invalid_free_during_capture_reports_error). So the pointer is checked here the way CUDA's driver checks it. Memory the GPU
    // allocated is freed, and anything else is reported as CUDA reports it and left alone. A graph allocation made by this build is
    // already known and never queried, because its address may not answer a pointer query before its graph runs.
    if (ptr && g_graph_allocs.find(ptr) == g_graph_allocs.end() && g_untracked_graph_allocs.count(ptr) == 0) {
        hipPointerAttribute_t attr = {};
        const hipError_t got = hipPointerGetAttributes(&attr, ptr);
        if (got != hipSuccess || (attr.type != hipMemoryTypeDevice && attr.type != hipMemoryTypeManaged)) {
            ignore_cuda_error(hipGetLastError());  // An unknown pointer leaves its error set, so clear it and report the real error below.
            if (dbg_node_ret)
                *dbg_node_ret = NULL;
            check_cu(CUDA_ERROR_INVALID_VALUE);
            return;
        }
    }
#endif
    if (g_alloc_tracker.enabled && ptr)
        g_alloc_tracker.record_free(ptr);
#if defined(__HIP_PLATFORM_AMD__)
    if (ptr)
        hip_pooled_forget(ptr);
#endif

    // stream-ordered allocators generally don't rely on the current context,
    // but we set the context here for consistent behaviour
    ContextGuard guard(context);

    // NB: Stream-ordered deallocations are tricky, because the memory could still be used on another stream
    // or even multiple streams.  To avoid use-after-free errors, we need to ensure that all preceding work
    // completes before releasing the memory.  The strategy is different for regular stream-ordered allocations
    // and allocations made during graph capture.  See below for details.

    // check if this allocation was made during graph capture
    auto alloc_iter = g_graph_allocs.find(ptr);
    if (alloc_iter == g_graph_allocs.end()) {
        ReleaseKind kind
            = g_untracked_graph_allocs.erase(ptr) ? ReleaseKind::DeviceFreeAsyncUnmapped : ReleaseKind::DeviceFreeAsync;
        release_or_defer(kind, context, (uint64_t)ptr);
    } else {
        // get the graph allocation details
        GraphAllocInfo& alloc_info = alloc_iter->second;
        uint64_t capture_id = alloc_info.capture_id;

        if (alloc_info.is_static) {
            // Allocated outside stream order during a capture (HIP on Windows, wp_alloc_device_async), so no free node can hold it.
            auto static_capture = g_captures.find(capture_id);
            if (static_capture != g_captures.end()) {
                // Freed while its capture, or a conditional body inside it, is in progress. The graph keeps it as one of its own
                // temporaries and frees it when the graph is destroyed. While its capture is still recording, an empty node goes where
                // its free node would be, after everything that used it (standin_free).
                cudaGraphNode_t free_node = NULL;
                if (alloc_info.node && capture_id == get_capture_id(static_capture->second->stream))
                    free_node = standin_free(get_capture_graph(static_capture->second->stream), alloc_info.node, ptr);
                FreeInfo free_info;
                free_info.ptr = ptr;
                free_info.context = alloc_info.context;
                free_info.is_async = false;
                static_capture->second->tmp_allocs.push_back(free_info);
                g_graph_allocs.erase(alloc_iter);
                if (dbg_node_ret)
                    *dbg_node_ret = free_node;
                return;
            } else if (alloc_info.graph_destroyed) {
                // Neither its graph nor its user holds it now, so free it the way it was allocated (cudaFree), or once no capture is
                // in progress.
                void* free_context = alloc_info.context;
                g_graph_allocs.erase(alloc_iter);
                release_or_defer(ReleaseKind::DeviceFree, free_context, (uint64_t)ptr);
            } else {
                // Its graph is still alive, so the memory is freed along with the graph.
                alloc_info.ref_exists = false;
            }
            if (dbg_node_ret)
                *dbg_node_ret = NULL;
            return;
        }

        // Check if the capture is still active on its stream. The second condition guards
        // against conditional body captures (wp.capture_if()/wp.capture_while()): while a
        // body graph is being captured, the owning capture is paused and its stream is
        // capturing the body graph under a different capture id. The allocation's node and
        // dependencies live in the paused parent graph, so a free node cannot be added
        // here; fall through and let the graph retain the allocation instead.
        // Note: the capture id is stable across pause/resume of the same graph, so this
        // path is taken again for frees that occur after the conditional completes.
        auto capture_iter = g_captures.find(capture_id);
        if (capture_iter != g_captures.end() && capture_id == get_capture_id(capture_iter->second->stream)) {
#if defined(__HIP_PLATFORM_AMD__)
            // hipGraphAddMemFreeNode rejects pointers from hipMallocAsync during
            // stream capture (hipErrorInvalidValue). It only accepts pointers from
            // hipGraphAddMemAllocNode on explicitly constructed graphs.
            // Use hipFreeAsync on the capturing stream instead, the stream capture
            // mechanism records it as a proper free node in the graph.
            {
                CaptureInfo* capture = capture_iter->second;
                check_cuda(cudaFreeAsync(ptr, capture->stream));
            }
            if (dbg_node_ret)
                *dbg_node_ret = NULL;
#else
            CaptureInfo* capture = capture_iter->second;
            cudaGraph_t graph = get_capture_graph(capture->stream);
            if (!graph) {
                fprintf(stderr, "Warp warning: %s: failed to get capture graph\n", __FUNCTION__);
                g_graph_allocs.erase(alloc_iter);
                return;
            }

            cudaGraphNode_t free_node = NULL;

            if (alloc_info.node) {
                // Find all leaf nodes that depend on the alloc node.
                std::vector<cudaGraphNode_t> alloc_leaf_nodes;
                if (!get_dependent_leaf_nodes(alloc_info.node, alloc_leaf_nodes)) {
                    fprintf(stderr, "Warp warning: %s: failed to get allocation-dependent nodes\n", __FUNCTION__);
                    g_graph_allocs.erase(alloc_iter);
                    return;
                }

                // Add a mem free node. All graph leaf nodes that are descendants of the alloc node
                // will be used as dependencies to ensure that all prior work completes before deallocating.
                // This works with both Warp-initiated and external captures and avoids the need to explicitly
                // track all streams used during the capture.
                if (!check_cuda(cudaGraphAddMemFreeNode(
                        &free_node, graph, alloc_leaf_nodes.data(), alloc_leaf_nodes.size(), ptr
                    ))) {
                    fprintf(stderr, "Warp warning: %s: failed to add a memory free node\n", __FUNCTION__);
                    g_graph_allocs.erase(alloc_iter);
                    return;
                }

                // Update the capture dependencies for affected child streams, if the streams are still alive.
                // Adding the MemFreeNode as a dependency allows the memory to be reused later,
                // leading to smaller graph memory footprints.
                // This creates an implicit synchronization point between all streams that
                // potentially depend on the allocation, but other streams are unaffected.
                // Implementation notes:
                // - Brute force search through all streams known to Warp, since we don't track
                //   which streams are part of a capture. FIXME?
                // - If a stream's frontier includes alloc-dependent nodes:
                //   - The free node already depends on these nodes
                //     (get_dependent_leaf_nodes() + cudaGraphAddMemFreeNode() above).
                //   - We replace all alloc-dependent deps with the new free node. Other deps remain unchanged.
                for (const auto& kv : g_streams) {
                    cudaStream_t other_stream = kv.first;
                    CUstreamCaptureStatus other_capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
                    uint64_t other_capture_id = 0;
                    const cudaGraphNode_t* other_capture_deps = NULL;
                    size_t other_dep_count = 0;
                    if (check_cu(cuStreamGetCaptureInfo_f(
                            other_stream, &other_capture_status, &other_capture_id, NULL, &other_capture_deps,
                            &other_dep_count
                        ))) {
                        // check if the other stream is part of the same capture
                        if (other_capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE && other_capture_id == capture_id) {
                            // check if the stream's frontier includes alloc-dependent nodes
                            if (contains_any(
                                    other_capture_deps, other_capture_deps + other_dep_count, alloc_leaf_nodes.begin(),
                                    alloc_leaf_nodes.end()
                                )) {
                                // Update the stream's capture deps. We replace all alloc-dependent deps with the new
                                // free node. Other deps remain unchanged.
                                std::vector<cudaGraphNode_t> new_deps { free_node };
                                for (size_t i = 0; i < other_dep_count; i++) {
                                    cudaGraphNode_t dep = other_capture_deps[i];
                                    if (std::find(alloc_leaf_nodes.begin(), alloc_leaf_nodes.end(), dep)
                                        == alloc_leaf_nodes.end()) {
                                        new_deps.push_back(dep);
                                    }
                                }
                                check_cu(cuStreamUpdateCaptureDependencies_f(
                                    other_stream, new_deps.data(), new_deps.size(), CU_STREAM_SET_CAPTURE_DEPENDENCIES
                                ));
                            }
                        }
                    }
                }
            } else {
                // Fallback if the allocation node was not found in wp_alloc_device_async().
                // Use all current leaf nodes as dependencies to ensure that all prior work completes before
                // deallocating. This produces correct graphs, but may introduce unnecessary stream serialization with
                // multi-stream captures.
                std::vector<cudaGraphNode_t> leaf_nodes;
                if (get_graph_leaf_nodes(graph, leaf_nodes)) {
                    if (check_cuda(
                            cudaGraphAddMemFreeNode(&free_node, graph, leaf_nodes.data(), leaf_nodes.size(), ptr)
                        )) {
                        check_cu(cuStreamUpdateCaptureDependencies_f(
                            capture->stream, &free_node, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES
                        ));
                    }
                }
            }

            // return the free node for testing/debugging
            if (dbg_node_ret)
                *dbg_node_ret = free_node;
#endif  // defined(__HIP_PLATFORM_AMD__)

            // we're done with this allocation, it's owned by the graph
            g_graph_allocs.erase(alloc_iter);
        } else {
            // The capture has ended, or it is paused while a conditional body graph is
            // being captured. If the owning graph was already destroyed, we can free the
            // pointer now.
            if (alloc_info.graph_destroyed) {
                release_or_defer(ReleaseKind::DeviceFreeAsyncUnmapped, context, (uint64_t)ptr);

                // we're done with this allocation
                g_graph_allocs.erase(alloc_iter);
            } else {
                // The graph still exists (or is still being captured).
                // Unreference the pointer so it will be deallocated once the graph instance
                // is destroyed. This keeps the memory alive for the lifetime of the graph,
                // which is required when the free occurs during conditional body capture:
                // the body may reference the allocation on any launch of the graph.
                alloc_info.ref_exists = false;
            }
        }
    }
}


bool wp_memcpy_h2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);
    WP_HIP_ISSUE(cuda_stream);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy HtoD");

#if defined(__HIP_PLATFORM_AMD__)
    bool result = hip_priority_copy(dest, src, n, cudaMemcpyHostToDevice, cuda_stream);  // Split into pieces when high-priority work runs.
#else
    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, cuda_stream));
#endif
#if defined(__HIP_PLATFORM_AMD__)
    if (result && n > 0)
        result = hip_complete_pageable_copy(cuda_stream, src);
#endif

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}

bool wp_memcpy_d2h(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);
    WP_HIP_ISSUE(cuda_stream);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy DtoH");

#if defined(__HIP_PLATFORM_AMD__)
    bool result = hip_priority_copy(dest, src, n, cudaMemcpyDeviceToHost, cuda_stream);  // Split into pieces when high-priority work runs.
#else
    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, cuda_stream));
#endif
#if defined(__HIP_PLATFORM_AMD__)
    if (result && n > 0)
        result = hip_complete_pageable_copy(cuda_stream, dest);
#endif

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}

bool wp_memcpy_d2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);
    WP_HIP_ISSUE(cuda_stream);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy DtoD");

#if defined(__HIP_PLATFORM_AMD__)
    bool result = hip_priority_copy(dest, src, n, cudaMemcpyDeviceToDevice, cuda_stream);  // Split into pieces when high-priority work runs.
#else
    bool result = check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, cuda_stream));
#endif

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    // APIC recording.
    // TODO: When execution becomes fully deferred (like the CPU path), move
    // CUDA-result checking to replay time (capture_launch / load+launch) and
    // make recording unconditional at capture time. For now we still execute
    // the CUDA op live under stream capture, but the record itself is API-
    // intent only and doesn't depend on the live call's result.
    APICState* apic_state = wp_apic_get_cuda_recording_state();
    if (apic_state && n > 0) {
        APICAddress dst_addr = apic_resolve_live_ptr(apic_state, (uint64_t)dest, n);
        APICAddress src_addr = apic_resolve_live_ptr(apic_state, (uint64_t)src, n);
        apic_record_memcpy_d2d(apic_state, dst_addr.region_id, dst_addr.offset, src_addr.region_id, src_addr.offset, n);
    }

    return result;
}

bool wp_memcpy_p2p(void* dst_context, void* dst, void* src_context, void* src, size_t n, void* stream)
{
    // ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(dst_context);
    WP_HIP_ISSUE(cuda_stream);

    // Notes:
    // - cuMemcpyPeerAsync() works fine with both regular and pooled allocations (cudaMalloc() and cudaMallocAsync(),
    // respectively)
    //   when not capturing a graph.
    // - cuMemcpyPeerAsync() is not supported during graph capture, so we must use cudaMemcpyAsync() with
    // kind=cudaMemcpyDefault.
    // - cudaMemcpyAsync() works fine with regular allocations, but doesn't work with pooled allocations
    //   unless mempool access has been enabled.
    // - There is no reliable way to check if mempool access is enabled during graph capture,
    //   because cudaMemPoolGetAccess() cannot be called during graph capture.
    // - CUDA will report error 1 (invalid argument) if cudaMemcpyAsync() is called but mempool access is not enabled.

    if (!wp_cuda_stream_is_capturing(stream)) {
        begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, get_stream_context(stream), "memcpy PtoP");

        bool result = check_cu(cuMemcpyPeerAsync_f(
            (CUdeviceptr)dst, (CUcontext)dst_context, (CUdeviceptr)src, (CUcontext)src_context, n, cuda_stream
        ));

        end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

        return result;
    } else {
        cudaError_t result = cudaSuccess;

        // cudaMemcpyAsync() is sensitive to the bound context to resolve pointer locations.
        // If fails with cudaErrorInvalidValue if it cannot resolve an argument.
        // We first try the copy in the destination context, then if it fails we retry in the source context.
        // The cudaErrorInvalidValue error doesn't cause graph capture to fail, so it's ok to retry.
        // Since this trial-and-error shenanigans only happens during capture, there
        // is no perf impact when the graph is launched.
        // For bonus points, this approach simplifies memory pool access requirements.
        // Access only needs to be enabled one way, either from the source device to the destination device
        // or vice versa.  Sometimes, when it's really quiet, you can actually hear my genius.
        {
            // try doing the copy in the destination context
            ContextGuard guard(dst_context);
            result = cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, cuda_stream);

            if (result != cudaSuccess) {
                // clear error in destination context
                ignore_cuda_error(cudaGetLastError());

                // try doing the copy in the source context
                ContextGuard guard(src_context);
                result = cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, cuda_stream);

                // clear error in source context
                ignore_cuda_error(cudaGetLastError());
            }
        }

        // If the copy failed, try to detect if mempool allocations are involved to generate a helpful error message.
        if (!check_cuda(result)) {
            if (result == cudaErrorInvalidValue && src != NULL && dst != NULL) {
                // check if either of the pointers was allocated from a mempool
                void* src_mempool = NULL;
                void* dst_mempool = NULL;
                ignore_cu_result(cuPointerGetAttribute_f(&src_mempool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)src));
                ignore_cu_result(cuPointerGetAttribute_f(&dst_mempool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)dst));
                ignore_cuda_error(cudaGetLastError());  // clear any errors
                // check if either of the pointers was allocated during graph capture
                auto src_alloc = g_graph_allocs.find(src);
                auto dst_alloc = g_graph_allocs.find(dst);
                if (src_mempool != NULL || src_alloc != g_graph_allocs.end() || dst_mempool != NULL
                    || dst_alloc != g_graph_allocs.end()) {
                    wp::append_error_string(
                        "*** CUDA mempool allocations were used in a peer-to-peer copy during graph capture."
                    );
                    wp::append_error_string(
                        "*** This operation fails if mempool access is not enabled between the peer devices."
                    );
                    wp::append_error_string(
                        "*** Either enable mempool access between the devices or use the default CUDA allocator"
                    );
                    wp::append_error_string("*** to pre-allocate the arrays before graph capture begins.");
                }
            }

            return false;
        }

        return true;
    }
}

bool wp_memcpy_batch(void* context, void** dsts, void** srcs, size_t* sizes, size_t count, void* stream)
{
    ContextGuard guard(context);

    CUstream cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream(context);
    WP_HIP_ISSUE(cuda_stream);

    begin_cuda_range(WP_TIMING_MEMCPY, cuda_stream, context, "memcpy batch");

    bool result = true;

#if WP_HAS_MEMCPY_BATCH
    bool use_batch = true;
#if !defined(__HIP_PLATFORM_AMD__)
    use_batch = (wp_cuda_driver_version() >= 12080);
#endif
    if (use_batch) {
        CUmemcpyAttributes attr = {};
        attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        // attr.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        size_t attr_idx = 0;
        size_t fail_idx = 0;
        result = check_cuda(cuMemcpyBatchAsync_f(
            (CUdeviceptr*)dsts, (CUdeviceptr*)srcs, sizes, count, &attr, &attr_idx, 1, &fail_idx, cuda_stream
        ));
    } else {
        for (size_t i = 0; i < count; i++)
            result = result && check_cuda(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, cuda_stream));
    }
#else
    for (size_t i = 0; i < count; i++)
        result = result && check_cuda(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, cuda_stream));
#endif

    end_cuda_range(WP_TIMING_MEMCPY, cuda_stream);

    return result;
}


__global__ void memset_kernel(int* dest, int value, size_t n)
{
    // Grid-stride loop: the launch may clamp the grid (notably on HIP, where the
    // total work-item count is limited to 2^32-1), so each thread must process
    // multiple elements to cover all of `n`.
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t tid
         = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         tid < n; tid += grid_stride) {
        dest[tid] = value;
    }
}

bool wp_memset_device(void* context, void* dest, int value, size_t n, void* stream)
{
    // A zero-length memset is a no-op. CUDA's cudaMemsetAsync accepts (dest,0)
    // (even a null dest) and returns success, but ROCm's hipMemsetAsync rejects
    // it with hipErrorInvalidValue ("invalid argument"). Short-circuit to match
    // CUDA semantics; this removes the spurious errors emitted by zero-length
    // array .zero_() calls (e.g. during hipGraph capture warmup on gfx942).
    if (n == 0)
        return true;

    ContextGuard guard(context);

    cudaStream_t cuda_stream;
    if (stream != WP_CURRENT_STREAM)
        cuda_stream = static_cast<CUstream>(stream);
    else
        cuda_stream = get_current_stream();
    WP_HIP_ISSUE(cuda_stream);

    begin_cuda_range(WP_TIMING_MEMSET, cuda_stream, context, "memset");

    bool result = check_cuda(cudaMemsetAsync(dest, value, n, cuda_stream));

    end_cuda_range(WP_TIMING_MEMSET, cuda_stream);

    // APIC recording.
    // TODO: When execution becomes fully deferred (like the CPU path), move
    // CUDA-result checking to replay time (capture_launch / load+launch) and
    // make recording unconditional at capture time. For now we still execute
    // the CUDA op live under stream capture, but the record itself is API-
    // intent only and doesn't depend on the live call's result.
    APICState* apic_state = wp_apic_get_cuda_recording_state();
    if (apic_state && n > 0) {
        APICAddress addr = apic_resolve_live_ptr(apic_state, (uint64_t)dest, n);
        apic_record_memset(apic_state, addr.region_id, addr.offset, n, value);
    }
    return result;
}

// POD value buffer passed by value to fill kernels so they don't need to read the
// fill bytes from a device pointer (which would otherwise require host->device
// staging through capturable_tmp_alloc + pause/resume capture, which breaks under
// forked-stream and shared-graph captures). The bucket sizes trade off kernel
// instantiation count against support for reasonable user-defined struct dtypes.
constexpr size_t WP_FILL_VALUE_INLINE_BYTES_0 = 256;
constexpr size_t WP_FILL_VALUE_INLINE_BYTES_1 = 1024;
constexpr size_t WP_FILL_VALUE_INLINE_BYTES_2 = 3968;

template <size_t N> struct FillValue {
    uint8_t bytes[N];
};

template <size_t N> static FillValue<N> make_fill_value(const void* src, size_t srcsize)
{
    FillValue<N> value = {};
    memcpy(value.bytes, src, srcsize);
    return value;
}

// fill memory buffer with a value: generic memtile kernel using memcpy for each element
__global__ void memtile_kernel(void* dst, const void* src, size_t srcsize, size_t n)
{
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t tid
         = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         tid < n; tid += grid_stride) {
        memcpy((int8_t*)dst + srcsize * tid, src, srcsize);
    }
}

template <size_t N> __global__ void memtile_kernel_by_value(void* dst, FillValue<N> value, size_t srcsize, size_t n)
{
    size_t tid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
    if (tid < n) {
        memcpy((int8_t*)dst + srcsize * tid, value.bytes, srcsize);
    }
}

template <size_t N> static void launch_memtile_kernel_by_value(void* dst, const void* src, size_t srcsize, size_t n)
{
    FillValue<N> value = make_fill_value<N>(src, srcsize);
    wp_launch_device(WP_CURRENT_CONTEXT, (memtile_kernel_by_value<N>), n, (dst, value, srcsize, n));
}

static bool launch_memtile_kernel_by_value(void* dst, const void* src, size_t srcsize, size_t n)
{
    if (srcsize <= WP_FILL_VALUE_INLINE_BYTES_0) {
        launch_memtile_kernel_by_value<WP_FILL_VALUE_INLINE_BYTES_0>(dst, src, srcsize, n);
        return true;
    }
    if (srcsize <= WP_FILL_VALUE_INLINE_BYTES_1) {
        launch_memtile_kernel_by_value<WP_FILL_VALUE_INLINE_BYTES_1>(dst, src, srcsize, n);
        return true;
    }
    if (srcsize <= WP_FILL_VALUE_INLINE_BYTES_2) {
        launch_memtile_kernel_by_value<WP_FILL_VALUE_INLINE_BYTES_2>(dst, src, srcsize, n);
        return true;
    }
    return false;
}

// this should be faster than memtile_kernel, but requires proper alignment of dst
template <typename T> __global__ void memtile_value_kernel(T* dst, T value, size_t n)
{
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t tid
         = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         tid < n; tid += grid_stride) {
        dst[tid] = value;
    }
}

// Record-and-execute a contiguous memtile (multi-byte arr.fill_) under CUDA APIC
// capture: record the fill value and destination into the byte stream, then fall
// through so the live tile issues onto the captured stream. Only small fills are
// recordable -- the value is embedded inline and replayed by value. Large fills
// (> WP_FILL_VALUE_INLINE_BYTES_2) use capturable_tmp_alloc (pause/resume +
// device staging) that the byte-stream rebuild cannot reproduce; the Python
// fill_() path rejects those under CUDA APIC capture, so they never reach here
// during a capture. No-op outside a CUDA APIC capture.
static void apic_capture_memtile_device(void* dst, const void* src, size_t srcsize, size_t n)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n == 0 || srcsize == 0 || srcsize > WP_FILL_VALUE_INLINE_BYTES_2)
        return;
    // Guard the byte-span multiplication against overflow (srcsize is already
    // bounded above, so this is defensive) before resolving the region.
    if (n > (~static_cast<size_t>(0)) / srcsize)
        return;
    APICAddress addr = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(dst), srcsize * n);
    apic_record_memtile(state, addr.region_id, addr.offset, static_cast<uint32_t>(srcsize), src, n);
}

void wp_memtile_device(void* context, void* dst, const void* src, size_t srcsize, size_t n)
{
    ContextGuard guard(context);

    apic_capture_memtile_device(dst, src, srcsize, n);

    size_t dst_addr = reinterpret_cast<size_t>(dst);
    size_t src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0) {
        int64_t* p = reinterpret_cast<int64_t*>(dst);
        int64_t value = *reinterpret_cast<const int64_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0) {
        int32_t* p = reinterpret_cast<int32_t*>(dst);
        int32_t value = *reinterpret_cast<const int32_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0) {
        int16_t* p = reinterpret_cast<int16_t*>(dst);
        int16_t value = *reinterpret_cast<const int16_t*>(src);
        wp_launch_device(WP_CURRENT_CONTEXT, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 1) {
        wp_memset_device(context, dst, *reinterpret_cast<const int8_t*>(src), n, WP_CURRENT_STREAM);
    } else if (!launch_memtile_kernel_by_value(dst, src, srcsize, n)) {
        // generic fallback for values too large to pass through kernel args
        void* value_devptr = NULL;  // fill value in device memory
        bool free_devptr = true;  // whether we need to free the memory

        if (!capturable_tmp_alloc(WP_CURRENT_CONTEXT, src, srcsize, &value_devptr, &free_devptr)) {
            fprintf(stderr, "Warp fill error: failed to copy value to device memory\n");
            return;
        }

        wp_launch_device(WP_CURRENT_CONTEXT, memtile_kernel, n, (dst, value_devptr, srcsize, n));

        if (free_devptr) {
            wp_free_device(WP_CURRENT_CONTEXT, value_devptr);
        }
    }
}


static __global__ void array_copy_1d_kernel(
    void* dst,
    const void* src,
    size_t dst_stride,
    size_t src_stride,
    const int* dst_indices,
    const int* src_indices,
    size_t n,
    size_t elem_size
)
{
    // Grid-stride loop: the launch may clamp the grid (notably on HIP, where the
    // total work-item count is limited to 2^32-1), so each thread must process
    // multiple elements to cover all of `n`.
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); i < n; i += grid_stride) {
        size_t src_idx = src_indices ? src_indices[i] : i;
        size_t dst_idx = dst_indices ? dst_indices[i] : i;
        const char* p = (const char*)src + src_idx * src_stride;
        char* q = (char*)dst + dst_idx * dst_stride;
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_2d_kernel(
    void* dst,
    const void* src,
    wp::vec_t<2, size_t> dst_strides,
    wp::vec_t<2, size_t> src_strides,
    wp::vec_t<2, const int*> dst_indices,
    wp::vec_t<2, const int*> src_indices,
    wp::vec_t<2, size_t> shape,
    size_t elem_size
)
{
    size_t n = shape[1];
    size_t total = shape[0] * n;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / n;
        size_t j = tid % n;
        size_t src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        size_t dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        size_t src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        size_t dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        const char* p = (const char*)src + src_idx0 * src_strides[0] + src_idx1 * src_strides[1];
        char* q = (char*)dst + dst_idx0 * dst_strides[0] + dst_idx1 * dst_strides[1];
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_3d_kernel(
    void* dst,
    const void* src,
    wp::vec_t<3, size_t> dst_strides,
    wp::vec_t<3, size_t> src_strides,
    wp::vec_t<3, const int*> dst_indices,
    wp::vec_t<3, const int*> src_indices,
    wp::vec_t<3, size_t> shape,
    size_t elem_size
)
{
    size_t n = shape[1];
    size_t o = shape[2];
    size_t total = shape[0] * n * o;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / (n * o);
        size_t j = tid % (n * o) / o;
        size_t k = tid % o;
        size_t src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        size_t dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        size_t src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        size_t dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        size_t src_idx2 = src_indices[2] ? src_indices[2][k] : k;
        size_t dst_idx2 = dst_indices[2] ? dst_indices[2][k] : k;
        const char* p
            = (const char*)src + src_idx0 * src_strides[0] + src_idx1 * src_strides[1] + src_idx2 * src_strides[2];
        char* q = (char*)dst + dst_idx0 * dst_strides[0] + dst_idx1 * dst_strides[1] + dst_idx2 * dst_strides[2];
        memcpy(q, p, elem_size);
    }
}

static __global__ void array_copy_4d_kernel(
    void* dst,
    const void* src,
    wp::vec_t<4, size_t> dst_strides,
    wp::vec_t<4, size_t> src_strides,
    wp::vec_t<4, const int*> dst_indices,
    wp::vec_t<4, const int*> src_indices,
    wp::vec_t<4, size_t> shape,
    size_t elem_size
)
{
    size_t n = shape[1];
    size_t o = shape[2];
    size_t pdim = shape[3];
    size_t total = shape[0] * n * o * pdim;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / (n * o * pdim);
        size_t j = tid % (n * o * pdim) / (o * pdim);
        size_t k = tid % (o * pdim) / pdim;
        size_t l = tid % pdim;
        size_t src_idx0 = src_indices[0] ? src_indices[0][i] : i;
        size_t dst_idx0 = dst_indices[0] ? dst_indices[0][i] : i;
        size_t src_idx1 = src_indices[1] ? src_indices[1][j] : j;
        size_t dst_idx1 = dst_indices[1] ? dst_indices[1][j] : j;
        size_t src_idx2 = src_indices[2] ? src_indices[2][k] : k;
        size_t dst_idx2 = dst_indices[2] ? dst_indices[2][k] : k;
        size_t src_idx3 = src_indices[3] ? src_indices[3][l] : l;
        size_t dst_idx3 = dst_indices[3] ? dst_indices[3][l] : l;
        const char* p = (const char*)src + src_idx0 * src_strides[0] + src_idx1 * src_strides[1]
            + src_idx2 * src_strides[2] + src_idx3 * src_strides[3];
        char* q = (char*)dst + dst_idx0 * dst_strides[0] + dst_idx1 * dst_strides[1] + dst_idx2 * dst_strides[2]
            + dst_idx3 * dst_strides[3];
        memcpy(q, p, elem_size);
    }
}


static __global__ void array_copy_from_fabric_kernel(
    wp::fabricarray_t<void> src, void* dst_data, size_t dst_stride, const int* dst_indices, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < src.size; tid += grid_stride) {
        size_t dst_idx = dst_indices ? dst_indices[tid] : tid;
        void* dst_ptr = (char*)dst_data + dst_idx * dst_stride;
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_from_fabric_indexed_kernel(
    wp::indexedfabricarray_t<void> src, void* dst_data, size_t dst_stride, const int* dst_indices, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < src.size; tid += grid_stride) {
        size_t src_index = src.indices[tid];
        size_t dst_idx = dst_indices ? dst_indices[tid] : tid;
        void* dst_ptr = (char*)dst_data + dst_idx * dst_stride;
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_to_fabric_kernel(
    wp::fabricarray_t<void> dst, const void* src_data, size_t src_stride, const int* src_indices, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        size_t src_idx = src_indices ? src_indices[tid] : tid;
        const void* src_ptr = (const char*)src_data + src_idx * src_stride;
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static __global__ void array_copy_to_fabric_indexed_kernel(
    wp::indexedfabricarray_t<void> dst,
    const void* src_data,
    size_t src_stride,
    const int* src_indices,
    size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        size_t src_idx = src_indices ? src_indices[tid] : tid;
        const void* src_ptr = (const char*)src_data + src_idx * src_stride;
        size_t dst_idx = dst.indices[tid];
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void
array_copy_fabric_to_fabric_kernel(wp::fabricarray_t<void> dst, wp::fabricarray_t<void> src, size_t elem_size)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_to_fabric_indexed_kernel(
    wp::indexedfabricarray_t<void> dst, wp::fabricarray_t<void> src, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        const void* src_ptr = fabricarray_element_ptr(src, tid, elem_size);
        size_t dst_index = dst.indices[tid];
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_indexed_to_fabric_kernel(
    wp::fabricarray_t<void> dst, wp::indexedfabricarray_t<void> src, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        size_t src_index = src.indices[tid];
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst, tid, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static __global__ void array_copy_fabric_indexed_to_fabric_indexed_kernel(
    wp::indexedfabricarray_t<void> dst, wp::indexedfabricarray_t<void> src, size_t elem_size
)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < dst.size; tid += grid_stride) {
        size_t src_index = src.indices[tid];
        size_t dst_index = dst.indices[tid];
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_index, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_index, elem_size);
        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


WP_API bool wp_array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    if (!src || !dst)
        return false;

    const void* src_data = NULL;
    void* dst_data = NULL;
    int src_ndim = 0;
    int dst_ndim = 0;
    const int* src_shape = NULL;
    const int* dst_shape = NULL;
    const int* src_strides = NULL;
    const int* dst_strides = NULL;
    const int* const* src_indices = NULL;
    const int* const* dst_indices = NULL;

    const wp::fabricarray_t<void>* src_fabricarray = NULL;
    wp::fabricarray_t<void>* dst_fabricarray = NULL;

    const wp::indexedfabricarray_t<void>* src_indexedfabricarray = NULL;
    wp::indexedfabricarray_t<void>* dst_indexedfabricarray = NULL;

    const int* null_indices[wp::ARRAY_MAX_DIMS] = { NULL };

    if (src_type == wp::ARRAY_TYPE_REGULAR) {
        const wp::array_t<void>& src_arr = *static_cast<const wp::array_t<void>*>(src);
        src_data = src_arr.data;
        src_ndim = src_arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.strides;
        src_indices = null_indices;
    } else if (src_type == wp::ARRAY_TYPE_INDEXED) {
        const wp::indexedarray_t<void>& src_arr = *static_cast<const wp::indexedarray_t<void>*>(src);
        src_data = src_arr.arr.data;
        src_ndim = src_arr.arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.arr.strides;
        src_indices = src_arr.indices;
    } else if (src_type == wp::ARRAY_TYPE_FABRIC) {
        src_fabricarray = static_cast<const wp::fabricarray_t<void>*>(src);
        src_ndim = 1;
    } else if (src_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        src_indexedfabricarray = static_cast<const wp::indexedfabricarray_t<void>*>(src);
        src_ndim = 1;
    } else {
        fprintf(stderr, "Warp copy error: Invalid array type (%d)\n", src_type);
        return false;
    }

    if (dst_type == wp::ARRAY_TYPE_REGULAR) {
        const wp::array_t<void>& dst_arr = *static_cast<const wp::array_t<void>*>(dst);
        dst_data = dst_arr.data;
        dst_ndim = dst_arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.strides;
        dst_indices = null_indices;
    } else if (dst_type == wp::ARRAY_TYPE_INDEXED) {
        const wp::indexedarray_t<void>& dst_arr = *static_cast<const wp::indexedarray_t<void>*>(dst);
        dst_data = dst_arr.arr.data;
        dst_ndim = dst_arr.arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.arr.strides;
        dst_indices = dst_arr.indices;
    } else if (dst_type == wp::ARRAY_TYPE_FABRIC) {
        dst_fabricarray = static_cast<wp::fabricarray_t<void>*>(dst);
        dst_ndim = 1;
    } else if (dst_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        dst_indexedfabricarray = static_cast<wp::indexedfabricarray_t<void>*>(dst);
        dst_ndim = 1;
    } else {
        fprintf(stderr, "Warp copy error: Invalid array type (%d)\n", dst_type);
        return false;
    }

    if (src_ndim != dst_ndim) {
        fprintf(stderr, "Warp copy error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return false;
    }

    ContextGuard guard(context);

    // handle fabric arrays
    if (dst_fabricarray) {
        size_t n = dst_fabricarray->size;
        if (src_fabricarray) {
            // copy from fabric to fabric
            if (src_fabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_kernel, n,
                (*dst_fabricarray, *src_fabricarray, elem_size)
            );
            return true;
        } else if (src_indexedfabricarray) {
            // copy from fabric indexed to fabric
            if (src_indexedfabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_kernel, n,
                (*dst_fabricarray, *src_indexedfabricarray, elem_size)
            );
            return true;
        } else {
            // copy to fabric
            if (size_t(src_shape[0]) != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_to_fabric_kernel, n,
                (*dst_fabricarray, src_data, src_strides[0], src_indices[0], elem_size)
            );
            return true;
        }
    }
    if (dst_indexedfabricarray) {
        size_t n = dst_indexedfabricarray->size;
        if (src_fabricarray) {
            // copy from fabric to fabric indexed
            if (src_fabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_fabric_to_fabric_indexed_kernel, n,
                (*dst_indexedfabricarray, *src_fabricarray, elem_size)
            );
            return true;
        } else if (src_indexedfabricarray) {
            // copy from fabric indexed to fabric indexed
            if (src_indexedfabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_fabric_indexed_to_fabric_indexed_kernel, n,
                (*dst_indexedfabricarray, *src_indexedfabricarray, elem_size)
            );
            return true;
        } else {
            // copy to fabric indexed
            if (size_t(src_shape[0]) != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_copy_to_fabric_indexed_kernel, n,
                (*dst_indexedfabricarray, src_data, src_strides[0], src_indices[0], elem_size)
            );
            return true;
        }
    } else if (src_fabricarray) {
        // copy from fabric
        size_t n = src_fabricarray->size;
        if (size_t(dst_shape[0]) != n) {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_from_fabric_kernel, n,
            (*src_fabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size)
        );
        return true;
    } else if (src_indexedfabricarray) {
        // copy from fabric indexed
        size_t n = src_indexedfabricarray->size;
        if (size_t(dst_shape[0]) != n) {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_from_fabric_indexed_kernel, n,
            (*src_indexedfabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size)
        );
        return true;
    }

    size_t n = 1;
    for (int i = 0; i < src_ndim; i++) {
        if (src_shape[i] != dst_shape[i]) {
            fprintf(stderr, "Warp copy error: Incompatible array shapes\n");
            return false;
        }
        n *= src_shape[i];
    }

    switch (src_ndim) {
    case 1: {
        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_1d_kernel, n,
            (dst_data, src_data, dst_strides[0], src_strides[0], dst_indices[0], src_indices[0], src_shape[0],
             elem_size)
        );
        break;
    }
    case 2: {
        wp::vec_t<2, size_t> shape_v(src_shape[0], src_shape[1]);
        wp::vec_t<2, size_t> src_strides_v(src_strides[0], src_strides[1]);
        wp::vec_t<2, size_t> dst_strides_v(dst_strides[0], dst_strides[1]);
        wp::vec_t<2, const int*> src_indices_v(src_indices[0], src_indices[1]);
        wp::vec_t<2, const int*> dst_indices_v(dst_indices[0], dst_indices[1]);

        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_2d_kernel, n,
            (dst_data, src_data, dst_strides_v, src_strides_v, dst_indices_v, src_indices_v, shape_v, elem_size)
        );
        break;
    }
    case 3: {
        wp::vec_t<3, size_t> shape_v(src_shape[0], src_shape[1], src_shape[2]);
        wp::vec_t<3, size_t> src_strides_v(src_strides[0], src_strides[1], src_strides[2]);
        wp::vec_t<3, size_t> dst_strides_v(dst_strides[0], dst_strides[1], dst_strides[2]);
        wp::vec_t<3, const int*> src_indices_v(src_indices[0], src_indices[1], src_indices[2]);
        wp::vec_t<3, const int*> dst_indices_v(dst_indices[0], dst_indices[1], dst_indices[2]);

        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_3d_kernel, n,
            (dst_data, src_data, dst_strides_v, src_strides_v, dst_indices_v, src_indices_v, shape_v, elem_size)
        );
        break;
    }
    case 4: {
        wp::vec_t<4, size_t> shape_v(src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        wp::vec_t<4, size_t> src_strides_v(src_strides[0], src_strides[1], src_strides[2], src_strides[3]);
        wp::vec_t<4, size_t> dst_strides_v(dst_strides[0], dst_strides[1], dst_strides[2], dst_strides[3]);
        wp::vec_t<4, const int*> src_indices_v(src_indices[0], src_indices[1], src_indices[2], src_indices[3]);
        wp::vec_t<4, const int*> dst_indices_v(dst_indices[0], dst_indices[1], dst_indices[2], dst_indices[3]);

        wp_launch_device(
            WP_CURRENT_CONTEXT, array_copy_4d_kernel, n,
            (dst_data, src_data, dst_strides_v, src_strides_v, dst_indices_v, src_indices_v, shape_v, elem_size)
        );
        break;
    }
    default:
        fprintf(stderr, "Warp copy error: invalid array dimensionality (%d)\n", src_ndim);
        return false;
    }

    return check_cuda(cudaGetLastError());
}


// "by_value" variants take the fill bytes inline through the kernel-arg buffer
// using a bucketed ``FillValue<N>`` POD. They are graph-capturable on any stream
// (no host->device staging). The ``const void*`` originals below are kept for the
// fallback case when the fill value is too large for kernel args.
template <size_t N>
static __global__ void array_fill_1d_kernel_by_value(
    void* data, size_t n, size_t stride, const int* indices, FillValue<N> value, size_t value_size
)
{
    size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (i < n) {
        size_t idx = indices ? indices[i] : i;
        char* p = (char*)data + idx * stride;
        memcpy(p, value.bytes, value_size);
    }
}

template <size_t N>
static __global__ void array_fill_2d_kernel_by_value(
    void* data,
    wp::vec_t<2, size_t> shape,
    wp::vec_t<2, size_t> strides,
    wp::vec_t<2, const int*> indices,
    FillValue<N> value,
    size_t value_size
)
{
    size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    size_t n = shape[1];
    size_t i = tid / n;
    size_t j = tid % n;
    if (i < shape[0] /*&& j < shape[1]*/) {
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1];
        memcpy(p, value.bytes, value_size);
    }
}

template <size_t N>
static __global__ void array_fill_3d_kernel_by_value(
    void* data,
    wp::vec_t<3, size_t> shape,
    wp::vec_t<3, size_t> strides,
    wp::vec_t<3, const int*> indices,
    FillValue<N> value,
    size_t value_size
)
{
    size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    size_t n = shape[1];
    size_t o = shape[2];
    size_t i = tid / (n * o);
    size_t j = tid % (n * o) / o;
    size_t k = tid % o;
    if (i < shape[0] && j < shape[1] /*&& k < shape[2]*/) {
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        size_t idx2 = indices[2] ? indices[2][k] : k;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2];
        memcpy(p, value.bytes, value_size);
    }
}

template <size_t N>
static __global__ void array_fill_4d_kernel_by_value(
    void* data,
    wp::vec_t<4, size_t> shape,
    wp::vec_t<4, size_t> strides,
    wp::vec_t<4, const int*> indices,
    FillValue<N> value,
    size_t value_size
)
{
    size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    size_t n = shape[1];
    size_t o = shape[2];
    size_t p = shape[3];
    size_t i = tid / (n * o * p);
    size_t j = tid % (n * o * p) / (o * p);
    size_t k = tid % (o * p) / p;
    size_t l = tid % p;
    if (i < shape[0] && j < shape[1] && k < shape[2] /*&& l < shape[3]*/) {
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        size_t idx2 = indices[2] ? indices[2][k] : k;
        size_t idx3 = indices[3] ? indices[3][l] : l;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2] + idx3 * strides[3];
        memcpy(p, value.bytes, value_size);
    }
}

template <size_t N>
static __global__ void
array_fill_fabric_kernel_by_value(wp::fabricarray_t<void> fa, FillValue<N> value, size_t value_size)
{
    size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (tid < fa.size) {
        void* dst_ptr = fabricarray_element_ptr(fa, tid, value_size);
        memcpy(dst_ptr, value.bytes, value_size);
    }
}

template <size_t N>
static __global__ void
array_fill_fabric_indexed_kernel_by_value(wp::indexedfabricarray_t<void> ifa, FillValue<N> value, size_t value_size)
{
    size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (tid < ifa.size) {
        size_t idx = size_t(ifa.indices[tid]);
        if (idx < ifa.fa.size) {
            void* dst_ptr = fabricarray_element_ptr(ifa.fa, idx, value_size);
            memcpy(dst_ptr, value.bytes, value_size);
        }
    }
}

template <size_t N>
static void launch_array_fill_by_value(
    void* data,
    int ndim,
    const int* shape,
    const int* strides,
    const int* const* indices,
    wp::fabricarray_t<void>* fa,
    wp::indexedfabricarray_t<void>* ifa,
    size_t n,
    const void* value_ptr,
    size_t value_size
)
{
    FillValue<N> value = make_fill_value<N>(value_ptr, value_size);

    if (fa) {
        wp_launch_device(WP_CURRENT_CONTEXT, (array_fill_fabric_kernel_by_value<N>), n, (*fa, value, value_size));
    } else if (ifa) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, (array_fill_fabric_indexed_kernel_by_value<N>), n, (*ifa, value, value_size)
        );
    } else {
        switch (ndim) {
        case 1: {
            wp_launch_device(
                WP_CURRENT_CONTEXT, (array_fill_1d_kernel_by_value<N>), n,
                (data, shape[0], strides[0], indices[0], value, value_size)
            );
            break;
        }
        case 2: {
            wp::vec_t<2, size_t> shape_v(shape[0], shape[1]);
            wp::vec_t<2, size_t> strides_v(strides[0], strides[1]);
            wp::vec_t<2, const int*> indices_v(indices[0], indices[1]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, (array_fill_2d_kernel_by_value<N>), n,
                (data, shape_v, strides_v, indices_v, value, value_size)
            );
            break;
        }
        case 3: {
            wp::vec_t<3, size_t> shape_v(shape[0], shape[1], shape[2]);
            wp::vec_t<3, size_t> strides_v(strides[0], strides[1], strides[2]);
            wp::vec_t<3, const int*> indices_v(indices[0], indices[1], indices[2]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, (array_fill_3d_kernel_by_value<N>), n,
                (data, shape_v, strides_v, indices_v, value, value_size)
            );
            break;
        }
        case 4: {
            wp::vec_t<4, size_t> shape_v(shape[0], shape[1], shape[2], shape[3]);
            wp::vec_t<4, size_t> strides_v(strides[0], strides[1], strides[2], strides[3]);
            wp::vec_t<4, const int*> indices_v(indices[0], indices[1], indices[2], indices[3]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, (array_fill_4d_kernel_by_value<N>), n,
                (data, shape_v, strides_v, indices_v, value, value_size)
            );
            break;
        }
        default:
            fprintf(stderr, "Warp fill error: invalid array dimensionality (%d)\n", ndim);
            break;
        }
    }
}

static bool launch_array_fill_by_value(
    void* data,
    int ndim,
    const int* shape,
    const int* strides,
    const int* const* indices,
    wp::fabricarray_t<void>* fa,
    wp::indexedfabricarray_t<void>* ifa,
    size_t n,
    const void* value_ptr,
    size_t value_size
)
{
    if (value_size <= WP_FILL_VALUE_INLINE_BYTES_0) {
        launch_array_fill_by_value<WP_FILL_VALUE_INLINE_BYTES_0>(
            data, ndim, shape, strides, indices, fa, ifa, n, value_ptr, value_size
        );
        return true;
    }
    if (value_size <= WP_FILL_VALUE_INLINE_BYTES_1) {
        launch_array_fill_by_value<WP_FILL_VALUE_INLINE_BYTES_1>(
            data, ndim, shape, strides, indices, fa, ifa, n, value_ptr, value_size
        );
        return true;
    }
    if (value_size <= WP_FILL_VALUE_INLINE_BYTES_2) {
        launch_array_fill_by_value<WP_FILL_VALUE_INLINE_BYTES_2>(
            data, ndim, shape, strides, indices, fa, ifa, n, value_ptr, value_size
        );
        return true;
    }
    return false;
}


// Original ``const void*`` kernels: read the fill bytes from a device pointer
// staged via ``capturable_tmp_alloc``. Used as a fallback for fill values too
// large to fit in the inline kernel-arg buffers.
// This path is graph-capturable only on the begin stream of a capture.

static __global__ void
array_fill_1d_kernel(void* data, size_t n, size_t stride, const int* indices, const void* value, size_t value_size)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); i < n; i += grid_stride) {
        size_t idx = indices ? indices[i] : i;
        char* p = (char*)data + idx * stride;
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_2d_kernel(
    void* data,
    wp::vec_t<2, size_t> shape,
    wp::vec_t<2, size_t> strides,
    wp::vec_t<2, const int*> indices,
    const void* value,
    size_t value_size
)
{
    size_t n = shape[1];
    size_t total = shape[0] * n;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / n;
        size_t j = tid % n;
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1];
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_3d_kernel(
    void* data,
    wp::vec_t<3, size_t> shape,
    wp::vec_t<3, size_t> strides,
    wp::vec_t<3, const int*> indices,
    const void* value,
    size_t value_size
)
{
    size_t n = shape[1];
    size_t o = shape[2];
    size_t total = shape[0] * n * o;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / (n * o);
        size_t j = tid % (n * o) / o;
        size_t k = tid % o;
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        size_t idx2 = indices[2] ? indices[2][k] : k;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2];
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_4d_kernel(
    void* data,
    wp::vec_t<4, size_t> shape,
    wp::vec_t<4, size_t> strides,
    wp::vec_t<4, const int*> indices,
    const void* value,
    size_t value_size
)
{
    size_t n = shape[1];
    size_t o = shape[2];
    size_t pdim = shape[3];
    size_t total = shape[0] * n * o * pdim;
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < total; tid += grid_stride) {
        size_t i = tid / (n * o * pdim);
        size_t j = tid % (n * o * pdim) / (o * pdim);
        size_t k = tid % (o * pdim) / pdim;
        size_t l = tid % pdim;
        size_t idx0 = indices[0] ? indices[0][i] : i;
        size_t idx1 = indices[1] ? indices[1][j] : j;
        size_t idx2 = indices[2] ? indices[2][k] : k;
        size_t idx3 = indices[3] ? indices[3][l] : l;
        char* p = (char*)data + idx0 * strides[0] + idx1 * strides[1] + idx2 * strides[2] + idx3 * strides[3];
        memcpy(p, value, value_size);
    }
}

static __global__ void array_fill_fabric_kernel(wp::fabricarray_t<void> fa, const void* value, size_t value_size)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < fa.size; tid += grid_stride) {
        void* dst_ptr = fabricarray_element_ptr(fa, tid, value_size);
        memcpy(dst_ptr, value, value_size);
    }
}

static __global__ void
array_fill_fabric_indexed_kernel(wp::indexedfabricarray_t<void> ifa, const void* value, size_t value_size)
{
    const size_t grid_stride = size_t(gridDim.x) * size_t(blockDim.x);
    for (size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x); tid < ifa.size; tid += grid_stride) {
        size_t idx = size_t(ifa.indices[tid]);
        if (idx < ifa.fa.size) {
            void* dst_ptr = fabricarray_element_ptr(ifa.fa, idx, value_size);
            memcpy(dst_ptr, value, value_size);
        }
    }
}


WP_API void wp_array_fill_device(void* context, void* arr_ptr, int arr_type, const void* value_ptr, int value_size)
{
    if (!arr_ptr || !value_ptr)
        return;

    void* data = NULL;
    int ndim = 0;
    const int* shape = NULL;
    const int* strides = NULL;
    const int* const* indices = NULL;

    wp::fabricarray_t<void>* fa = NULL;
    wp::indexedfabricarray_t<void>* ifa = NULL;

    const int* null_indices[wp::ARRAY_MAX_DIMS] = { NULL };

    if (arr_type == wp::ARRAY_TYPE_REGULAR) {
        wp::array_t<void>& arr = *static_cast<wp::array_t<void>*>(arr_ptr);
        data = arr.data;
        ndim = arr.ndim;
        shape = arr.shape.dims;
        strides = arr.strides;
        indices = null_indices;
    } else if (arr_type == wp::ARRAY_TYPE_INDEXED) {
        wp::indexedarray_t<void>& ia = *static_cast<wp::indexedarray_t<void>*>(arr_ptr);
        data = ia.arr.data;
        ndim = ia.arr.ndim;
        shape = ia.shape.dims;
        strides = ia.arr.strides;
        indices = ia.indices;
    } else if (arr_type == wp::ARRAY_TYPE_FABRIC) {
        fa = static_cast<wp::fabricarray_t<void>*>(arr_ptr);
    } else if (arr_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        ifa = static_cast<wp::indexedfabricarray_t<void>*>(arr_ptr);
    } else {
        fprintf(stderr, "Warp fill error: Invalid array type id %d\n", arr_type);
        return;
    }

    size_t n = 1;
    for (int i = 0; i < ndim; i++)
        n *= shape[i];

    ContextGuard guard(context);

    void* value_devptr = NULL;
    bool free_devptr = true;

    // Prefer inline-by-value kernels so graph capture does not need temporary
    // device storage. Oversized values fall back to the staged pointer kernels.
    if (launch_array_fill_by_value(
            data, ndim, shape, strides, indices, fa, ifa, n, value_ptr, static_cast<size_t>(value_size)
        )) {
        return;
    }

    if (!capturable_tmp_alloc(WP_CURRENT_CONTEXT, value_ptr, value_size, &value_devptr, &free_devptr)) {
        fprintf(stderr, "Warp fill error: failed to copy value to device memory\n");
        return;
    }

    if (fa) {
        // handle fabric arrays
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_fabric_kernel, n, (*fa, value_devptr, value_size));
    } else if (ifa) {
        // handle indexed fabric arrays
        wp_launch_device(WP_CURRENT_CONTEXT, array_fill_fabric_indexed_kernel, n, (*ifa, value_devptr, value_size));
    } else {
        // handle regular or indexed arrays
        switch (ndim) {
        case 1: {
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_fill_1d_kernel, n,
                (data, shape[0], strides[0], indices[0], value_devptr, value_size)
            );
            break;
        }
        case 2: {
            wp::vec_t<2, size_t> shape_v(shape[0], shape[1]);
            wp::vec_t<2, size_t> strides_v(strides[0], strides[1]);
            wp::vec_t<2, const int*> indices_v(indices[0], indices[1]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_fill_2d_kernel, n,
                (data, shape_v, strides_v, indices_v, value_devptr, value_size)
            );
            break;
        }
        case 3: {
            wp::vec_t<3, size_t> shape_v(shape[0], shape[1], shape[2]);
            wp::vec_t<3, size_t> strides_v(strides[0], strides[1], strides[2]);
            wp::vec_t<3, const int*> indices_v(indices[0], indices[1], indices[2]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_fill_3d_kernel, n,
                (data, shape_v, strides_v, indices_v, value_devptr, value_size)
            );
            break;
        }
        case 4: {
            wp::vec_t<4, size_t> shape_v(shape[0], shape[1], shape[2], shape[3]);
            wp::vec_t<4, size_t> strides_v(strides[0], strides[1], strides[2], strides[3]);
            wp::vec_t<4, const int*> indices_v(indices[0], indices[1], indices[2], indices[3]);
            wp_launch_device(
                WP_CURRENT_CONTEXT, array_fill_4d_kernel, n,
                (data, shape_v, strides_v, indices_v, value_devptr, value_size)
            );
            break;
        }
        default:
            fprintf(stderr, "Warp fill error: invalid array dimensionality (%d)\n", ndim);
            break;
        }
    }

    if (free_devptr) {
        wp_free_device(WP_CURRENT_CONTEXT, value_devptr);
    }
}

// Record an array_scan into the APIC byte stream during a CUDA capture, then
// fall through so the live device scan still issues onto the captured stream
// (record-and-execute). Unlike the CPU host path (record-only), the CUDA op
// must execute so the driver captures it into the native graph; the byte
// stream additionally carries it for persistent .wrp save/load. No-op outside
// a CUDA APIC capture (wp_apic_get_cuda_recording_state returns null, including
// during a CPU-targeted capture and during graph rebuild).
static void apic_capture_array_scan_device(
    uint64_t in, uint64_t out, int len, int in_stride, int out_stride, int type_len, uint8_t dtype, bool inclusive
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || len <= 0)
        return;
    uint64_t scalar_size = apic_type_size(dtype);
    uint64_t src_bytes = apic_strided_access_bytes(len, in_stride, type_len, scalar_size);
    uint64_t dst_bytes = apic_strided_access_bytes(len, out_stride, type_len, scalar_size);
    if (src_bytes == 0 || dst_bytes == 0)
        return;
    APICAddress dst_addr = apic_resolve_live_ptr(state, out, dst_bytes);
    APICAddress src_addr = apic_resolve_live_ptr(state, in, src_bytes);
    apic_record_scan(
        state, dst_addr.region_id, dst_addr.offset, src_addr.region_id, src_addr.offset, static_cast<uint32_t>(len),
        in_stride, out_stride, type_len, dtype, inclusive ? uint8_t(1) : uint8_t(0)
    );
}

bool wp_array_scan_int_device(
    uint64_t in, uint64_t out, int len, int in_stride, int out_stride, int type_len, bool inclusive
)
{
    apic_capture_array_scan_device(in, out, len, in_stride, out_stride, type_len, APIC_TYPE_INT32, inclusive);
    return scan_device((const int*)in, (int*)out, len, in_stride, out_stride, type_len, inclusive);
}

bool wp_array_scan_int64_device(
    uint64_t in, uint64_t out, int len, int in_stride, int out_stride, int type_len, bool inclusive
)
{
    apic_capture_array_scan_device(in, out, len, in_stride, out_stride, type_len, APIC_TYPE_INT64, inclusive);
    return scan_device((const int64_t*)in, (int64_t*)out, len, in_stride, out_stride, type_len, inclusive);
}

bool wp_array_scan_float_device(
    uint64_t in, uint64_t out, int len, int in_stride, int out_stride, int type_len, bool inclusive
)
{
    apic_capture_array_scan_device(in, out, len, in_stride, out_stride, type_len, APIC_TYPE_FLOAT32, inclusive);
    return scan_device((const float*)in, (float*)out, len, in_stride, out_stride, type_len, inclusive);
}

bool wp_array_scan_double_device(
    uint64_t in, uint64_t out, int len, int in_stride, int out_stride, int type_len, bool inclusive
)
{
    apic_capture_array_scan_device(in, out, len, in_stride, out_stride, type_len, APIC_TYPE_FLOAT64, inclusive);
    return scan_device((const double*)in, (double*)out, len, in_stride, out_stride, type_len, inclusive);
}

int wp_cuda_driver_version()
{
    int version;
    if (check_cu(cuDriverGetVersion_f(&version)))
        return version;
    else
        return 0;
}

int wp_cuda_toolkit_version() { return CUDA_VERSION; }

int wp_nvrtc_version()
{
    int major = 0, minor = 0;
    nvrtcVersion(&major, &minor);
    return major * 1000 + minor * 10;
}

const char* wp_libmathdx_version()
{
#if WP_ENABLE_MATHDX
    static char version[64];
    snprintf(version, sizeof(version), "%d.%d.%d", LIBMATHDX_VER_MAJOR, LIBMATHDX_VER_MINOR, LIBMATHDX_VER_PATCH);
    return version;
#else
    return "";
#endif
}

bool wp_cuda_driver_is_initialized() { return is_cuda_driver_initialized(); }

int wp_nvrtc_supported_arch_count()
{
    int count;
    if (check_nvrtc(nvrtcGetNumSupportedArchs(&count)))
        return count;
    else
        return 0;
}

void wp_nvrtc_supported_archs(int* archs)
{
    if (archs) {
        check_nvrtc(nvrtcGetSupportedArchs(archs));
    }
}

int wp_cuda_device_get_count()
{
#if defined(__HIP_PLATFORM_AMD__)
    // The devices Warp shows: the cards, or their shares when WARP_HIP_DEVICES splits them (hip_split_devices).
    if (!g_devices.empty())
        return int(g_devices.size());
#endif
    int count = 0;
    check_cu(cuDeviceGetCount_f(&count));
    return count;
}

void* wp_cuda_device_get_primary_context(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size())) {
        DeviceInfo& device_info = g_devices[ordinal];

#if defined(__HIP_PLATFORM_AMD__)
        // A card's second or later share (hip_split_devices) has a context of its own on the card, a virtual one (cuda_util.cpp), as
        // its primary, so Warp keeps its streams, allocations and place on the current-context stack apart from the other shares.
        // The context names its share (g_hip_context_part), which get_context_info reads.
        if (!device_info.primary_context && device_info.hip_part > 1) {
            CUcontext ctx = NULL;
            if (check_cu(cuCtxCreate_f(&ctx, 0, device_info.device))) {
                CUcontext popped = NULL;
                check_cu(cuCtxPopCurrent_f(&popped));
                g_hip_context_part[ctx] = ordinal;
                device_info.primary_context = ctx;
            }
            return device_info.primary_context;
        }
#endif

        // acquire the primary context if we haven't already
        if (!device_info.primary_context)
            check_cu(cuDevicePrimaryCtxRetain_f(&device_info.primary_context, device_info.device));

        return device_info.primary_context;
    }

    return NULL;
}

const char* wp_cuda_device_get_name(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].name;
    return NULL;
}

const char* wp_cuda_device_get_arch(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].arch_str;
    return NULL;
}

int wp_cuda_device_get_sm_count(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].sm_count;
    return 0;
}

int wp_cuda_device_get_max_shared_memory(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].max_smem_bytes;
    return 0;
}

void wp_cuda_device_get_uuid(int ordinal, char uuid[16])
{
    memcpy(uuid, g_devices[ordinal].uuid.bytes, sizeof(char) * 16);
}

int wp_cuda_device_get_pci_domain_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_domain_id;
    return -1;
}

int wp_cuda_device_get_pci_bus_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_bus_id;
    return -1;
}

int wp_cuda_device_get_pci_device_id(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pci_device_id;
    return -1;
}

int wp_cuda_device_is_uva(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_uva;
    return 0;
}

int wp_cuda_device_get_pageable_memory_access(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pageable_memory_access;
    return 0;
}

int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].direct_managed_mem_access_from_host;
    return 0;
}

int wp_cuda_device_get_host_native_atomic_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].host_native_atomic_supported;
    return 0;
}

int wp_cuda_device_get_managed_memory_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].managed_memory;
    return 0;
}

int wp_cuda_device_get_concurrent_managed_access_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].concurrent_managed_access;
    return 0;
}

int wp_cuda_pointer_get_memory_kind(void* context, void* ptr)
{
    if (!ptr)
        return WP_MEMORY_KIND_UNKNOWN;

    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    // Managed memory made as mapped host memory counts as managed memory, even though the driver would call it host memory.
    if (hip_managed_contains(ptr))
        return WP_MEMORY_KIND_CUDA_MANAGED;
#if defined(_WIN32)
    // Default-allocator memory taken from the shareable pool counts as default-allocator memory, even though the driver would name
    // its pool, and so does memory opened from another process.
    {
        std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
        if (g_hip_ipc_allocs.count(ptr) || g_hip_ipc_imported.count(ptr))
            return WP_MEMORY_KIND_CUDA_DEVICE;
    }
#endif
    // A stream-ordered allocation is pool memory; HIP names no pool for a pointer (hip_pooled_remember), so the port's record says.
    if (hip_pooled_contains(ptr))
        return WP_MEMORY_KIND_CUDA_MEMPOOL;
#endif

    unsigned int is_managed = 0;
    CUresult managed_result
        = cuPointerGetAttribute_f(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, reinterpret_cast<CUdeviceptr>(ptr));
    if (managed_result != CUDA_SUCCESS)
        return WP_MEMORY_KIND_UNKNOWN;
    if (is_managed)
        return WP_MEMORY_KIND_CUDA_MANAGED;

    unsigned int memory_type = 0;
    CUresult memory_type_result
        = cuPointerGetAttribute_f(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(ptr));
    if (memory_type_result != CUDA_SUCCESS)
        return WP_MEMORY_KIND_UNKNOWN;

    if (memory_type == CU_MEMORYTYPE_HOST)
        return WP_MEMORY_KIND_PINNED;

    if (memory_type != CU_MEMORYTYPE_DEVICE)
        return WP_MEMORY_KIND_UNKNOWN;

    CUmemoryPool mempool = NULL;
    CUresult mempool_result
        = cuPointerGetAttribute_f(&mempool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, reinterpret_cast<CUdeviceptr>(ptr));
    if (mempool_result != CUDA_SUCCESS)
        return WP_MEMORY_KIND_CUDA_DEVICE;

    if (mempool)
        return WP_MEMORY_KIND_CUDA_MEMPOOL;

    return WP_MEMORY_KIND_CUDA_DEVICE;
}

int wp_cuda_device_is_mempool_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_mempool_supported;
    return 0;
}

int wp_cuda_device_is_ipc_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_ipc_supported;
    return 0;
}

int wp_cuda_device_set_mempool_release_threshold(int ordinal, uint64_t threshold)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(WP_DEVICE_POOL(&pool, ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    if (!check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold))) {
        fprintf(stderr, "Warp error: Failed to set memory pool attribute on device %d\n", ordinal);
        return 0;
    }

    return 1;  // success
}

uint64_t wp_cuda_device_get_mempool_release_threshold(int ordinal)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(WP_DEVICE_POOL(&pool, ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t threshold = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold))) {
        fprintf(stderr, "Warp error: Failed to get memory pool release threshold on device %d\n", ordinal);
        return 0;
    }

    return threshold;
}

uint64_t wp_cuda_device_get_mempool_used_mem_current(int ordinal)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(WP_DEVICE_POOL(&pool, ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t mem_used = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &mem_used))) {
        fprintf(
            stderr, "Warp error: Failed to get amount of currently used memory from the memory pool on device %d\n",
            ordinal
        );
        return 0;
    }

    return mem_used;
}

uint64_t wp_cuda_device_get_mempool_used_mem_high(int ordinal)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

    if (!g_devices[ordinal].is_mempool_supported)
        return 0;

    cudaMemPool_t pool;
    if (!check_cuda(WP_DEVICE_POOL(&pool, ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool on device %d\n", ordinal);
        return 0;
    }

    uint64_t mem_high_water_mark = 0;
    if (!check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &mem_high_water_mark))) {
        fprintf(
            stderr, "Warp error: Failed to get memory usage high water mark from the memory pool on device %d\n",
            ordinal
        );
        return 0;
    }

    return mem_high_water_mark;
}

uint64_t wp_cuda_device_get_graph_mem_current(int ordinal)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__)
    // HIP on Windows has no graph memory nodes, so memory a capture allocates is a plain allocation recorded in g_graph_allocs and
    // freed when its graph is destroyed and no reference remains (see the stand-ins above). The device attribute reads 0 while
    // graphs hold real memory, so a test waiting for a destroyed graph's memory to be released waited forever
    // (test_capture_release, multi_device_graph_release). Here, graph memory is what that record holds for the device.
    uint64_t held = 0;
    for (const auto& kv : g_graph_allocs) {
        if (wp_cuda_context_get_device_ordinal(kv.second.context) == ordinal)
            held += kv.second.size;
    }
    return held;
#else
    uint64_t mem_used = 0;
    if (!check_cuda(cudaDeviceGetGraphMemAttribute(ordinal, cudaGraphMemAttrUsedMemCurrent, &mem_used))) {
        fprintf(stderr, "Warp error: Failed to get graph memory usage on device %d\n", ordinal);
        return 0;
    }

    return mem_used;
#endif
}

void wp_cuda_device_graph_mem_trim(int ordinal)
{
    if (ordinal < 0 || ordinal >= int(g_devices.size())) {
        fprintf(stderr, "Invalid device ordinal %d\n", ordinal);
        return;
    }

    if (!check_cuda(cudaDeviceGraphMemTrim(ordinal))) {
        fprintf(stderr, "Warp error: Failed to trim graph memory on device %d\n", ordinal);
    }
}

void wp_cuda_device_get_memory_info(int ordinal, size_t* free_mem, size_t* total_mem)
{
    // use temporary storage if user didn't specify pointers
    size_t tmp_free_mem, tmp_total_mem;

    if (free_mem)
        *free_mem = 0;
    else
        free_mem = &tmp_free_mem;

    if (total_mem)
        *total_mem = 0;
    else
        total_mem = &tmp_total_mem;

    if (ordinal >= 0 && ordinal < int(g_devices.size())) {
#if defined(__HIP_PLATFORM_AMD__)
        // A share reports its own budget, and what its pool holds against it, bounded by what its card still has free.
        DeviceInfo& d = g_devices[ordinal];
        if (d.hip_parts > 1 && d.hip_memory_budget) {
            size_t card_free = 0;
            size_t card_total = 0;
            int prev = 0;
            ignore_cuda_error(hipGetDevice(&prev));
            if (hipSetDevice(d.device) == hipSuccess)
                ignore_cuda_error(hipMemGetInfo(&card_free, &card_total));
            else
                ignore_cuda_error(hipGetLastError());
            ignore_cuda_error(hipSetDevice(prev));
            uint64_t used = 0;
            if (hipMemPool_t pool = hip_share_pool(d))
                if (hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used) != hipSuccess)
                    ignore_cuda_error(hipGetLastError());
            const size_t room = used < d.hip_memory_budget ? d.hip_memory_budget - size_t(used) : 0;
            *total_mem = d.hip_memory_budget;
            *free_mem = room < card_free ? room : card_free;
            return;
        }
#endif
        if (g_devices[ordinal].primary_context) {
            ContextGuard guard(g_devices[ordinal].primary_context, true);
            check_cu(cuMemGetInfo_f(free_mem, total_mem));
        } else {
            // if we haven't acquired the primary context yet, acquire it temporarily
            CUcontext primary_context = NULL;
            check_cu(cuDevicePrimaryCtxRetain_f(&primary_context, g_devices[ordinal].device));
            {
                ContextGuard guard(primary_context, true);
                check_cu(cuMemGetInfo_f(free_mem, total_mem));
            }
            check_cu(cuDevicePrimaryCtxRelease_f(g_devices[ordinal].device));
        }
    }
}


void* wp_cuda_context_get_current() { return get_current_context(); }

void wp_cuda_context_set_current(void* context)
{
    CUcontext ctx = static_cast<CUcontext>(context);
    CUcontext prev_ctx = NULL;
    check_cu(cuCtxGetCurrent_f(&prev_ctx));
    if (ctx != prev_ctx) {
        check_cu(cuCtxSetCurrent_f(ctx));
    }
}

void wp_cuda_context_push_current(void* context) { check_cu(cuCtxPushCurrent_f(static_cast<CUcontext>(context))); }

void wp_cuda_context_pop_current()
{
    CUcontext context;
    check_cu(cuCtxPopCurrent_f(&context));
}

void* wp_cuda_context_create(int device_ordinal)
{
    CUcontext ctx = NULL;
    CUdevice device;
    if (check_cu(cuDeviceGet_f(&device, device_ordinal)))
        check_cu(cuCtxCreate_f(&ctx, 0, device));
#if defined(__HIP_PLATFORM_AMD__)
    /* On HIP, the context create can hand back the device's primary context (HIP's own hipCtxCreate does this, returning the same
       handle as cuda:0's). Returning it as a new context would map it as a second device and rename cuda:0, and its caller would then
       destroy cuda:0's own context under every later call ("context is destroyed", test_capture_release). So no context is
       returned, and the caller is told why. */
    if (ctx && wp_cuda_context_is_primary(ctx)) {
        // The create pushed it as current, so pop it to leave the thread's stack as it was.
        CUcontext popped = NULL;
        check_cu(cuCtxPopCurrent_f(&popped));
        wp::set_error_string(
            "Warp HIP error: HIP has one context per device - a new context on device %d is its primary context, which is not "
            "created here (use the device itself)",
            device_ordinal
        );
        return NULL;
    }
#endif
    return ctx;
}

void wp_cuda_context_destroy(void* context)
{
    if (context) {
        CUcontext ctx = static_cast<CUcontext>(context);

        // ensure this is not the current context
        if (ctx == wp_cuda_context_get_current())
            wp_cuda_context_set_current(NULL);

        // release the radix sort side stream associated with this context
        radix_sort_context_release(context);

        // release the cached info about this context
        ContextInfo* info = get_context_info(ctx);
        void* device_context = NULL;
        if (info) {
            if (info->device_info)
                device_context = wp_cuda_device_get_primary_context(info->device_info->ordinal);

            if (info->stream)
                check_cu(cuStreamDestroy_f(info->stream));

            if (info->conditional_module)
                check_cu(cuModuleUnload_f(info->conditional_module));

            g_contexts.erase(ctx);
        }

        discard_deferred_releases(context, device_context);

        check_cu(cuCtxDestroy_f(ctx));
    }
}

bool wp_cuda_context_synchronize(void* context)
{
    ContextGuard guard(context);

    // Report failures from earlier asynchronous work before running deferred actions,
    // which would replace the error string. Pending actions run after the next
    // successful synchronization or graph capture.
    if (!check_cu(cuCtxSynchronize_f())) {
#if defined(__HIP_PLATFORM_AMD__)
        hip_print_flush();  // Whatever a failed kernel printed before it failed, such as an assert's message, is still written.
#endif
        return false;
    }

    if (!context)
        context = get_current_context();

    if (run_deferred_actions(context) > 0) {
        // ensure deferred asynchronous operations complete
        const bool ok = check_cu(cuCtxSynchronize_f());
#if defined(__HIP_PLATFORM_AMD__)
        if (const int trapped = hip_print_flush())  // A kernel trapped (hip_print.h trap): its error, as CUDA reports it.
            return check_cu(static_cast<CUresult>(trapped));
#endif
        return ok;
    }

    // check_cuda(cudaDeviceGraphMemTrim(wp_cuda_context_get_device_ordinal(context)));

#if defined(__HIP_PLATFORM_AMD__)
    const int trapped = hip_print_flush();
    hip_priority_dump();
    if (trapped)  // A kernel trapped (hip_print.h trap): its error, as CUDA reports it.
        return check_cu(static_cast<CUresult>(trapped));
#endif
    return true;
}

bool wp_cuda_profiler_start(void* context)
{
    ContextGuard guard(context, true);
    return check_cu(cuProfilerStart_f());
}

bool wp_cuda_profiler_stop(void* context)
{
    ContextGuard guard(context, true);
    return check_cu(cuProfilerStop_f());
}

uint64_t wp_cuda_context_check(void* context)
{
    ContextGuard guard(context);

    // check errors before syncing
    cudaError_t e = cudaGetLastError();
    check_cuda(e);

    CUcontext current_context = get_current_context();

    // Device-wide synchronization is illegal while a Warp-known stream capture
    // is active in this context, even if the current stream itself is not capturing.
    if (!is_context_capturing(current_context)) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        check_cuda(cudaStreamIsCapturing(get_current_stream(), &status));

        // synchronize if the stream is not capturing
        if (status == cudaStreamCaptureStatusNone) {
            check_cuda(cudaDeviceSynchronize());
            e = cudaGetLastError();
#if defined(__HIP_PLATFORM_AMD__)
            if (const int trapped = hip_print_flush()) {  // A kernel trapped (hip_print.h trap): its error, as CUDA reports it.
                e = static_cast<cudaError_t>(trapped);
                check_cuda(e);
            }
#endif
        }
    }

    return static_cast<uint64_t>(e);
}


int wp_cuda_context_get_device_ordinal(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    return info && info->device_info ? info->device_info->ordinal : -1;
}

int wp_cuda_context_is_primary(void* context)
{
    CUcontext ctx = static_cast<CUcontext>(context);
    ContextInfo* context_info = get_context_info(ctx);
    if (!context_info) {
        fprintf(stderr, "Warp error: Failed to get context info\n");
        return 0;
    }

    // if the device primary context is known, check if it matches the given context
    DeviceInfo* device_info = context_info->device_info;
    if (device_info->primary_context)
        return int(ctx == device_info->primary_context);

    // there is no CUDA API to check if a context is primary, but we can temporarily
    // acquire the device's primary context to check the pointer
    CUcontext primary_ctx;
    if (check_cu(cuDevicePrimaryCtxRetain_f(&primary_ctx, device_info->device))) {
        check_cu(cuDevicePrimaryCtxRelease_f(device_info->device));
        return int(ctx == primary_ctx);
    }

    return 0;
}

void* wp_cuda_context_get_stream(void* context)
{
    ContextInfo* info = get_context_info(static_cast<CUcontext>(context));
    if (info) {
        return info->stream;
    }
    return NULL;
}

void wp_cuda_context_set_stream(void* context, void* stream, int sync)
{
    ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
    if (context_info) {
        CUstream new_stream = static_cast<CUstream>(stream);

        // check whether we should sync with the previous stream on this device
        if (sync) {
            CUstream old_stream = context_info->stream;
            StreamInfo* old_stream_info = get_stream_info(old_stream);
            if (old_stream_info) {
                CUevent cached_event = old_stream_info->cached_event;
                check_cu(cuEventRecord_f(cached_event, old_stream));
#if defined(__HIP_PLATFORM_AMD__)
                // Submit the event's batch before the new stream waits on it, as wp_cuda_event_record does. A ScopedStream's sync_enter
                // and sync_exit come through here, and test_tape_copy_adjoint_stream's copy adjoint ran alongside the backward kernel it
                // was supposed to follow.
                hip_submit_after_event(old_stream);
#endif
                check_cu(cuStreamWaitEvent_f(new_stream, cached_event, CU_EVENT_WAIT_DEFAULT));
            }
        }

        context_info->stream = new_stream;
    }
}

int wp_cuda_is_peer_access_supported(int target_ordinal, int peer_ordinal)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid target device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;

#if defined(__HIP_PLATFORM_AMD__)
    // Two shares of one card (hip_split_devices) reach each other's memory: it is one card's. Otherwise the cards are asked.
    if (g_devices[target_ordinal].device == g_devices[peer_ordinal].device)
        return 1;
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, g_devices[peer_ordinal].device, g_devices[target_ordinal].device));
    return can_access;
#else
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_ordinal, target_ordinal));

    return can_access;
#endif
}

int wp_cuda_is_peer_access_enabled(void* target_context, void* peer_context)
{
    if (!target_context || !peer_context) {
        fprintf(stderr, "Warp error: invalid CUDA context\n");
        return 0;
    }

    if (target_context == peer_context)
        return 1;

    int target_ordinal = wp_cuda_context_get_device_ordinal(target_context);
    int peer_ordinal = wp_cuda_context_get_device_ordinal(peer_context);

#if defined(__HIP_PLATFORM_AMD__)
    if (hip_same_card(target_ordinal, peer_ordinal))
        return g_hip_peer_on.count({ target_ordinal, peer_ordinal }) ? 1 : 0;
    if (target_ordinal >= 0 && peer_ordinal >= 0) {
        target_ordinal = g_devices[target_ordinal].device;
        peer_ordinal = g_devices[peer_ordinal].device;
    }
#endif

    // check if peer access is supported
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_ordinal, target_ordinal));
    if (!can_access)
        return 0;

    // There is no CUDA API to query if peer access is enabled, but we can try to enable it and check the result.

    ContextGuard guard(peer_context, true);

    CUcontext target_ctx = static_cast<CUcontext>(target_context);

    CUresult result = cuCtxEnablePeerAccess_f(target_ctx, 0);
    if (result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        return 1;
    } else if (result == CUDA_SUCCESS) {
        // undo enablement
        check_cu(cuCtxDisablePeerAccess_f(target_ctx));
        return 0;
    } else {
        // report error
        check_cu(result);
        return 0;
    }
}

int wp_cuda_set_peer_access_enabled(void* target_context, void* peer_context, int enable)
{
    if (!target_context || !peer_context) {
        fprintf(stderr, "Warp error: invalid CUDA context\n");
        return 0;
    }

    if (target_context == peer_context)
        return 1;  // no-op

    int target_ordinal = wp_cuda_context_get_device_ordinal(target_context);
    int peer_ordinal = wp_cuda_context_get_device_ordinal(peer_context);

#if defined(__HIP_PLATFORM_AMD__)
    if (hip_same_card(target_ordinal, peer_ordinal)) {
        if (enable)
            g_hip_peer_on[{ target_ordinal, peer_ordinal }] = 1;
        else
            g_hip_peer_on.erase({ target_ordinal, peer_ordinal });
        return 1;
    }
    const int target_card = target_ordinal >= 0 ? g_devices[target_ordinal].device : target_ordinal;
    const int peer_card = peer_ordinal >= 0 ? g_devices[peer_ordinal].device : peer_ordinal;
#else
    const int target_card = target_ordinal;
    const int peer_card = peer_ordinal;
#endif

    // check if peer access is supported
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, peer_card, target_card));
    if (!can_access) {
        // failure if enabling, success if disabling
        if (enable) {
            fprintf(stderr, "Warp error: device %d cannot access device %d\n", peer_ordinal, target_ordinal);
            return 0;
        } else
            return 1;
    }

    ContextGuard guard(peer_context, true);

    CUcontext target_ctx = static_cast<CUcontext>(target_context);

    if (enable) {
        CUresult status = cuCtxEnablePeerAccess_f(target_ctx, 0);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            check_cu(status);
            fprintf(
                stderr, "Warp error: failed to enable peer access from device %d to device %d\n", peer_ordinal,
                target_ordinal
            );
            return 0;
        }
    } else {
        CUresult status = cuCtxDisablePeerAccess_f(target_ctx);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_NOT_ENABLED) {
            check_cu(status);
            fprintf(
                stderr, "Warp error: failed to disable peer access from device %d to device %d\n", peer_ordinal,
                target_ordinal
            );
            return 0;
        }
    }

    return 1;  // success
}

int wp_cuda_is_mempool_access_enabled(int target_ordinal, int peer_ordinal)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;

#if defined(__HIP_PLATFORM_AMD__)
    // (two shares of one card: the state Warp set, g_hip_pool_access_on; otherwise the cards' pools are asked by the cards' numbers)
    if (hip_same_card(target_ordinal, peer_ordinal))
        return g_hip_pool_access_on.count({ target_ordinal, peer_ordinal }) ? 1 : 0;
    peer_ordinal = g_devices[peer_ordinal].device;
    target_ordinal = g_devices[target_ordinal].device;
#endif

    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, target_ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool of device %d\n", target_ordinal);
        return 0;
    }

    cudaMemAccessFlags flags = cudaMemAccessFlagsProtNone;
    cudaMemLocation location;
    location.id = peer_ordinal;
    location.type = cudaMemLocationTypeDevice;
    if (check_cuda(cudaMemPoolGetAccess(&flags, pool, &location)))
        return int(flags != cudaMemAccessFlagsProtNone);

    return 0;
}

int wp_cuda_set_mempool_access_enabled(int target_ordinal, int peer_ordinal, int enable)
{
    int num_devices = int(g_devices.size());

    if (target_ordinal < 0 || target_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid device ordinal %d\n", target_ordinal);
        return 0;
    }

    if (peer_ordinal < 0 || peer_ordinal >= num_devices) {
        fprintf(stderr, "Warp error: Invalid peer device ordinal %d\n", peer_ordinal);
        return 0;
    }

    if (target_ordinal == peer_ordinal)
        return 1;  // no-op

#if defined(__HIP_PLATFORM_AMD__)
    if (hip_same_card(target_ordinal, peer_ordinal)) {
        if (enable)
            g_hip_pool_access_on[{ target_ordinal, peer_ordinal }] = 1;
        else
            g_hip_pool_access_on.erase({ target_ordinal, peer_ordinal });
        return 1;
    }
    peer_ordinal = g_devices[peer_ordinal].device;
    target_ordinal = g_devices[target_ordinal].device;
#endif

    // get the memory pool
    cudaMemPool_t pool;
    if (!check_cuda(cudaDeviceGetDefaultMemPool(&pool, target_ordinal))) {
        fprintf(stderr, "Warp error: Failed to get memory pool of device %d\n", target_ordinal);
        return 0;
    }

    cudaMemAccessDesc desc;
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id = peer_ordinal;

    // only cudaMemAccessFlagsProtReadWrite and cudaMemAccessFlagsProtNone are supported
    if (enable)
        desc.flags = cudaMemAccessFlagsProtReadWrite;
    else
        desc.flags = cudaMemAccessFlagsProtNone;

    if (!check_cuda(cudaMemPoolSetAccess(pool, &desc, 1))) {
        fprintf(
            stderr, "Warp error: Failed to set mempool access from device %d to device %d\n", peer_ordinal,
            target_ordinal
        );
        return 0;
    }

    return 1;  // success
}

void wp_cuda_ipc_get_mem_handle(void* ptr, char* out_buffer)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    // Shares memory through NT handles from the shareable pool (hip_ipc_export). The handle is all zeros when the memory is not
    // shareable.
    if (!hip_ipc_export(ptr, out_buffer))
        fprintf(stderr, "%s\n", wp::get_error_string());
    return;
#endif
    CUipcMemHandle memHandle;
    check_cu(cuIpcGetMemHandle_f(&memHandle, (CUdeviceptr)ptr));
    memcpy(out_buffer, memHandle.reserved, CU_IPC_HANDLE_SIZE);
}

void* wp_cuda_ipc_open_mem_handle(void* context, char* handle)
{
    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    if (hip_ipc_is_ours(handle)) {
        void* ptr = hip_ipc_import(handle);
        if (!ptr)
            fprintf(stderr, "%s\n", wp::get_error_string());
        return ptr;
    }
#endif

    CUipcMemHandle memHandle;
    memcpy(memHandle.reserved, handle, CU_IPC_HANDLE_SIZE);

    CUdeviceptr device_ptr;

    // Strangely, the CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag is required
    if check_cu (cuIpcOpenMemHandle_f(&device_ptr, memHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS))
        return (void*)device_ptr;
    else
        return NULL;
}

void wp_cuda_ipc_close_mem_handle(void* ptr)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    bool imported = false;
    {
        std::lock_guard<std::mutex> lock(g_hip_ipc_mutex);
        imported = g_hip_ipc_imported.erase(ptr) > 0;
    }
    if (imported) {
        // Memory imported from another process's pool is released by a free in this process.
        check_cuda(hipFree(ptr));
        return;
    }
#endif
    check_cu(cuIpcCloseMemHandle_f((CUdeviceptr)ptr));
}

void wp_cuda_ipc_get_event_handle(void* context, void* event, char* out_buffer)
{
    ContextGuard guard(context);

    // Zero the handle first. A refused handle, such as one for an event created without interprocess=True, then stays all zeros,
    // which is how Warp's Python side knows to warn. HIP leaves the struct as it found it, and an uninitialized struct read as a
    // valid handle.
    CUipcEventHandle eventHandle = {};
    if (!check_cu(cuIpcGetEventHandle_f(&eventHandle, static_cast<CUevent>(event))))
        memset(&eventHandle, 0, sizeof(eventHandle));
#if defined(__HIP_PLATFORM_AMD__)
    else
        g_hip_shared_work.store(true, std::memory_order_relaxed);  // From now on, every command is submitted as it is issued.
#endif
    memcpy(out_buffer, eventHandle.reserved, CU_IPC_HANDLE_SIZE);
}

void* wp_cuda_ipc_open_event_handle(void* context, char* handle)
{
    ContextGuard guard(context);

    CUipcEventHandle eventHandle;
    memcpy(eventHandle.reserved, handle, CU_IPC_HANDLE_SIZE);

    CUevent event;

    if (check_cu(cuIpcOpenEventHandle_f(&event, eventHandle))) {
#if defined(__HIP_PLATFORM_AMD__)
        g_hip_shared_work.store(true, std::memory_order_relaxed);
#endif
        return event;
    } else
        return NULL;
}

void* wp_cuda_stream_create(void* context, int priority)
{
    ContextGuard guard(context, true);

#if defined(__HIP_PLATFORM_AMD__)
    // HIP accepts a stream's priority, but this device never acts on it (see hip_priority_before), so priorities are enforced in
    // software. They are clamped to CUDA's two levels, 0 (the default) and -1 (high), the way a CUDA device with that range clamps
    // them. HIP's own lowest level (1) is never used.
    priority = priority < 0 ? -1 : 0;
#endif

    CUstream stream;
    // (A logical device's streams run on the whole card: HIP on Windows holds a stream to no part of it. Timed with a kernel that
    // fills the card, a stream masked to half, a quarter or one of the 8060S's 40 compute units, in any form of mask
    // hipExtStreamCreateWithCUMask takes, ran as fast as an unmasked one. So the shares divide memory and contexts, not compute.)
    if (check_cu(cuStreamCreateWithPriority_f(&stream, CU_STREAM_DEFAULT, priority))) {
        wp_cuda_stream_register(WP_CURRENT_CONTEXT, stream);
#if defined(__HIP_PLATFORM_AMD__)
        hip_priority_stream_made(stream, priority);
#endif
        return stream;
    } else
        return NULL;
}

void wp_cuda_stream_destroy(void* context, void* stream)
{
    if (!stream)
        return;

    wp_cuda_stream_unregister(context, stream);
#if defined(__HIP_PLATFORM_AMD__)
    hip_priority_stream_gone(stream);
#endif

    check_cu(cuStreamDestroy_f(static_cast<CUstream>(stream)));
}

int wp_cuda_stream_query(void* stream)
{
    CUresult res = cuStreamQuery_f(static_cast<CUstream>(stream));

    if ((res != CUDA_SUCCESS) && (res != CUDA_ERROR_NOT_READY)) {
        // Abnormal, print out error
        check_cu(res);
    }

    return res;
}

void wp_cuda_stream_register(void* context, void* stream)
{
    if (!stream)
        return;

    ContextGuard guard(context);

    // populate stream info
    StreamInfo& stream_info = g_streams[static_cast<CUstream>(stream)];
    check_cu(cuEventCreate_f(&stream_info.cached_event, CU_EVENT_DISABLE_TIMING));
}

void wp_cuda_stream_unregister(void* context, void* stream)
{
    if (!stream)
        return;

#if defined(__HIP_PLATFORM_AMD__)
    hip_forget_stream(stream);
#endif

    CUstream cuda_stream = static_cast<CUstream>(stream);

    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (stream_info) {
        // release stream info
        check_cu(cuEventDestroy_f(stream_info->cached_event));
        g_streams.erase(cuda_stream);
    }

    // make sure we don't leave dangling references to this stream
    ContextInfo* context_info = get_context_info(context);
    if (context_info) {
        if (cuda_stream == context_info->stream)
            context_info->stream = NULL;
    }

    // release temporary radix sort scratch space associated with this stream
    radix_sort_stream_release(context, stream);
}

void* wp_cuda_stream_get_current() { return get_current_stream(); }

bool wp_cuda_stream_synchronize(void* stream)
{
    const bool ok = check_cu(cuStreamSynchronize_f(static_cast<CUstream>(stream)));
#if defined(__HIP_PLATFORM_AMD__)
    if (const int trapped = hip_print_flush())  // A kernel trapped (hip_print.h trap): its error, as CUDA reports it.
        return check_cu(static_cast<CUresult>(trapped));
    static const bool alloc_debug = getenv("WARP_HIP_ALLOC_DEBUG") != NULL;
    if (alloc_debug) {
        cudaStreamCaptureStatus cur = cudaStreamCaptureStatusNone;
        CUstream current = get_current_stream(NULL);
        ignore_cuda_error(cudaStreamIsCapturing(current, &cur));
        fprintf(stderr, "[hip-alloc] stream %p synchronized (%d); the current stream %p status %d\n", stream, int(ok), (void*)current,
                int(cur));
    }
#endif
    return ok;
}

void wp_cuda_stream_wait_event(void* stream, void* event, bool external)
{
    WP_HIP_ISSUE_SYNC(stream);
    // the external flag can only be used during graph capture
    if (external && !g_captures.empty() && wp_cuda_stream_is_capturing(stream)) {
        // wait for an external event during graph capture
        check_cu(
            cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), CU_EVENT_WAIT_EXTERNAL)
        );
    } else {
        check_cu(
            cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), CU_EVENT_WAIT_DEFAULT)
        );
    }
}

void wp_cuda_stream_wait_stream(void* stream, void* other_stream, void* event, bool external)
{
    // Record the event on the other stream, then wait on it on this one.
    WP_HIP_ISSUE_SYNC(other_stream);
    WP_HIP_ISSUE_SYNC(stream);
    unsigned record_flags = CU_EVENT_RECORD_DEFAULT;
    unsigned wait_flags = CU_EVENT_WAIT_DEFAULT;

    // the external flag can only be used during graph capture
    if (external && !g_captures.empty()) {
        if (wp_cuda_stream_is_capturing(other_stream))
            record_flags = CU_EVENT_RECORD_EXTERNAL;
        if (wp_cuda_stream_is_capturing(stream))
            wait_flags = CU_EVENT_WAIT_EXTERNAL;
    }

    check_cu(cuEventRecordWithFlags_f(static_cast<CUevent>(event), static_cast<CUstream>(other_stream), record_flags));
#if defined(__HIP_PLATFORM_AMD__)
    // Submit the event's batch before another stream waits on it, as wp_cuda_event_record does. When it was left in an unsubmitted
    // batch, the wait did not hold. test_tape_copy_adjoint_stream's copy adjoint (a ScopedStream's sync_enter) ran alongside the
    // backward kernel it was supposed to follow, and in 30 of 40 runs both read-modify-wrote the same gradient.
    if (record_flags == CU_EVENT_RECORD_DEFAULT)
        hip_submit_after_event(other_stream);
#endif
    check_cu(cuStreamWaitEvent_f(static_cast<CUstream>(stream), static_cast<CUevent>(event), wait_flags));
}

int wp_cuda_stream_is_capturing(void* stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status));

    return int(status != cudaStreamCaptureStatusNone);
}

int wp_cuda_stream_is_blocking(void* stream)
{
#if defined(__HIP_PLATFORM_AMD__)
    // hipStreamGetFlags rejects the null stream with "invalid argument" (printed by test_stream_is_blocking), while
    // cudaStreamGetFlags reports its flags. The null stream is blocking by definition.
    if (stream == nullptr)
        return 1;
#endif
    unsigned int flags = 0;
    check_cuda(cudaStreamGetFlags(static_cast<cudaStream_t>(stream), &flags));
    return !(flags & cudaStreamNonBlocking);
}

int wp_cuda_thread_exchange_capture_mode(int mode)
{
    // Swap this thread's stream capture mode and return the previous mode.
    // Passing cudaStreamCaptureModeRelaxed allows otherwise-forbidden
    // operations (e.g. legacy allocations on non-capturing streams) while a
    // thread-local capture is active on another stream. Callers must restore
    // the returned mode afterwards.
    cudaStreamCaptureMode capture_mode = static_cast<cudaStreamCaptureMode>(mode);
    if (!check_cuda(cudaThreadExchangeStreamCaptureMode(&capture_mode)))
        return -1;

    return int(capture_mode);
}

uint64_t wp_cuda_stream_get_capture_id(void* stream) { return get_capture_id(static_cast<CUstream>(stream)); }

int wp_cuda_stream_get_priority(void* stream)
{
    int priority = 0;
    check_cuda(cuStreamGetPriority_f(static_cast<CUstream>(stream), &priority));

    return priority;
}

void* wp_cuda_event_create(void* context, unsigned flags)
{
    ContextGuard guard(context, true);

    CUevent event;
    if (check_cu(cuEventCreate_f(&event, flags)))
        return event;
    else
        return NULL;
}

void wp_cuda_event_destroy(void* event) { check_cu(cuEventDestroy_f(static_cast<CUevent>(event))); }

int wp_cuda_event_query(void* event)
{
    CUresult res = cuEventQuery_f(static_cast<CUevent>(event));

    if ((res != CUDA_SUCCESS) && (res != CUDA_ERROR_NOT_READY)) {
        // Abnormal, print out error
        check_cu(res);
    }

    return res;
}

void wp_cuda_event_record(void* event, void* stream, bool external)
{
    WP_HIP_ISSUE_SYNC(stream);
    // the external flag can only be used during graph capture
    if (external && !g_captures.empty() && wp_cuda_stream_is_capturing(stream)) {
        // record external event during graph capture (e.g., for timing or when explicitly specified by the user)
        check_cu(cuEventRecordWithFlags_f(
            static_cast<CUevent>(event), static_cast<CUstream>(stream), CU_EVENT_RECORD_EXTERNAL
        ));
    } else {
        check_cu(cuEventRecord_f(static_cast<CUevent>(event), static_cast<CUstream>(stream)));
    }
#if defined(__HIP_PLATFORM_AMD__)
    // Submit right away. HIP on Windows timestamps an event when its batch runs, so an event left in an unsubmitted batch got the
    // time of the next synchronization, long after the work it was meant to follow had finished. test_stream_priority_timings read
    // both streams' end events at that synchronization and concluded the priorities were equal, while the GPU's own trace showed
    // the high stream's copy running alone and the low stream's copy after it. CUDA timestamps an event when the stream reaches it.
    // A query alone does not submit a batch that ends in an event (hip_submit_after_event explains why).
    hip_submit_after_event(stream);
#endif
}

bool wp_cuda_event_synchronize(void* event)
{
    const bool ok = check_cu(cuEventSynchronize_f(static_cast<CUevent>(event)));
#if defined(__HIP_PLATFORM_AMD__)
    if (const int trapped = hip_print_flush())  // A kernel trapped (hip_print.h trap): its error, as CUDA reports it.
        return check_cu(static_cast<CUresult>(trapped));
#endif
    return ok;
}

float wp_cuda_event_elapsed_time(void* start_event, void* end_event)
{
    float elapsed = 0.0f;
    cudaEvent_t start = static_cast<cudaEvent_t>(start_event);
    cudaEvent_t end = static_cast<cudaEvent_t>(end_event);
    check_cuda(cudaEventElapsedTime(&elapsed, start, end));
    return elapsed;
}

bool wp_cuda_graph_begin_capture(void* context, void* stream, int external, int mode)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (!stream_info) {
        wp::set_error_string("Warp error: unknown stream");
        return false;
    }

#if defined(__HIP_PLATFORM_AMD__)
    // Submit the stream's earlier work before its capture opens, so it runs as it would on CUDA (hip_issue).
    if (!external && g_captures.empty())
        hip_submit(cuda_stream);
#endif

    cudaStreamCaptureMode capture_mode;
    switch (mode) {
    case WP_CUDA_GRAPH_CAPTURE_MODE_GLOBAL:
        capture_mode = cudaStreamCaptureModeGlobal;
        break;
    case WP_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL:
        capture_mode = cudaStreamCaptureModeThreadLocal;
        break;
    case WP_CUDA_GRAPH_CAPTURE_MODE_RELAXED:
        capture_mode = cudaStreamCaptureModeRelaxed;
        break;
    default:
        wp::set_error_string("Warp error: invalid capture mode");
        return false;
    }

    if (external) {
        // if it's an external capture, make sure it's already active so we can get the capture id
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (!check_cuda(cudaStreamIsCapturing(cuda_stream, &status)))
            return false;
        if (status != cudaStreamCaptureStatusActive) {
            wp::set_error_string("Warp error: stream is not capturing");
            return false;
        }
    } else {
        // Lazily registering a context warms up CUDA mempool state with default-stream
        // cudaMallocAsync/cudaFreeAsync calls; do that before capture starts.
        if (!get_context_info(static_cast<CUcontext>(context))) {
            wp::set_error_string("Warp error: failed to initialize CUDA context info");
            return false;
        }
        if (!check_cuda(cudaStreamBeginCapture(cuda_stream, capture_mode)))
            return false;
    }

    uint64_t capture_id = get_capture_id(cuda_stream);

    CaptureInfo* capture = new CaptureInfo();
    capture->stream = cuda_stream;
    capture->context = context ? static_cast<CUcontext>(context) : get_current_context();
    capture->id = capture_id;
    capture->external = bool(external);
    capture->mode = capture_mode;

    // update stream info
    stream_info->capture = capture;

    // add to known captures
    g_captures[capture_id] = capture;

    return true;
}

bool wp_cuda_graph_end_capture(void* context, void* stream, void** graph_ret)
{
    ContextGuard guard(context);

    // check if this is a known stream
    CUstream cuda_stream = static_cast<CUstream>(stream);
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (!stream_info) {
        wp::set_error_string("Warp error: unknown capture stream");
        return false;
    }

    // check if this stream was used to start a capture
    CaptureInfo* capture = stream_info->capture;
    if (!capture) {
        wp::set_error_string("Warp error: stream has no capture started");
        return false;
    }

    // clean up any capture-specific radix sort buffers
    END_TRACE("end: radix sort");
    radix_sort_end_capture(capture->id);
    END_TRACE("end: capture info");

    // get capture info
    bool external = capture->external;
    uint64_t capture_id = capture->id;
    std::vector<FreeInfo> tmp_allocs = capture->tmp_allocs;

    // clear capture info
    stream_info->capture = NULL;
    g_captures.erase(capture_id);
    delete capture;

    // a lambda to clean up on exit in case of error
    auto clean_up = [cuda_stream, capture_id, external]() {
        // unreference outstanding graph allocs so that they will be released with the user reference
        for (auto it = g_graph_allocs.begin(); it != g_graph_allocs.end(); /*noop*/) {
            GraphAllocInfo& alloc_info = it->second;
            if (alloc_info.capture_id == capture_id) {
                if (!alloc_info.ref_exists) {
                    // The user reference was already dropped (e.g., freed during conditional
                    // body capture). No graph instance will exist to own the allocation, so
                    // free it once graph captures complete. On HIP on Windows, a static allocation is freed with cudaFree, the way it was made.
                    defer_release(alloc_info.is_static ? ReleaseKind::DeviceFree : ReleaseKind::DeviceFreeAsyncUnmapped,
                                  alloc_info.context, (uint64_t)it->first);
                    it = g_graph_allocs.erase(it);
                    continue;
                }
                alloc_info.graph_destroyed = true;
            }
            ++it;
        }

        // make sure we terminate the capture
        if (!external) {
            cudaGraph_t graph = NULL;
            ignore_cuda_error(cudaStreamEndCapture(cuda_stream, &graph));
            ignore_cuda_error(cudaGetLastError());
        }
    };

    // get captured graph without ending the capture in case it is external
    cudaGraph_t graph = get_capture_graph(cuda_stream);
    if (!graph) {
        clean_up();
        return false;
    }

    // ensure that all forked streams are joined to the main capture stream by manually
    // adding outstanding capture dependencies gathered from the graph leaf nodes
    std::vector<cudaGraphNode_t> stream_dependencies;
    std::vector<cudaGraphNode_t> leaf_nodes;
    if (get_capture_dependencies(cuda_stream, stream_dependencies) && get_graph_leaf_nodes(graph, leaf_nodes)) {
        // compute set difference to get unjoined dependencies
        std::vector<cudaGraphNode_t> unjoined_dependencies;
        std::sort(stream_dependencies.begin(), stream_dependencies.end());
        std::sort(leaf_nodes.begin(), leaf_nodes.end());
        std::set_difference(
            leaf_nodes.begin(), leaf_nodes.end(), stream_dependencies.begin(), stream_dependencies.end(),
            std::back_inserter(unjoined_dependencies)
        );
        if (!unjoined_dependencies.empty()) {
            check_cu(cuStreamUpdateCaptureDependencies_f(
                cuda_stream, unjoined_dependencies.data(), unjoined_dependencies.size(),
                CU_STREAM_ADD_CAPTURE_DEPENDENCIES
            ));
            // ensure graph is still valid
            if (get_capture_graph(cuda_stream) != graph) {
                clean_up();
                return false;
            }
        }
    }

    // check if this graph has unfreed allocations, which require special handling
    std::vector<void*> unfreed_allocs;
    for (auto it = g_graph_allocs.begin(); it != g_graph_allocs.end(); ++it) {
        GraphAllocInfo& alloc_info = it->second;
        if (alloc_info.capture_id == capture_id)
            unfreed_allocs.push_back(it->first);
    }

    END_TRACE("end: unfreed allocations read");
    if (!unfreed_allocs.empty() || !tmp_allocs.empty()) {
        END_TRACE("end: user object");
        // Create a user object that will notify us when the instantiated graph is destroyed.
        // This works for external captures also, since we wouldn't otherwise know when
        // the externally-created graph instance gets deleted.
        // This callback is guaranteed to arrive after the graph has finished executing on the device,
        // not necessarily when cudaGraphExecDestroy() is called.
        GraphDestroyCallbackInfo* graph_info = new GraphDestroyCallbackInfo;
        graph_info->context = context ? context : get_current_context();
        graph_info->unfreed_allocs = unfreed_allocs;
        graph_info->tmp_allocs = tmp_allocs;
        cudaUserObject_t user_object;
        check_cuda(cudaUserObjectCreate(&user_object, graph_info, on_graph_destroy, 1, cudaUserObjectNoDestructorSync));
        check_cuda(cudaGraphRetainUserObject(graph, user_object, 1, cudaGraphUserObjectMove));

        // ensure graph is still valid
        if (get_capture_graph(cuda_stream) != graph) {
            clean_up();
            return false;
        }
    }

    // for external captures, we don't instantiate the graph ourselves, so we're done
    if (external)
        return true;

    // end the capture
    END_TRACE("end: stream end capture");
    if (!check_cuda(cudaStreamEndCapture(cuda_stream, &graph)))
        return false;
    END_TRACE("end: ended");

    // process deferred free list if no more captures are ongoing
    if (g_captures.empty()) {
        run_deferred_actions();
    }
    END_TRACE("end: done");

    if (graph_ret)
        *graph_ret = graph;

    return true;
}

bool wp_capture_debug_dot_print(void* graph, const char* path, uint32_t flags)
{
    if (!check_cuda(cudaGraphDebugDotPrint((cudaGraph_t)graph, path, flags)))
        return false;
    return true;
}

bool wp_cuda_graph_create_exec(void* context, void* stream, void* graph, void** graph_exec_ret)
{
    ContextGuard guard(context);

    cudaGraphExec_t graph_exec = NULL;
    if (!check_cuda(
            cudaGraphInstantiateWithFlags(&graph_exec, (cudaGraph_t)graph, cudaGraphInstantiateFlagAutoFreeOnLaunch)
        ))
        return false;

    // Usually uploading the graph explicitly is optional, but when updating graph nodes (e.g., indirect dispatch)
    // then the upload is required because otherwise the graph nodes that get updated might not yet be uploaded, which
    // results in undefined behavior.
    CUstream cuda_stream = static_cast<CUstream>(stream);
    if (!check_cuda(cudaGraphUpload(graph_exec, cuda_stream)))
        return false;

    if (graph_exec_ret)
        *graph_exec_ret = graph_exec;

    return true;
}

void* wp_cuda_graph_insert_memcpy(void* context, void* stream, void* dst, void* src, size_t size, int kind)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);
    cudaMemcpyKind memcpy_kind = static_cast<cudaMemcpyKind>(kind);

    // Get the current stream capturing graph
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &graph, &capture_deps, &dep_count)))
        return NULL;

    // abort if not capturing
    if (!graph || capture_status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        wp::set_error_string("Stream is not capturing");
        return NULL;
    }

    cudaGraphNode_t node = NULL;
    if (!check_cuda(cudaGraphAddMemcpyNode1D(&node, graph, capture_deps, dep_count, dst, src, size, memcpy_kind)))
        return NULL;

    if (!check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, &node, 1, cudaStreamSetCaptureDependencies)))
        return NULL;

    return node;
}

bool wp_cuda_graph_insert_memcpy_batch(
    void* context, void* stream, void** dsts, void** srcs, size_t* sizes, int* kinds, int count, void** nodes_ret
)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    // Get the current stream capturing graph
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &graph, &capture_deps, &dep_count)))
        return false;

    // abort if not capturing
    if (!graph || capture_status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        wp::set_error_string("Stream is not capturing");
        return false;
    }

#if 1
    // sequential version (copies executed on the same stream)
    //
    // TODO:
    // - figure out why sequential is faster
    // - overhead due to graph launch on multiple streams?
    //

    for (int i = 0; i < count; i++) {
        if (!check_cu(
                cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &graph, &capture_deps, &dep_count)
            ))
            return false;

        cudaMemcpyKind memcpy_kind = static_cast<cudaMemcpyKind>(kinds[i]);
        cudaGraphNode_t node = NULL;
        if (!check_cuda(
                cudaGraphAddMemcpyNode1D(&node, graph, capture_deps, dep_count, dsts[i], srcs[i], sizes[i], memcpy_kind)
            ))
            return false;
        nodes_ret[i] = node;

        if (!check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, &node, 1, cudaStreamSetCaptureDependencies)))
            return false;
    }
#else
    // parallel version (copies can execute on multiple streams)
    for (int i = 0; i < count; i++) {
        cudaMemcpyKind memcpy_kind = static_cast<cudaMemcpyKind>(kinds[i]);
        cudaGraphNode_t node = NULL;
        if (!check_cuda(
                cudaGraphAddMemcpyNode1D(&node, graph, capture_deps, dep_count, dsts[i], srcs[i], sizes[i], memcpy_kind)
            ))
            return false;
        nodes_ret[i] = node;
    }

    if (!check_cu(cuStreamUpdateCaptureDependencies_f(
            cuda_stream, (cudaGraphNode_t*)nodes_ret, count, cudaStreamSetCaptureDependencies
        )))
        return false;
#endif

    return true;
}

bool wp_cuda_graph_update_memcpy(void* graph_exec, void* node, void* dst, void* src, size_t size, int kind)
{
    cudaGraphExec_t cuda_graph_exec = static_cast<cudaGraphExec_t>(graph_exec);
    cudaGraphNode_t cuda_node = static_cast<cudaGraphNode_t>(node);
    cudaMemcpyKind memcpy_kind = static_cast<cudaMemcpyKind>(kind);

    if (!check_cuda(cudaGraphExecMemcpyNodeSetParams1D(cuda_graph_exec, cuda_node, dst, src, size, memcpy_kind)))
        return false;

    return true;
}

bool wp_cuda_graph_update_memcpy_batch(
    void* graph_exec, void** nodes, void** dsts, void** srcs, size_t* sizes, int* kinds, int count
)
{
    cudaGraphExec_t cuda_graph_exec = static_cast<cudaGraphExec_t>(graph_exec);

    for (int i = 0; i < count; i++) {
        cudaGraphNode_t cuda_node = static_cast<cudaGraphNode_t>(nodes[i]);
        cudaMemcpyKind memcpy_kind = static_cast<cudaMemcpyKind>(kinds[i]);

        if (!check_cuda(
                cudaGraphExecMemcpyNodeSetParams1D(cuda_graph_exec, cuda_node, dsts[i], srcs[i], sizes[i], memcpy_kind)
            ))
            return false;
    }

    return true;
}

void* wp_cuda_graph_insert_alloc_node(void* context, size_t size)
{
    // This function is used to exercise wp_alloc_device_async() during graph capture
    // and return the newly created alloc node. It is primarily a testing/debugging tool
    // and modifications are discouraged unless critical flaws are found.

    void* ptr = wp_alloc_device_async(context, size);

    // get the resulting alloc node
    if (ptr) {
        auto alloc_iter = g_graph_allocs.find(ptr);
        if (alloc_iter != g_graph_allocs.end()) {
            // NOTE: Set ``ref_exists`` to false for this allocation. This ensures that
            // the allocation is freed when the graph is destroyed.
            // Use wp_cuda_graph_insert_free_node() to explicitly free this allocation
            // during capture, but that can't free allocations that leak out of the capture.
            alloc_iter->second.ref_exists = false;
            return alloc_iter->second.node;
        }
    }

    wp::set_error_string("Warp error: failed to get MemAllocNode");
    return NULL;
}

void* wp_cuda_graph_insert_free_node(void* context, void* alloc_node_)
{
    // This function is used to exercise wp_free_device_async() during graph capture
    // and return the newly created free node. It is primarily a testing/debugging tool
    // and modifications are discouraged unless critical flaws are found.

    cudaGraphNode_t alloc_node = static_cast<cudaGraphNode_t>(alloc_node_);

    // verify input is an alloc node, or an empty node standing in for one on HIP on Windows (graph_standin_allocs)
    CUgraphNodeType alloc_node_type;
    if (!check_cu(cuGraphNodeGetType_f(alloc_node, &alloc_node_type)))
        return NULL;
    void* ptr = NULL;
    if (alloc_node_type == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
        // get the allocation pointer
        cudaMemAllocNodeParams alloc_params;
        if (!check_cuda(cudaGraphMemAllocNodeGetParams(alloc_node, &alloc_params)))
            return NULL;
        ptr = alloc_params.dptr;
    } else {
        auto standin = graph_standin_allocs().find(static_cast<void*>(alloc_node));
        if (alloc_node_type != CU_GRAPH_NODE_TYPE_EMPTY || standin == graph_standin_allocs().end()) {
            wp::set_error_string("Warp error: invalid node type, expected MemAllocNode");
            return NULL;
        }
        ptr = standin->second;
    }

    // use wp_free_device_async() and get the free node it added
    void* free_node = NULL;
    wp_free_device_async(context, ptr, &free_node);

    return free_node;
}

void* wp_cuda_graph_insert_empty_node(void* context)
{
    CUstream cuda_stream = get_current_stream(context);

    // get capture info
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &graph, &capture_deps, &dep_count))) {
        wp::append_error_string("Failed to get capture info");
        return NULL;
    }

    // abort if not capturing
    if (!graph || capture_status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        wp::set_error_string("Stream is not capturing");
        return NULL;
    }

    // add empty node
    cudaGraphNode_t empty_node = NULL;
    if (!check_cuda(cudaGraphAddEmptyNode(&empty_node, graph, capture_deps, dep_count))) {
        wp::append_error_string("Failed to add empty node");
        return NULL;
    }

    // update capture dependencies
    if (!check_cu(
            cuStreamUpdateCaptureDependencies_f(cuda_stream, &empty_node, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES)
        )) {
        wp::append_error_string("Failed to update capture dependencies");
        return NULL;
    }

    return empty_node;
}

int wp_cuda_graph_node_depends_on(void* argument, void* referent)
{
    return static_cast<int>(
        graph_node_depends_on(static_cast<cudaGraphNode_t>(argument), static_cast<cudaGraphNode_t>(referent))
    );
}

int wp_cuda_graph_alloc_query(void* alloc_node, void* query_node)
{
    return static_cast<int>(
        graph_alloc_query(static_cast<cudaGraphNode_t>(alloc_node), static_cast<cudaGraphNode_t>(query_node))
    );
}

// Pause/resume of an in-progress stream capture. These use only standard
// capture APIs (cudaStreamEndCapture / cudaStreamBeginCaptureToGraph, both
// aliased for HIP in hip_util.h) and are independent of conditional graph
// nodes, so they are compiled for all backends including HIP/ROCm.
bool wp_cuda_graph_pause_capture(void* context, void* stream, void** graph_ret)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);
    if (!check_cuda(cudaStreamEndCapture(cuda_stream, (cudaGraph_t*)graph_ret)))
        return false;
    return true;
}

bool wp_cuda_graph_resume_capture(void* context, void* stream, void* graph)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);
    cudaGraph_t cuda_graph = static_cast<cudaGraph_t>(graph);

    std::vector<cudaGraphNode_t> leaf_nodes;
    if (!get_graph_leaf_nodes(cuda_graph, leaf_nodes))
        return false;

    // Resume with the same capture mode the user picked at begin time so a
    // pause/resume cycle (driven by conditional/while graph nodes) does not
    // silently downgrade Global/Relaxed captures to ThreadLocal. The stream
    // must already be known to Warp with an active CaptureInfo at this
    // point because the resume path is only reached after a matching pause
    // on a Warp-managed capture; if either is missing something is badly
    // out of sync, fail fast rather than guess a mode.
    StreamInfo* stream_info = get_stream_info(cuda_stream);
    if (!stream_info) {
        wp::set_error_string("Warp error: resume_capture called on unknown stream");
        return false;
    }
    CaptureInfo* capture = stream_info->capture;
    if (!capture) {
        wp::set_error_string("Warp error: resume_capture called on stream with no active capture");
        return false;
    }
    cudaStreamCaptureMode resume_mode = capture->mode;

    if (!check_cuda(cudaStreamBeginCaptureToGraph(
            cuda_stream, cuda_graph, leaf_nodes.data(), nullptr, leaf_nodes.size(), resume_mode
        )))
        return false;

    return true;
}

bool wp_cuda_graph_create_empty(void* context, void** graph_ret)
{
    ContextGuard guard(context);
    cudaGraph_t graph = NULL;
    if (!check_cuda(cudaGraphCreate(&graph, 0)))
        return false;
    *graph_ret = graph;
    return true;
}

int wp_cuda_graph_node_count(void* graph)
{
    size_t count = 0;
    if (!check_cuda(cudaGraphGetNodes(static_cast<cudaGraph_t>(graph), NULL, &count)))
        return -1;
    return int(count);
}

// Support for conditional graph nodes available with CUDA 12.4+.
#if !defined(__HIP_PLATFORM_AMD__) && CUDA_VERSION >= 12040

// CUBIN or PTX data for compiled conditional modules, loaded on demand, keyed on device architecture
using ModuleKey = std::pair<int, bool>;  // <arch, use_ptx>
static std::map<ModuleKey, void*> g_conditional_modules;

// Compile module with conditional helper kernels
static void* compile_conditional_module(int arch, bool use_ptx)
{
    static const char* kernel_source = R"(
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

        extern "C" __global__ void set_conditional_if_handle_kernel(cudaGraphConditionalHandle handle, int* value)
        {
            if (threadIdx.x + blockIdx.x * blockDim.x == 0)
                cudaGraphSetConditional(handle, *value);
        }

        extern "C" __global__ void set_conditional_else_handle_kernel(cudaGraphConditionalHandle handle, int* value)
        {
            if (threadIdx.x + blockIdx.x * blockDim.x == 0)
                cudaGraphSetConditional(handle, !*value);
        }

        extern "C" __global__ void set_conditional_if_else_handles_kernel(cudaGraphConditionalHandle if_handle, cudaGraphConditionalHandle else_handle, int* value)
        {
            if (threadIdx.x + blockIdx.x * blockDim.x == 0)
            {
                cudaGraphSetConditional(if_handle, *value);
                cudaGraphSetConditional(else_handle, !*value);
            }
        }
    )";

    // avoid recompilation
    ModuleKey key = { arch, use_ptx };
    auto it = g_conditional_modules.find(key);
    if (it != g_conditional_modules.end())
        return it->second;

    nvrtcProgram prog;
    if (!check_nvrtc(nvrtcCreateProgram(&prog, kernel_source, "conditional_kernels", 0, NULL, NULL)))
        return NULL;

    char arch_opt[128];
    if (use_ptx)
        snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d", arch);
    else
        snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=sm_%d", arch);

    std::vector<const char*> opts;
    opts.push_back(arch_opt);

    const bool print_debug = (std::getenv("WARP_DEBUG") != nullptr);
    if (print_debug) {
        printf("NVRTC options (conditional module, arch=%d, use_ptx=%s):\n", arch, use_ptx ? "true" : "false");
        for (auto o : opts) {
            printf("%s\n", o);
        }
    }

    if (!check_nvrtc(nvrtcCompileProgram(prog, int(opts.size()), opts.data()))) {
        size_t log_size;
        if (check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size))) {
            std::vector<char> log(log_size);
            if (check_nvrtc(nvrtcGetProgramLog(prog, log.data())))
                fprintf(stderr, "%s", log.data());
        }
        nvrtcDestroyProgram(&prog);
        return NULL;
    }

    // get output
    char* output = NULL;
    size_t output_size = 0;

    if (use_ptx) {
        check_nvrtc(nvrtcGetPTXSize(prog, &output_size));
        if (output_size > 0) {
            output = new char[output_size];
            if (check_nvrtc(nvrtcGetPTX(prog, output)))
                g_conditional_modules[key] = output;
        }
    } else {
        check_nvrtc(nvrtcGetCUBINSize(prog, &output_size));
        if (output_size > 0) {
            output = new char[output_size];
            if (check_nvrtc(nvrtcGetCUBIN(prog, output)))
                g_conditional_modules[key] = output;
        }
    }

    nvrtcDestroyProgram(&prog);

    // return CUBIN or PTX data
    return output;
}


// Load module with conditional helper kernels
static CUmodule load_conditional_module(void* context, int arch, bool use_ptx)
{
    ContextInfo* context_info = get_context_info(context);
    if (!context_info)
        return NULL;

    // check if already loaded
    if (context_info->conditional_module)
        return context_info->conditional_module;

    // compile if needed
    void* compiled_module = compile_conditional_module(arch, use_ptx);
    if (!compiled_module) {
        fprintf(stderr, "Warp error: Failed to compile conditional kernels\n");
        return NULL;
    }

    // load module (handles both PTX and CUBIN data automatically)
    CUmodule module = NULL;
    if (!check_cu(cuModuleLoadDataEx_f(&module, compiled_module, 0, NULL, NULL))) {
        fprintf(stderr, "Warp error: Failed to load conditional kernels module\n");
        return NULL;
    }

    context_info->conditional_module = module;

    return module;
}

static CUfunction get_conditional_kernel(void* context, int arch, bool use_ptx, const char* name)
{
    // load module if needed
    CUmodule module = load_conditional_module(context, arch, use_ptx);
    if (!module)
        return NULL;

    CUfunction kernel;
    if (!check_cu(cuModuleGetFunction_f(&kernel, module, name))) {
        fprintf(stderr, "Warp error: Failed to get kernel %s\n", name);
        return NULL;
    }

    return kernel;
}

// https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/#combined_approach
// https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
// condition is a gpu pointer
// if_graph_ret and else_graph_ret should be NULL if not needed
bool wp_cuda_graph_insert_if_else(
    void* context, void* stream, int arch, bool use_ptx, int* condition, void** if_graph_ret, void** else_graph_ret
)
{
    bool has_if = if_graph_ret != NULL;
    bool has_else = else_graph_ret != NULL;
    int num_branches = int(has_if) + int(has_else);

    // if neither the IF nor ELSE branches are required, it's a no-op
    if (num_branches == 0)
        return true;

    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    // Get the current stream capturing graph
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t cuda_graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(
            cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &cuda_graph, &capture_deps, &dep_count)
        ))
        return false;

    // abort if not capturing
    if (!cuda_graph || capture_status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        wp::set_error_string("Stream is not capturing");
        return false;
    }

    // Multi-graph IF-ELSE nodes are supported starting with CUDA 12.8, but CUDA 12.8 and 12.9 drivers
    // can reparent nodes from child graphs incorrectly during instantiation. Use two single-body conditional nodes
    // until CUDA 12.x is no longer supported; the driver issue was fixed in CUDA 13.0.
    if (num_branches == 1) {
        cudaGraphConditionalHandle handle;
        check_cuda(cudaGraphConditionalHandleCreate(&handle, cuda_graph));

        // run a kernel to set the condition handle from the condition pointer
        // (need to negate the condition if only the else branch is used)
        CUfunction kernel;
        if (has_if)
            kernel = get_conditional_kernel(context, arch, use_ptx, "set_conditional_if_handle_kernel");
        else
            kernel = get_conditional_kernel(context, arch, use_ptx, "set_conditional_else_handle_kernel");

        if (!kernel) {
            wp::set_error_string("Failed to get built-in conditional kernel");
            return false;
        }

        void* kernel_args[2];
        kernel_args[0] = &handle;
        kernel_args[1] = &condition;

        if (!check_cu(cuLaunchKernel_f(kernel, 1, 1, 1, 1, 1, 1, 0, cuda_stream, kernel_args, NULL)))
            return false;

        if (!check_cu(
                cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &cuda_graph, &capture_deps, &dep_count)
            ))
            return false;

        // create conditional node
        CUgraphNode condition_node;
        CUgraphNodeParams condition_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
        condition_params.conditional.handle = handle;
        condition_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
        condition_params.conditional.size = num_branches;
        condition_params.conditional.ctx = get_current_context();
        if (!check_cu(cuGraphAddNode_f(&condition_node, cuda_graph, capture_deps, NULL, dep_count, &condition_params)))
            return false;

        if (!check_cu(
                cuStreamUpdateCaptureDependencies_f(cuda_stream, &condition_node, 1, cudaStreamSetCaptureDependencies)
            ))
            return false;

        if (num_branches == 1) {
            if (has_if)
                *if_graph_ret = condition_params.conditional.phGraph_out[0];
            else
                *else_graph_ret = condition_params.conditional.phGraph_out[0];
        } else {
            *if_graph_ret = condition_params.conditional.phGraph_out[0];
            *else_graph_ret = condition_params.conditional.phGraph_out[1];
        }
    } else {
        // Create IF node followed by an additional IF node with negated condition
        cudaGraphConditionalHandle if_handle, else_handle;
        check_cuda(cudaGraphConditionalHandleCreate(&if_handle, cuda_graph));
        check_cuda(cudaGraphConditionalHandleCreate(&else_handle, cuda_graph));

        CUfunction kernel = get_conditional_kernel(context, arch, use_ptx, "set_conditional_if_else_handles_kernel");
        if (!kernel) {
            wp::set_error_string("Failed to get built-in conditional kernel");
            return false;
        }

        void* kernel_args[3];
        kernel_args[0] = &if_handle;
        kernel_args[1] = &else_handle;
        kernel_args[2] = &condition;

        if (!check_cu(cuLaunchKernel_f(kernel, 1, 1, 1, 1, 1, 1, 0, cuda_stream, kernel_args, NULL)))
            return false;

        if (!check_cu(
                cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &cuda_graph, &capture_deps, &dep_count)
            ))
            return false;

        CUgraphNode if_node;
        CUgraphNodeParams if_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
        if_params.conditional.handle = if_handle;
        if_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
        if_params.conditional.size = 1;
        if_params.conditional.ctx = get_current_context();
        if (!check_cu(cuGraphAddNode_f(&if_node, cuda_graph, capture_deps, NULL, dep_count, &if_params)))
            return false;

        CUgraphNode else_node;
        CUgraphNodeParams else_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
        else_params.conditional.handle = else_handle;
        else_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
        else_params.conditional.size = 1;
        else_params.conditional.ctx = get_current_context();
        if (!check_cu(cuGraphAddNode_f(&else_node, cuda_graph, &if_node, NULL, 1, &else_params)))
            return false;

        if (!check_cu(
                cuStreamUpdateCaptureDependencies_f(cuda_stream, &else_node, 1, cudaStreamSetCaptureDependencies)
            ))
            return false;

        *if_graph_ret = if_params.conditional.phGraph_out[0];
        *else_graph_ret = else_params.conditional.phGraph_out[0];
    }

    return true;
}

// graph node type names for intelligible error reporting
static const char* get_graph_node_type_name(CUgraphNodeType type)
{
    static const std::unordered_map<CUgraphNodeType, const char*> names {
        { CU_GRAPH_NODE_TYPE_KERNEL, "kernel launch" },
        { CU_GRAPH_NODE_TYPE_MEMCPY, "memcpy" },
        { CU_GRAPH_NODE_TYPE_MEMSET, "memset" },
        { CU_GRAPH_NODE_TYPE_HOST, "host execution" },
        { CU_GRAPH_NODE_TYPE_GRAPH, "graph launch" },
        { CU_GRAPH_NODE_TYPE_EMPTY, "empty node" },
        { CU_GRAPH_NODE_TYPE_WAIT_EVENT, "event wait" },
        { CU_GRAPH_NODE_TYPE_EVENT_RECORD, "event record" },
        { CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL, "semaphore signal" },
        { CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT, "semaphore wait" },
        { CU_GRAPH_NODE_TYPE_MEM_ALLOC, "memory allocation" },
        { CU_GRAPH_NODE_TYPE_MEM_FREE, "memory deallocation" },
        { CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, "batched mem op" },
        { CU_GRAPH_NODE_TYPE_CONDITIONAL, "conditional node" },
    };

    auto it = names.find(type);
    if (it != names.end())
        return it->second;
    else
        return "unknown node";
}

// check if a graph can be launched as a child graph
static bool is_valid_child_graph(void* child_graph)
{
    // disallowed child graph nodes according to the documentation of cuGraphAddChildGraphNode()
    static const std::unordered_set<CUgraphNodeType> disallowed_nodes {
        CU_GRAPH_NODE_TYPE_MEM_ALLOC,
        CU_GRAPH_NODE_TYPE_MEM_FREE,
        CU_GRAPH_NODE_TYPE_CONDITIONAL,
    };

    if (!child_graph) {
        wp::set_error_string("Child graph is null");
        return false;
    }

    size_t num_nodes = 0;
    if (!check_cuda(cudaGraphGetNodes((cudaGraph_t)child_graph, NULL, &num_nodes)))
        return false;
    std::vector<cudaGraphNode_t> nodes(num_nodes);
    if (!check_cuda(cudaGraphGetNodes((cudaGraph_t)child_graph, nodes.data(), &num_nodes)))
        return false;

    for (size_t i = 0; i < num_nodes; i++) {
        // note: we use the driver API to get the node type, otherwise some nodes are not recognized correctly
        CUgraphNodeType node_type;
        check_cu(cuGraphNodeGetType_f(nodes[i], &node_type));
        auto it = disallowed_nodes.find(node_type);
        if (it != disallowed_nodes.end()) {
            wp::set_error_string(
                "Child graph contains an unsupported operation (%s)", get_graph_node_type_name(node_type)
            );
            return false;
        }
    }

    return true;
}

// check if a graph can be used as a conditional body graph
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements
bool wp_cuda_graph_check_conditional_body(void* body_graph)
{
    static const std::unordered_set<CUgraphNodeType> allowed_nodes {
        CU_GRAPH_NODE_TYPE_MEMCPY, CU_GRAPH_NODE_TYPE_MEMSET, CU_GRAPH_NODE_TYPE_KERNEL,
        CU_GRAPH_NODE_TYPE_GRAPH,  CU_GRAPH_NODE_TYPE_EMPTY,  CU_GRAPH_NODE_TYPE_CONDITIONAL,
    };

    if (!body_graph) {
        wp::set_error_string("Conditional body graph is null");
        return false;
    }

    size_t num_nodes = 0;
    if (!check_cuda(cudaGraphGetNodes((cudaGraph_t)body_graph, NULL, &num_nodes)))
        return false;
    std::vector<cudaGraphNode_t> nodes(num_nodes);
    if (!check_cuda(cudaGraphGetNodes((cudaGraph_t)body_graph, nodes.data(), &num_nodes)))
        return false;

    for (size_t i = 0; i < num_nodes; i++) {
        // note: we use the driver API to get the node type, otherwise some nodes are not recognized correctly
        CUgraphNodeType node_type;
        check_cu(cuGraphNodeGetType_f(nodes[i], &node_type));
        if (allowed_nodes.find(node_type) == allowed_nodes.end()) {
            wp::set_error_string(
                "Conditional body graph contains an unsupported operation (%s)", get_graph_node_type_name(node_type)
            );
            return false;
        } else if (node_type == CU_GRAPH_NODE_TYPE_GRAPH) {
            // check nested child graphs recursively
            cudaGraph_t child_graph = NULL;
            if (!check_cuda(cudaGraphChildGraphNodeGetGraph(nodes[i], &child_graph)))
                return false;
            if (!wp_cuda_graph_check_conditional_body(child_graph))
                return false;
        }
    }

    return true;
}

bool wp_cuda_graph_insert_child_graph(void* context, void* stream, void* child_graph)
{
    if (!is_valid_child_graph(child_graph))
        return false;

    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    // Get the current stream capturing graph
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    void* cuda_graph = NULL;
    const CUgraphNode* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(cuStreamGetCaptureInfo_f(
            cuda_stream, &capture_status, nullptr, (cudaGraph_t*)&cuda_graph, &capture_deps, &dep_count
        )))
        return false;

    if (!wp_cuda_graph_pause_capture(context, cuda_stream, &cuda_graph))
        return false;

    cudaGraphNode_t body_node;
    if (!check_cuda(cudaGraphAddChildGraphNode(
            &body_node, static_cast<cudaGraph_t>(cuda_graph), capture_deps, dep_count,
            static_cast<cudaGraph_t>(child_graph)
        )))
        return false;

    if (!wp_cuda_graph_resume_capture(context, cuda_stream, cuda_graph))
        return false;

    if (!check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, &body_node, 1, cudaStreamSetCaptureDependencies)))
        return false;

    return true;
}

bool wp_cuda_graph_insert_while(
    void* context, void* stream, int arch, bool use_ptx, int* condition, void** body_graph_ret, uint64_t* handle_ret
)
{
    // if there's no body, it's a no-op
    if (!body_graph_ret)
        return true;

    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    // Get the current stream capturing graph
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    cudaGraph_t cuda_graph = NULL;
    const cudaGraphNode_t* capture_deps = NULL;
    size_t dep_count = 0;
    if (!check_cu(
            cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &cuda_graph, &capture_deps, &dep_count)
        ))
        return false;

    // abort if not capturing
    if (!cuda_graph || capture_status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        wp::set_error_string("Stream is not capturing");
        return false;
    }

    cudaGraphConditionalHandle handle;
    if (!check_cuda(cudaGraphConditionalHandleCreate(&handle, cuda_graph)))
        return false;

    // launch a kernel to set the condition handle from condition pointer
    CUfunction kernel = get_conditional_kernel(context, arch, use_ptx, "set_conditional_if_handle_kernel");
    if (!kernel) {
        wp::set_error_string("Failed to get built-in conditional kernel");
        return false;
    }

    void* kernel_args[2];
    kernel_args[0] = &handle;
    kernel_args[1] = &condition;

    if (!check_cu(cuLaunchKernel_f(kernel, 1, 1, 1, 1, 1, 1, 0, cuda_stream, kernel_args, NULL)))
        return false;

    if (!check_cu(
            cuStreamGetCaptureInfo_f(cuda_stream, &capture_status, nullptr, &cuda_graph, &capture_deps, &dep_count)
        ))
        return false;

    // insert conditional graph node
    CUgraphNode while_node;
    CUgraphNodeParams while_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
    while_params.conditional.handle = handle;
    while_params.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
    while_params.conditional.size = 1;
    while_params.conditional.ctx = get_current_context();
    if (!check_cu(cuGraphAddNode_f(&while_node, cuda_graph, capture_deps, NULL, dep_count, &while_params)))
        return false;

    if (!check_cu(cuStreamUpdateCaptureDependencies_f(cuda_stream, &while_node, 1, cudaStreamSetCaptureDependencies)))
        return false;

    *body_graph_ret = while_params.conditional.phGraph_out[0];
    *handle_ret = handle;

    return true;
}

bool wp_cuda_graph_set_condition(void* context, void* stream, int arch, bool use_ptx, int* condition, uint64_t handle)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    // launch a kernel to set the condition handle from condition pointer
    CUfunction kernel = get_conditional_kernel(context, arch, use_ptx, "set_conditional_if_handle_kernel");
    if (!kernel) {
        wp::set_error_string("Failed to get built-in conditional kernel");
        return false;
    }

    void* kernel_args[2];
    kernel_args[0] = &handle;
    kernel_args[1] = &condition;

    if (!check_cu(cuLaunchKernel_f(kernel, 1, 1, 1, 1, 1, 1, 0, cuda_stream, kernel_args, NULL)))
        return false;

    return true;
}

#else
// stubs for conditional graph node API if CUDA toolkit is too old.

bool wp_cuda_graph_insert_if_else(
    void* context, void* stream, int arch, bool use_ptx, int* condition, void** if_graph_ret, void** else_graph_ret
)
{
    wp::set_error_string("Warp error: Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes");
    return false;
}

bool wp_cuda_graph_insert_while(
    void* context, void* stream, int arch, bool use_ptx, int* condition, void** body_graph_ret, uint64_t* handle_ret
)
{
    wp::set_error_string("Warp error: Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes");
    return false;
}

bool wp_cuda_graph_set_condition(void* context, void* stream, int arch, bool use_ptx, int* condition, uint64_t handle)
{
    wp::set_error_string("Warp error: Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes");
    return false;
}

bool wp_cuda_graph_insert_child_graph(void* context, void* stream, void* child_graph)
{
    wp::set_error_string("Warp error: Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes");
    return false;
}

bool wp_cuda_graph_check_conditional_body(void* body_graph)
{
    wp::set_error_string("Warp error: Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes");
    return false;
}

#endif  // support for conditional graph nodes


bool wp_cuda_graph_launch(void* graph_exec, void* stream)
{
    WP_HIP_ISSUE(stream);
    // TODO: allow naming graphs?
    begin_cuda_range(WP_TIMING_GRAPH, stream, get_stream_context(stream), "graph");

    bool result = check_cuda(cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream));

    end_cuda_range(WP_TIMING_GRAPH, stream);

    return result;
}

bool wp_cuda_graph_destroy(void* context, void* graph)
{
    return release_or_defer(ReleaseKind::Graph, context, (uint64_t)graph);
}

bool wp_cuda_graph_exec_destroy(void* context, void* graph_exec)
{
    return release_or_defer(ReleaseKind::GraphExec, context, (uint64_t)graph_exec);
}

bool write_file(const char* data, size_t size, std::string filename, const char* mode)
{
    const bool print_debug = (std::getenv("WARP_DEBUG") != nullptr);
    if (print_debug) {
        printf("Writing %zu B to %s (%s)\n", size, filename.c_str(), mode);
    }
    FILE* file = fopen(filename.c_str(), mode);
    if (file) {
        if (fwrite(data, 1, size, file) != size) {
            fprintf(stderr, "Warp error: Failed to write to output file '%s'\n", filename.c_str());
            return false;
        }
        fclose(file);
        return true;
    } else {
        fprintf(stderr, "Warp error: Failed to open file '%s'\n", filename.c_str());
        return false;
    }
}

#if WP_ENABLE_MATHDX && !defined(__HIP_PLATFORM_AMD__)
bool check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result, const char* file, int line)
{
    if (result != NVJITLINK_SUCCESS) {
        fprintf(stderr, "nvJitLink error: %d on %s:%d\n", (int)result, file, line);
        size_t lsize;
        result = nvJitLinkGetErrorLogSize(handle, &lsize);
        if (result == NVJITLINK_SUCCESS && lsize > 0) {
            std::vector<char> log(lsize);
            result = nvJitLinkGetErrorLog(handle, log.data());
            if (result == NVJITLINK_SUCCESS) {
                fprintf(stderr, "%s\n", log.data());
            }
        }
        return false;
    } else {
        return true;
    }
}
#endif

// Append every NVRTC option shared by Warp's normal module compiler and APIC's
// portable source-recompile path. Keep path-specific options (include paths,
// precompiled headers, link inputs, and compile-time tracing) in the callers.
// In particular, CUDA-version workarounds belong here so a source-recompiled
// APIC module cannot silently drift from the module used during capture.
static void append_common_nvrtc_compile_options(
    std::vector<const char*>& options,
    bool use_ptx,
    int arch,
    bool debug,
    int optimization_level,
    bool verify_fp,
    bool fast_math,
    bool fuse_fp,
    bool lineinfo
)
{
    options.push_back("--std=c++17");

#if CUDA_VERSION >= 12080 && CUDA_VERSION < 13000
    // CUDA 12 miscompiles optimized CUBIN texture sampling when texture handles
    // vary across lanes on sm_90 and newer targets. CUDA 12.8 is covered as a
    // precaution because it is not exercised in CI.
    const bool is_affected_texture_cubin_target = arch >= 90;
#else
    // CUDA 13 CUBIN has not reproduced the miscompile on any target tested so far.
    const bool is_affected_texture_cubin_target = false;
#endif

#if CUDA_VERSION >= 12080 && CUDA_VERSION < 12090
    // Precaution, untested: CUDA 12.8 is not expected to honor optimization
    // levels here, so even level 0 may trigger the miscompile.
    const bool optimizations_may_trigger_texture_bug = true;
#else
    const bool optimizations_may_trigger_texture_bug = optimization_level > 0;
#endif

    if (!use_ptx && is_affected_texture_cubin_target && optimizations_may_trigger_texture_bug)
        options.push_back("--define-macro=WP_WORKAROUND_CUDA_TEXTURE_CUBIN");

    // CUDA 12.9+ supports --Ofast-compile, which works inversely to normal -O
    // optimization levels.
#if CUDA_VERSION >= 12090
    switch (optimization_level) {
    case 0:
#if CUDA_VERSION >= 13010
        // CUDA 13.1+ corrupts process-wide NVRTC compiler state after a
        // --Ofast-compile=max build. Later builds can then emit invalid parameter
        // addressing, so use the next-fastest setting until NVIDIA fixes NVRTC.
        options.push_back("--Ofast-compile=mid");
#else
        options.push_back("--Ofast-compile=max");
#endif
        break;
    case 1:
        options.push_back("--Ofast-compile=mid");
        break;
    case 2:
        options.push_back("--Ofast-compile=min");
        break;
    default:
        options.push_back("--Ofast-compile=0");
        break;  // 3 and up
    }
#endif

    if (debug) {
        options.push_back("--define-macro=_DEBUG");
        options.push_back("--generate-line-info");
#ifndef _WIN32
        options.push_back("--device-debug");  // -G
#endif
    } else {
        options.push_back("--define-macro=NDEBUG");

        if (lineinfo)
            options.push_back("--generate-line-info");
    }

    if (verify_fp)
        options.push_back("--define-macro=WP_VERIFY_FP");
    else
        options.push_back("--undefine-macro=WP_VERIFY_FP");

#if WP_ENABLE_MATHDX
    options.push_back("--define-macro=WP_ENABLE_MATHDX=1");
#else
    options.push_back("--define-macro=WP_ENABLE_MATHDX=0");
#endif

    if (fast_math)
        options.push_back("--use_fast_math");

    if (fuse_fp)
        options.push_back("--fmad=true");
    else
        options.push_back("--fmad=false");

    options.push_back("--device-as-default-execution-space");
    options.push_back("--extra-device-vectorization");
    options.push_back("--restrict");
    options.push_back("--diag-suppress=177,550");  // "was declared but never referenced", "was set but never used"
}

#if defined(__HIP_PLATFORM_AMD__)
// Builds hipRTC's options for a Warp module, matching NVRTC's semantics. The same list is used for the module compile and for
// APIC's source fallback, so a module recompiled from a saved graph's source is the same module Warp itself builds.
static std::vector<std::string> hip_compile_options(
    const char* gfx_arch,
    const char* include_dir,
    int num_cuda_include_dirs,
    const char** cuda_include_dirs,
    bool debug,
    int optimization_level,
    bool verify_fp,
    bool fast_math,
    bool fuse_fp,
    bool lineinfo
)
{
    std::vector<std::string> options;
    if (gfx_arch && gfx_arch[0])
        options.push_back(std::string("--gpu-architecture=") + gfx_arch);
    if (include_dir && include_dir[0])
        options.push_back(std::string("-I") + include_dir);
    for (int i = 0; i < num_cuda_include_dirs; ++i) {
        if (cuda_include_dirs && cuda_include_dirs[i])
            options.push_back(std::string("-I") + cuda_include_dirs[i]);
    }

    options.push_back("--std=c++17");

    if (debug) {
        options.push_back("-O0");
        options.push_back("-g");
        options.push_back("-D_DEBUG");
    } else {
        if (optimization_level >= 0)
            options.push_back(std::string("-O") + std::to_string(optimization_level));
        // Match NVRTC: define NDEBUG in release mode
        options.push_back("-DNDEBUG");
    }

    // Match NVRTC by defining WP_ENABLE_MATHDX as 1 while the HIP device library is on (wp_is_mathdx_enabled). A kind the library
    // provides is called through its linked symbol (WP_MATHDX_GEMM_LINKED in tile_matmul.h), and every other kind keeps the header
    // path.
    options.push_back(wp_is_mathdx_enabled() ? "-DWP_ENABLE_MATHDX=1" : "-DWP_ENABLE_MATHDX=0");

    if (verify_fp)
        options.push_back("-DWP_VERIFY_FP");

    if (fast_math) {
        options.push_back("-ffast-math");
        // Match NVRTC --use_fast_math: substitute sinf/cosf/expf/logf etc.
        // with single-instruction approximate hardware intrinsics.
        // powf is unaffected (no approximate hardware instruction on CDNA).
        options.push_back("-fgpu-approx-transcendentals");
        // Match NVRTC --ftz=true (part of --use_fast_math): flush
        // single-precision denormals to zero for faster FP execution.
        options.push_back("-fgpu-flush-denormals-to-zero");
        // The -fno-* options below leave clang's __FAST_MATH__ undefined, so this macro tells Warp's headers instead. Under it,
        // builtin.h's pow takes NVRTC's __powf form.
        options.push_back("-DWP_FAST_MATH=1");
    }

    // Match NVRTC default: strict IEEE 754 floating-point semantics.
    // These override the dangerous parts of -ffast-math while keeping
    // the transcendental substitutions from -fgpu-approx-transcendentals.
    options.push_back("-fno-finite-math-only");  // Keep inf and NaN.
    options.push_back("-fno-associative-math");  // Don't reorder min/max chains.
    options.push_back("-fno-reciprocal-math");  // Don't apply unsafe 1/x transforms.

    // Match NVRTC default: safe pointer aliasing for reinterpret_casts
    options.push_back("-fno-strict-aliasing");

    // Use hardware instructions for floating-point atomics (global_atomic_add_f32 and so on) instead of compare-and-swap loops. The
    // CAS expansion exists for fine-grained (host-coherent) memory and for denormal semantics, but Warp's device arrays are
    // coarse-grained device allocations, and CUDA's float atomics also flush subnormals.
    options.push_back("-munsafe-fp-atomics");

    // Match NVRTC's --fmad by fusing as requested or not at all. Use fast-honor-pragmas, never fast. The fast setting reaches into
    // the device library linked into the module, and AMD's own exp and pow came out 45 and 52 ulp off because their error-free steps
    // were fused. Under fast-honor-pragmas they stay at 1 ulp, and a user's multiply and add still fuse across statements and inline
    // calls, which is how Warp writes them. The on setting keeps the library correct but only fuses inside a single expression,
    // which covers none of Warp's code.
    options.push_back(fuse_fp ? "-ffp-contract=fast-honor-pragmas" : "-ffp-contract=off");

    if (lineinfo)
        options.push_back("-gline-tables-only");

    // Suppress warnings matching NVRTC's --diag-suppress=177,550
    options.push_back("-Wno-unused-variable");
    options.push_back("-Wno-unused-but-set-variable");
    return options;
}

// A .wrp file stores its target architecture as one uint32. A CUDA sm number is stored as it is, and a HIP gfx target is stored
// with this flag set (gfx1151 becomes 0x40001151), which no sm number reaches. Python's _APIC_HIP_ARCH_FLAG is the same number.
#define WP_APIC_HIP_ARCH_FLAG 0x40000000

// Returns the flagged code of a device's gfx target ("gfx1151" or "gfx90a:sramecc+:xnack-" become their hex digits), or 0 if the
// name can't be read.
static int hip_apic_arch_code(const char* gfx_name)
{
    if (!gfx_name || strncmp(gfx_name, "gfx", 3) != 0)
        return 0;
    int digits = 0;
    int count = 0;
    for (const char* c = gfx_name + 3; *c && *c != ':'; ++c, ++count) {
        int value = -1;
        if (*c >= '0' && *c <= '9')
            value = *c - '0';
        else if (*c >= 'a' && *c <= 'f')
            value = *c - 'a' + 10;
        else if (*c >= 'A' && *c <= 'F')
            value = *c - 'A' + 10;
        if (value < 0 || count >= 7)
            return 0;
        digits = digits * 16 + value;
    }
    return count > 0 ? (WP_APIC_HIP_ARCH_FLAG | digits) : 0;
}

// Converts a flagged code back to the gfx name hipRTC targets, such as "gfx1151".
static std::string hip_apic_arch_name(int arch_code)
{
    char name[32];
    snprintf(name, sizeof(name), "gfx%x", static_cast<unsigned>(arch_code & ~WP_APIC_HIP_ARCH_FLAG));
    return name;
}
#endif

// Returns the target architecture APIC compares and compiles for on a device, which is the sm number on CUDA and the flagged gfx
// code on HIP.
static int apic_device_arch(const DeviceInfo* device_info)
{
    if (!device_info)
        return 0;
#if defined(__HIP_PLATFORM_AMD__)
    return hip_apic_arch_code(device_info->arch_str);
#else
    return device_info->arch;
#endif
}

#if defined(__HIP_PLATFORM_AMD__)
// The HIP device library provides for AMD GPUs what MathDx provides for a CUDA module. A GEMM that tile_matmul needs is generated
// as one extern "C" __device__ function on top of Warp's own GEMM for HIP. That GEMM (in tile_matmul.h) uses the matrix
// instructions for half and bfloat16 tiles made of whole 16x16 blocks, the register-blocked GEMM otherwise, and TF32 rounding
// when requested. The function is compiled separately by hipRTC to relocatable AMD bitcode (-fgpu-rdc), kept in Warp's LTO cache
// the same way cuBLASDx's LTO-IR is (build_lto_dot in build.py), and linked into every module that calls it by hipRTC's linker
// (hip_link_bitcode), where it gets inlined, as nvJitLink inlines a cuBLASDx GEMM. Its symbol name encodes everything that defines
// it (size, arrangement, leading dimensions, precision, alignment, and block size), so one compiled GEMM serves every module that
// asks for the same one.

// Compiles one source with hipRTC to relocatable LLVM bitcode. The compiler's log is printed on failure unless the caller expects
// one.
static bool hip_rtc_bitcode(
    const std::string& source, const char* name, const std::vector<std::string>& options, std::vector<char>& bitcode, bool report
)
{
    hiprtcProgram prog;
    if (hiprtcCreateProgram(&prog, source.c_str(), name, 0, nullptr, nullptr) != HIPRTC_SUCCESS)
        return false;
    std::vector<const char*> opts;
    for (const std::string& option : options)
        opts.push_back(option.c_str());
    const hiprtcResult res = hiprtcCompileProgram(prog, int(opts.size()), opts.data());
    if (res != HIPRTC_SUCCESS && report) {
        size_t log_size = 0;
        if (hiprtcGetProgramLogSize(prog, &log_size) == HIPRTC_SUCCESS && log_size > 1) {
            std::vector<char> log(log_size);
            if (hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                fprintf(stderr, "Warp HIP device library: %s failed to compile:\n%s\n", name, log.data());
        }
    }
    bool ok = res == HIPRTC_SUCCESS;
    size_t size = 0;
    if (ok)
        ok = hiprtcGetBitcodeSize(prog, &size) == HIPRTC_SUCCESS && size > 0;
    if (ok) {
        bitcode.resize(size);
        ok = hiprtcGetBitcode(prog, bitcode.data()) == HIPRTC_SUCCESS;
    }
    hiprtcDestroyProgram(&prog);
    return ok;
}

// Links a module's bitcode and its library functions' bitcode into one code object with hipRTC's linker, running the backend at the
// module's own optimization level.
static bool hip_link_bitcode(
    const std::vector<char>& module_bitcode,
    size_t num_inputs,
    char** inputs,
    size_t* input_sizes,
    int optimization_level,
    std::vector<char>& code
)
{
    const std::string level = "-O" + std::to_string(optimization_level < 0 ? 3 : optimization_level);
    const char* isa_options[] = { level.c_str() };
    hiprtcJIT_option keys[] = { HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT };
    void* values[] = { (void*)isa_options, (void*)(size_t)1 };
    hiprtcLinkState link;
    if (hiprtcLinkCreate(2, keys, values, &link) != HIPRTC_SUCCESS) {
        fprintf(stderr, "Warp HIP device library: the linker did not start\n");
        return false;
    }
    bool ok = hiprtcLinkAddData(
                  link, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)module_bitcode.data(), module_bitcode.size(), "module", 0, nullptr,
                  nullptr
              )
        == HIPRTC_SUCCESS;
    for (size_t i = 0; ok && i < num_inputs; ++i) {
        ok = hiprtcLinkAddData(link, HIPRTC_JIT_INPUT_LLVM_BITCODE, inputs[i], input_sizes[i], "library", 0, nullptr, nullptr)
            == HIPRTC_SUCCESS;
    }
    void* binary = nullptr;
    size_t binary_size = 0;
    if (ok)
        ok = hiprtcLinkComplete(link, &binary, &binary_size) == HIPRTC_SUCCESS && binary && binary_size > 0;
    if (ok)
        code.assign(static_cast<char*>(binary), static_cast<char*>(binary) + binary_size);  // The link state owns the binary.
    else
        fprintf(stderr, "Warp HIP device library: linking %zu library function(s) into the module failed\n", num_inputs);
    hiprtcLinkDestroy(link);
    return ok;
}

// Returns the C++ element type for a cuBLASDx precision code (commondxPrecision 2 is bf16, 3 f16, 4 tf32, 5 f32, and 6 f64), or ""
// for anything else.
static const char* hip_dx_element(int precision)
{
    switch (precision) {
    case 2:
        return "wp::bfloat16";
    case 3:
        return "wp::float16";
    case 4:
    case 5:
        return "wp::float32";
    case 6:
        return "wp::float64";
    default:
        return "";
    }
}

static int hip_dx_element_bytes(int precision)
{
    return precision == 6 ? 8 : (precision == 2 || precision == 3) ? 2 : 4;
}

// Generates the GEMM C = alpha * A @ B + beta * C on a block's shared-memory tiles, declared to Warp the same way cuBLASDx's is, as
// `void symbol(C* alpha, A* a, B* b, C* beta, C* c)`. A is M x K, B is K x N, and C is M x N. Each is row-major (1) or
// column-major (0) with its own leading dimension (0 means dense), and the block is num_threads wide. As in BLAS, C is not read
// when beta is zero. On HIP, `mathdx_include_dir` carries the gfx target the module is compiled for and `include_dirs` carries
// Warp's native headers (see build.py). `arch` is CUDA's and is not used.
bool wp_cuda_compile_dot(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int precision_A,
    int precision_B,
    int precision_C,
    int type,
    int arrangement_A,
    int arrangement_B,
    int arrangement_C,
    int num_threads,
    int lda,
    int ldb,
    int ldc,
    int alignment_A,
    int alignment_B,
    int alignment_C,
    int enable_static_block_dim,
    int suppress_errors
)
{
    (void)arch;
    (void)enable_static_block_dim;  // The GEMM is always compiled for exactly num_threads lanes (WP_TILE_BLOCK_DIM).
    const char* gfx_arch = mathdx_include_dir;
    if (!ltoir_output_path || !symbol_name || !gfx_arch || !gfx_arch[0] || num_include_dirs < 1 || !include_dirs
        || !include_dirs[0]) {
        fprintf(stderr, "Warp HIP device library: a GEMM needs its output, symbol, target and Warp's headers\n");
        return false;
    }
    const char* ta = hip_dx_element(precision_A);
    const char* tb = hip_dx_element(precision_B);
    const char* tc = hip_dx_element(precision_C);
    // tile_matmul never asks for complex GEMMs (type 1) or mixed TF32, so they are refused rather than computed some other way.
    const bool tf32 = precision_A == 4 && precision_B == 4;
    if (type != 0 || !ta[0] || !tb[0] || !tc[0] || precision_C == 4 || ((precision_A == 4) != (precision_B == 4)) || M <= 0
        || N <= 0 || K <= 0 || num_threads <= 0) {
        if (!suppress_errors)
            fprintf(stderr, "Warp HIP device library: no GEMM of this kind (%s)\n", symbol_name);
        return false;
    }
    // Each operand's strides follow from its arrangement and leading dimension, (ld, 1) for row-major and (1, ld) for column-major.
    auto strides = [](int arrangement, int rows, int cols, int ld, int& s0, int& s1) {
        if (ld <= 0)
            ld = arrangement == 1 ? cols : rows;
        if (arrangement == 1) {
            s0 = ld;
            s1 = 1;
        } else {
            s0 = 1;
            s1 = ld;
        }
    };
    int sa0, sa1, sb0, sb1, sc0, sc1;
    strides(arrangement_A, M, K, lda, sa0, sa1);
    strides(arrangement_B, K, N, ldb, sb0, sb1);
    strides(arrangement_C, M, N, ldc, sc0, sc1);
    // A requested alignment, in bytes, is promised to the compiler. Zero leaves each element's natural alignment.
    auto aligned = [](const char* name, const char* type_name, int alignment, int element_bytes) {
        const int a = alignment > 0 ? alignment : element_bytes;
        return std::string("    ") + type_name + "* " + name + "_ptr = static_cast<" + type_name + "*>(__builtin_assume_aligned("
            + name + ", " + std::to_string(a) + "));\n";
    };
    std::string src;
    src += "#define WP_TILE_BLOCK_DIM " + std::to_string(num_threads) + "\n";
    src += "#define WP_NO_CRT\n#include \"builtin.h\"\n\n";
    src += "namespace {\ntemplate <typename T> struct hip_dx_operand {\n    T* ptr;\n};\n}  // namespace\n\n";
    src += std::string("extern \"C\" __device__ void ") + symbol_name + "(" + tc + "* alpha, " + ta + "* a, " + tb + "* b, " + tc
        + "* beta, " + tc + "* c)\n{\n";
    src += "    using LayoutA = wp::tile_layout_strided_t<wp::tile_shape_t<" + std::to_string(M) + ", " + std::to_string(K)
        + ">, wp::tile_stride_t<" + std::to_string(sa0) + ", " + std::to_string(sa1) + ">>;\n";
    src += "    using LayoutB = wp::tile_layout_strided_t<wp::tile_shape_t<" + std::to_string(K) + ", " + std::to_string(N)
        + ">, wp::tile_stride_t<" + std::to_string(sb0) + ", " + std::to_string(sb1) + ">>;\n";
    src += "    using LayoutC = wp::tile_layout_strided_t<wp::tile_shape_t<" + std::to_string(M) + ", " + std::to_string(N)
        + ">, wp::tile_stride_t<" + std::to_string(sc0) + ", " + std::to_string(sc1) + ">>;\n";
    src += aligned("a", ta, alignment_A, hip_dx_element_bytes(precision_A));
    src += aligned("b", tb, alignment_B, hip_dx_element_bytes(precision_B));
    src += aligned("c", tc, alignment_C, hip_dx_element_bytes(precision_C));
    src += std::string("    hip_dx_operand<") + ta + "> A{a_ptr};\n    hip_dx_operand<" + tb + "> B{b_ptr};\n    hip_dx_operand<" + tc
        + "> C{c_ptr};\n";
    src += std::string("    wp::partitioned_gemm::scalar_matmul_impl<true, ") + (tf32 ? "true" : "false")
        + ", LayoutA, LayoutB, LayoutC>(A, B, C, *alpha, *beta);\n}\n";

    std::vector<std::string> options =
        hip_compile_options(gfx_arch, include_dirs[0], num_include_dirs - 1, include_dirs + 1, false, 3, false, false, true, false);
    options.push_back("-fgpu-rdc");
    std::vector<char> bitcode;
    const std::string name = std::string(symbol_name) + ".hip";
    if (!hip_rtc_bitcode(src, name.c_str(), options, bitcode, !suppress_errors))
        return false;
    return write_file(bitcode.data(), bitcode.size(), ltoir_output_path, "wb");
}

// Generates the FFT of one row the way Warp calls cuFFTDx's register API (tile_fft_entry in tile_fft.h), as
// `void symbol(Complex* data, char* shared)`. Every lane of the block calls it with elements_per_thread elements of the row in
// registers, where lane t holds elements t + j * (size / elements_per_thread), and `shared` is the number of bytes this compile
// reports. The row is gathered into shared memory, transformed in place there (with the cooperative radix-2 FFT for a power of two
// and the cooperative direct DFT otherwise, each twiddle computed from its own exact angle as in tile_fft.h), and scattered back in
// the same order. Neither direction is normalized, matching cuFFTDx. Direction 0 is forward (e^-i) and 1 is inverse (e^+i), and
// precision 5 is float32 and 6 is float64.
bool wp_cuda_compile_fft(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int size,
    int elements_per_thread,
    int direction,
    int precision,
    int* shared_memory_size
)
{
    (void)arch;
    const char* gfx_arch = mathdx_include_dir;
    if (!ltoir_output_path || !symbol_name || !shared_memory_size || !gfx_arch || !gfx_arch[0] || num_include_dirs < 1
        || !include_dirs || !include_dirs[0]) {
        fprintf(stderr, "Warp HIP device library: an FFT needs its output, symbol, target and Warp's headers\n");
        return false;
    }
    if (size <= 0 || elements_per_thread <= 0 || size % elements_per_thread != 0 || (precision != 5 && precision != 6)
        || (direction != 0 && direction != 1)) {
        fprintf(stderr, "Warp HIP device library: no FFT of this kind (%s)\n", symbol_name);
        return false;
    }
    const int threads = size / elements_per_thread;
    const bool pow2 = (size & (size - 1)) == 0;
    int log_n = 0;
    while ((1 << log_n) < size)
        ++log_n;
    const char* complex = precision == 5 ? "wp::vec2f" : "wp::vec2d";
    const char* real = precision == 5 ? "float" : "double";
    const int complex_bytes = precision == 5 ? 8 : 16;
    // Shared memory holds the row, and for a direct DFT also its transform and the twiddle table.
    *shared_memory_size = size * complex_bytes * (pow2 ? 1 : 3);

    const std::string n = std::to_string(size);
    const std::string per = std::to_string(threads);
    const std::string ept = std::to_string(elements_per_thread);
    const std::string sign = direction == 0 ? "-1" : "1";
    std::string src;
    src += "#define WP_TILE_BLOCK_DIM " + per + "\n#define WP_NO_CRT\n#include \"builtin.h\"\n\n";
    src += std::string("extern \"C\" __device__ void ") + symbol_name + "(" + complex + "* data, char* shared)\n{\n";
    src += std::string("    using Complex = ") + complex + ";\n    using T = " + real + ";\n";
    src += "    Complex* row = reinterpret_cast<Complex*>(shared);\n";
    src += "    const int lane = WP_TILE_THREAD_IDX;\n";
    src += "    if (lane < " + per + ")\n        for (int j = 0; j < " + ept + "; ++j)\n            row[lane + j * " + per
        + "] = data[j];\n";
    src += "    WP_TILE_SYNC();\n";
    // Each loop runs over the lanes the row needs. A block wider than size / elements_per_thread calls the same compiled FFT, and its
    // extra lanes only take part in the barriers, because the header's cooperative radix-2 strides by the block size and would race.
    const std::string log_s = std::to_string(log_n);
    if (pow2) {
        src += "    for (int i = lane; lane < " + per + " && i < " + n + "; i += " + per + ") {\n";
        src += "        const int j = wp::fft_bitrev(i, " + log_s + ");\n";
        src += "        if (i < j) {\n            const Complex tmp = row[i];\n            row[i] = row[j];\n            row[j] = tmp;\n        }\n";
        src += "    }\n    WP_TILE_SYNC();\n";
        src += "    for (int len = 2; len <= " + n + "; len <<= 1) {\n";
        src += "        const int half = len >> 1;\n";
        src += "        const T theta_base = T(" + sign + ") * wp::fft_consts<T>::two_pi() / T(len);\n";
        src += "        for (int b = lane; lane < " + per + " && b < " + std::to_string(size / 2) + "; b += " + per + ") {\n";
        src += "            const int group = b / half;\n            const int pos = b - group * half;\n";
        src += "            const int lo = group * len + pos;\n            const int hi = lo + half;\n";
        src += "            const T theta = theta_base * T(pos);\n            const T w_re = wp::cos(theta);\n            const T w_im = wp::sin(theta);\n";
        src += "            const T x_re = row[hi].c[0];\n            const T x_im = row[hi].c[1];\n";
        src += "            const T t_re = w_re * x_re - w_im * x_im;\n            const T t_im = w_re * x_im + w_im * x_re;\n";
        src += "            const T u_re = row[lo].c[0];\n            const T u_im = row[lo].c[1];\n";
        src += "            row[lo].c[0] = u_re + t_re;\n            row[lo].c[1] = u_im + t_im;\n";
        src += "            row[hi].c[0] = u_re - t_re;\n            row[hi].c[1] = u_im - t_im;\n        }\n";
        src += "        WP_TILE_SYNC();\n    }\n";
        src += "    const Complex* result = row;\n";
    } else {
        src += "    Complex* transformed = row + " + n + ";\n    Complex* twiddle = transformed + " + n + ";\n";
        src += "    for (int j = lane; lane < " + per + " && j < " + n + "; j += " + per + ") {\n";
        src += "        const T angle = T(" + sign + ") * wp::fft_consts<T>::two_pi() * T(j) / T(" + n + ");\n";
        src += "        twiddle[j].c[0] = wp::cos(angle);\n        twiddle[j].c[1] = wp::sin(angle);\n    }\n";
        src += "    WP_TILE_SYNC();\n";
        src += "    for (int k = lane; lane < " + per + " && k < " + n + "; k += " + per + ") {\n";
        src += "        T re = T(0);\n        T im = T(0);\n";
        src += "        for (int m = 0; m < " + n + "; ++m) {\n";
        src += "            const Complex w = twiddle[(k * m) % " + n + "];\n";
        src += "            re += row[m].c[0] * w.c[0] - row[m].c[1] * w.c[1];\n";
        src += "            im += row[m].c[0] * w.c[1] + row[m].c[1] * w.c[0];\n        }\n";
        src += "        transformed[k].c[0] = re;\n        transformed[k].c[1] = im;\n    }\n";
        src += "    WP_TILE_SYNC();\n    const Complex* result = transformed;\n";
    }
    src += "    if (lane < " + per + ")\n        for (int j = 0; j < " + ept + "; ++j)\n            data[j] = result[lane + j * " + per
        + "];\n";
    src += "    WP_TILE_SYNC();\n}\n";

    std::vector<std::string> options =
        hip_compile_options(gfx_arch, include_dirs[0], num_include_dirs - 1, include_dirs + 1, false, 3, false, false, true, false);
    options.push_back("-fgpu-rdc");
    std::vector<char> bitcode;
    const std::string name = std::string(symbol_name) + ".hip";
    if (!hip_rtc_bitcode(src, name.c_str(), options, bitcode, true))
        return false;
    return write_file(bitcode.data(), bitcode.size(), ltoir_output_path, "wb");
}

// Generates the dense solvers the way cuSOLVERDx's are declared to Warp (builtins.py, tile_cholesky.h, tile_solve.h). They work on
// a block's shared-memory tiles, each matrix dense in its own arrangement (row-major 1, column-major 0), cooperatively across
// num_threads lanes.
//   potrf (2) is `void symbol(T* a, int* info)`. It factors the fill triangle of the M x M matrix a in place (lower gives
//             a = L L^T and upper gives a = U^T U) and leaves the other triangle alone for the caller to zero. As in LAPACK, info is
//             the first column, 1-based, whose pivot is not positive.
//   potrs (3) is `void symbol(T* a, T* b)`. It takes the factor potrf left in a and solves L L^T x = b (or U^T U x = b) in place,
//             with K right-hand sides in the M x K matrix b.
//   trsm  (4) is `void symbol(T* a, T* b)`. It solves a x = b in place for an M x M triangular a (fill side, unit diagonal or not)
//             and an M x N matrix b, with a on the left only.
// Warp's own call sites never ask for getrf or a right-side solve, so those are refused rather than computed some other way.
// The universal fatbin that cuSOLVERDx also returns has no meaning on AMD GPUs, so an empty one is written to keep Warp's LTO cache
// complete. build.py does not link it.
bool wp_cuda_compile_solver(
    const char* fatbin_output_path,
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int function,
    int side,
    int diag,
    int precision,
    int arrangement_A,
    int arrangement_B,
    int fill_mode,
    int num_threads
)
{
    (void)arch;
    const char* gfx_arch = mathdx_include_dir;
    if (!fatbin_output_path || !ltoir_output_path || !symbol_name || !gfx_arch || !gfx_arch[0] || num_include_dirs < 1
        || !include_dirs || !include_dirs[0]) {
        fprintf(stderr, "Warp HIP device library: a solver needs its outputs, symbol, target and Warp's headers\n");
        return false;
    }
    const bool known = (function == 2 || function == 3 || (function == 4 && side == 0)) && (precision == 5 || precision == 6)
        && M > 0 && N > 0 && K > 0 && num_threads > 0 && (fill_mode == 0 || fill_mode == 1);
    if (!known) {
        fprintf(stderr, "Warp HIP device library: no solver of this kind (%s)\n", symbol_name);
        return false;
    }
    const std::string T = precision == 5 ? "wp::float32" : "wp::float64";
    const std::string m = std::to_string(M);
    const std::string lanes = std::to_string(num_threads);
    // The number of columns of b, which is K for potrs and N for trsm.
    const int rhs = function == 3 ? K : N;
    const std::string r = std::to_string(rhs);
    // Element (i, j) of a and of b, which are dense in their own arrangements.
    const std::string a_at = arrangement_A == 1 ? "a[(i) * " + m + " + (j)]" : "a[(i) + (j) * " + m + "]";
    const std::string b_at = arrangement_B == 1 ? "b[(i) * " + r + " + (j)]" : "b[(i) + (j) * " + m + "]";
    const bool upper = fill_mode == 0;

    std::string src;
    src += "#define WP_TILE_BLOCK_DIM " + lanes + "\n#define WP_NO_CRT\n#include \"builtin.h\"\n\n";
    src += "#define A_AT(i, j) " + a_at + "\n#define B_AT(i, j) " + b_at + "\n";
    // The lower factor's element in either triangle, using L(i, j) = U(j, i).
    src += upper ? "#define L_AT(i, j) A_AT(j, i)\n" : "#define L_AT(i, j) A_AT(i, j)\n";
    src += "using T = " + T + ";\n\n";
    const std::string lane = "const int lane = WP_TILE_THREAD_IDX;\n";
    if (function == 2) {
        src += std::string("extern \"C\" __device__ void ") + symbol_name + "(T* a, int* info)\n{\n    " + lane;
        src += "    for (int j = 0; j < " + m + "; ++j) {\n";
        src += "        if (lane == 0) {\n            T d = L_AT(j, j);\n";
        src += "            for (int k = 0; k < j; ++k)\n                d -= L_AT(j, k) * L_AT(j, k);\n";
        src += "            if (!(d > T(0)) && info[0] == 0)\n                info[0] = j + 1;\n";
        src += "            L_AT(j, j) = wp::sqrt(d);\n        }\n        WP_TILE_SYNC();\n";
        src += "        const T inv = T(1) / L_AT(j, j);\n";
        src += "        for (int i = j + 1 + lane; i < " + m + "; i += " + lanes + ") {\n";
        src += "            T s = L_AT(i, j);\n            for (int k = 0; k < j; ++k)\n                s -= L_AT(i, k) * L_AT(j, k);\n";
        src += "            L_AT(i, j) = s * inv;\n        }\n        WP_TILE_SYNC();\n    }\n}\n";
    } else {
        // Forward substitution for a lower triangle, which computes row j's x from column j and then updates every row below it,
        // and backward substitution for an upper one. potrs runs both in turn, on L and then L^T.
        auto forward = [&](const std::string& at, bool unit) {
            std::string s;
            s += "    for (int j = 0; j < " + m + "; ++j) {\n";
            if (!unit) {
                s += "        for (int c = lane; c < " + r + "; c += " + lanes + ")\n            B_AT(j, c) = B_AT(j, c) / " + at
                    + "(j, j);\n        WP_TILE_SYNC();\n";
            }
            s += "        for (int e = lane; e < (" + m + " - j - 1) * " + r + "; e += " + lanes + ") {\n";
            s += "            const int i = j + 1 + e / " + r + ";\n            const int c = e % " + r + ";\n";
            s += "            B_AT(i, c) -= " + at + "(i, j) * B_AT(j, c);\n        }\n        WP_TILE_SYNC();\n    }\n";
            return s;
        };
        auto backward = [&](const std::string& at, bool unit) {
            std::string s;
            s += "    for (int j = " + m + " - 1; j >= 0; --j) {\n";
            if (!unit) {
                s += "        for (int c = lane; c < " + r + "; c += " + lanes + ")\n            B_AT(j, c) = B_AT(j, c) / " + at
                    + "(j, j);\n        WP_TILE_SYNC();\n";
            }
            s += "        for (int e = lane; e < j * " + r + "; e += " + lanes + ") {\n";
            s += "            const int i = e / " + r + ";\n            const int c = e % " + r + ";\n";
            s += "            B_AT(i, c) -= " + at + "(i, j) * B_AT(j, c);\n        }\n        WP_TILE_SYNC();\n    }\n";
            return s;
        };
        src += "#define LT_AT(i, j) L_AT(j, i)\n";
        src += std::string("extern \"C\" __device__ void ") + symbol_name + "(T* a, T* b)\n{\n    " + lane;
        if (function == 3) {
            src += forward("L_AT", false);
            src += backward("LT_AT", false);
        } else if (upper) {
            src += backward("A_AT", diag == 0);
        } else {
            src += forward("A_AT", diag == 0);
        }
        src += "}\n";
    }

    std::vector<std::string> options =
        hip_compile_options(gfx_arch, include_dirs[0], num_include_dirs - 1, include_dirs + 1, false, 3, false, false, true, false);
    options.push_back("-fgpu-rdc");
    std::vector<char> bitcode;
    const std::string name = std::string(symbol_name) + ".hip";
    if (!hip_rtc_bitcode(src, name.c_str(), options, bitcode, true))
        return false;
    if (!write_file(nullptr, 0, fatbin_output_path, "wb"))
        return false;
    return write_file(bitcode.data(), bitcode.size(), ltoir_output_path, "wb");
}
#endif

size_t wp_cuda_compile_program(
    const char* cuda_src,
    const char* program_name,
    int arch,
    const char* arch_suffix,
    const char* include_dir,
    int num_cuda_include_dirs,
    const char** cuda_include_dirs,
    bool debug,
    int optimization_level,
    bool verbose,
    bool verify_fp,
    bool fast_math,
    bool fuse_fp,
    bool lineinfo,
    bool compile_time_trace,
    bool precompiled_headers,
    const char* output_path,
    const char* pch_dir,
    size_t num_ltoirs,
    char** ltoirs,
    size_t* ltoir_sizes,
    int* ltoir_input_types
)
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)ltoir_input_types;  // Every input of the HIP device library is AMD bitcode.
    (void)pch_dir;

    if (!cuda_src || !program_name || !output_path) {
        fprintf(stderr, "Warp HIP error: Invalid arguments to wp_cuda_compile_program\n");
        return size_t(-1);
    }

    // hipRTC has no precompiled headers. Warp asks for them by default, so the option is silently ignored.
    (void)precompiled_headers;
    // A module that calls the HIP device library is compiled as relocatable code and linked with the library's bitcode
    // (hip_link_bitcode), the same way a CUDA module is linked with MathDx's LTO-IR.
    const bool link_library = num_ltoirs > 0;
    if (link_library && (ltoirs == nullptr || ltoir_sizes == nullptr)) {
        fprintf(stderr, "Warp HIP error: %zu library inputs named but not given\n", num_ltoirs);
        return size_t(-1);
    }

    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, cuda_src, program_name, 0, nullptr, nullptr);
    if (!check_nvrtc(res))
        return size_t(res);

    // On HIP, the CUDA-style numeric `arch` is unused; the full gfx target string
    // (e.g. "gfx942") is passed in via `arch_suffix`. See the wp_cuda_compile_program
    // declaration in warp.h for the calling convention.
    (void)arch;
    // Build every string before taking any pointer, so no c_str() is invalidated by a later push.
    std::vector<std::string> stored_options = hip_compile_options(
        arch_suffix, include_dir, num_cuda_include_dirs, cuda_include_dirs, debug, optimization_level, verify_fp,
        fast_math, fuse_fp, lineinfo
    );
    if (link_library)
        stored_options.push_back("-fgpu-rdc");
    std::vector<const char*> opts;
    for (const std::string& option : stored_options)
        opts.push_back(option.c_str());

    if (verbose) {
        fprintf(stdout, "HIPRTC options:\n");
        for (auto option : opts)
            fprintf(stdout, "%s\n", option);
    }

    std::chrono::steady_clock::time_point compile_start;
    if (compile_time_trace)
        compile_start = std::chrono::steady_clock::now();

    res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());

    if (compile_time_trace) {
        const auto compile_end = std::chrono::steady_clock::now();
        const double compile_ms = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
        const std::string trace_path = std::string(output_path) + "_compile-time-trace.json";
        const std::string trace = std::string("{\n")
            + "  \"tool\": \"hiprtc\",\n"
            + "  \"compile_ms\": " + std::to_string(compile_ms) + ",\n"
            + "  \"result\": " + std::to_string(static_cast<int>(res)) + "\n"
            + "}\n";
        if (!write_file(trace.data(), trace.size(), trace_path, "wb")) {
            fprintf(stderr, "Warp warning: Failed to write HIPRTC compile_time_trace to '%s'\n", trace_path.c_str());
        }
    }

    if (res != NVRTC_SUCCESS || verbose) {
        size_t log_size = 0;
        if (check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size)) && log_size > 0) {
            std::vector<char> log(log_size);
            if (check_nvrtc(nvrtcGetProgramLog(prog, log.data()))) {
                if (res != NVRTC_SUCCESS)
                    fprintf(stderr, "%s", log.data());
                else
                    fprintf(stdout, "%s", log.data());
            }
        }

        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            return size_t(res);
        }
    }

    if (link_library) {
        // Link the module's relocatable bitcode with the library's into the module's code object.
        size_t bitcode_size = 0;
        std::vector<char> bitcode;
        bool ok = hiprtcGetBitcodeSize(prog, &bitcode_size) == HIPRTC_SUCCESS && bitcode_size > 0;
        if (ok) {
            bitcode.resize(bitcode_size);
            ok = hiprtcGetBitcode(prog, bitcode.data()) == HIPRTC_SUCCESS;
        }
        check_nvrtc(nvrtcDestroyProgram(&prog));
        std::vector<char> linked;
        if (!ok || !hip_link_bitcode(bitcode, num_ltoirs, ltoirs, ltoir_sizes, debug ? 0 : optimization_level, linked))
            return size_t(-1);
        if (!write_file(linked.data(), linked.size(), output_path, "wb"))
            return size_t(-1);
        return 0;
    }

    size_t code_size = 0;
    if (!check_nvrtc(nvrtcGetPTXSize(prog, &code_size))) {
        nvrtcDestroyProgram(&prog);
        return size_t(-1);
    }

    std::vector<char> code(code_size);
    if (!check_nvrtc(nvrtcGetPTX(prog, code.data()))) {
        nvrtcDestroyProgram(&prog);
        return size_t(-1);
    }

    if (!write_file(code.data(), code.size(), output_path, "wb")) {
        nvrtcDestroyProgram(&prog);
        return size_t(-1);
    }

    check_nvrtc(nvrtcDestroyProgram(&prog));
    return res;
#else
    // use file extension to determine whether to output PTX or CUBIN
    const char* output_ext = strrchr(output_path, '.');
    bool use_ptx = output_ext && strcmp(output_ext + 1, "ptx") == 0;
    const bool print_debug = (std::getenv("WARP_DEBUG") != nullptr);

    // check include dir path len (path + option)
    const int max_path = 4096 + 16;
    if (strlen(include_dir) > max_path) {
        fprintf(stderr, "Warp error: Include path too long\n");
        return size_t(-1);
    }

    if (print_debug) {
        // Not available in all nvJitLink versions
        // unsigned major = 0;
        // unsigned minor = 0;
        // nvJitLinkVersion(&major, &minor);
        // printf("nvJitLink version %d.%d\n", major, minor);
        int major = 0;
        int minor = 0;
        nvrtcVersion(&major, &minor);
        printf("NVRTC version %d.%d\n", major, minor);
    }

    char include_opt[max_path];
    strcpy(include_opt, "--include-path=");
    strcat(include_opt, include_dir);

    const int max_arch = 128;
    char arch_opt[max_arch];
    char arch_opt_lto[max_arch];

    // arch_suffix is "" (no suffix), "a" (arch-specific), or "f" (family-specific)
    const char* suffix = (arch_suffix != nullptr) ? arch_suffix : "";

    if (use_ptx) {
        snprintf(arch_opt, max_arch, "--gpu-architecture=compute_%d%s", arch, suffix);
        snprintf(arch_opt_lto, max_arch, "-arch=compute_%d%s", arch, suffix);
    } else {
        snprintf(arch_opt, max_arch, "--gpu-architecture=sm_%d%s", arch, suffix);
        snprintf(arch_opt_lto, max_arch, "-arch=sm_%d%s", arch, suffix);
    }

    std::vector<const char*> opts;
    opts.push_back(arch_opt);
    opts.push_back(include_opt);

    // Vector to store dynamically created option strings
    std::vector<std::string> stored_options;
    stored_options.reserve(static_cast<size_t>(num_cuda_include_dirs) + 2);

    if (precompiled_headers) {
        // CUDA 12.8+ supports precompiled headers
#if CUDA_VERSION >= 12080
        opts.push_back("-pch");
#if CUDA_VERSION < 13000
        // CUDA 12.x series puts .pch files in the current working directory unless explicitly set
        if (pch_dir != nullptr) {
            if (print_debug) {
                printf("PCH directory: %s\n", pch_dir);
            }
            std::string pch_dir_opt = std::string("--pch-dir=") + pch_dir;
            stored_options.push_back(pch_dir_opt);
            opts.push_back(stored_options.back().c_str());
        }
#endif
#endif
    }

    append_common_nvrtc_compile_options(
        opts, use_ptx, arch, debug, optimization_level, verify_fp, fast_math, fuse_fp, lineinfo
    );

    for (int i = 0; i < num_cuda_include_dirs; i++) {
        stored_options.push_back(std::string("--include-path=") + cuda_include_dirs[i]);
        opts.push_back(stored_options.back().c_str());
    }

    if (num_ltoirs > 0) {
        opts.push_back("-dlto");
        opts.push_back("--relocatable-device-code=true");
    }

    if (compile_time_trace) {
#if CUDA_VERSION >= 12080
        stored_options.push_back(
            std::string("--fdevice-time-trace=") + std::string(output_path).append("_compile-time-trace.json")
        );
        opts.push_back(stored_options.back().c_str());
#else
        fprintf(stderr, "Warp warning: CUDA version is less than 12.8, compile_time_trace is not supported\n");
#endif
    }

    nvrtcProgram prog;
    nvrtcResult res;

    res = nvrtcCreateProgram(
        &prog,  // prog
        cuda_src,  // buffer
        program_name,  // name
        0,  // numHeaders
        NULL,  // headers
        NULL
    );  // includeNames

    if (!check_nvrtc(res))
        return size_t(res);

    if (print_debug) {
        printf("NVRTC options:\n");
        for (auto o : opts) {
            printf("%s\n", o);
        }
    }
    res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());

    if (!check_nvrtc(res) || verbose) {
        // get program log
        size_t log_size;
        if (check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size))) {
            std::vector<char> log(log_size);
            if (check_nvrtc(nvrtcGetProgramLog(prog, log.data()))) {
                // todo: figure out better way to return this to python
                if (res != NVRTC_SUCCESS)
                    fprintf(stderr, "%s", log.data());
                else
                    fprintf(stdout, "%s", log.data());
            }
        }

        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            return size_t(res);
        }
    }

    nvrtcResult (*get_output_size)(nvrtcProgram, size_t*);
    nvrtcResult (*get_output_data)(nvrtcProgram, char*);
    const char* output_mode;
    if (num_ltoirs > 0) {
#if WP_ENABLE_MATHDX
        get_output_size = nvrtcGetLTOIRSize;
        get_output_data = nvrtcGetLTOIR;
        output_mode = "wb";
#else
        fprintf(stderr, "Warp error: num_ltoirs > 0 but Warp was not built with MathDx support\n");
        return size_t(-1);
#endif
    } else if (use_ptx) {
        get_output_size = nvrtcGetPTXSize;
        get_output_data = nvrtcGetPTX;
        output_mode = "wt";
    } else {
        get_output_size = nvrtcGetCUBINSize;
        get_output_data = nvrtcGetCUBIN;
        output_mode = "wb";
    }

    // save output
    size_t output_size;
    res = get_output_size(prog, &output_size);
    if (check_nvrtc(res)) {
        std::vector<char> output(output_size);
        res = get_output_data(prog, output.data());
        if (check_nvrtc(res)) {

            // LTOIR case - need an extra step
            if (num_ltoirs > 0) {
#if WP_ENABLE_MATHDX
                if (ltoir_input_types == nullptr || ltoirs == nullptr || ltoir_sizes == nullptr) {
                    fprintf(
                        stderr, "Warp error: num_ltoirs > 0 but ltoir_input_types, ltoirs or ltoir_sizes are NULL\n"
                    );
                    return size_t(-1);
                }
                nvJitLinkHandle handle = nullptr;
                std::vector<const char*> lopts = { "-dlto", arch_opt_lto };
                if (use_ptx) {
                    lopts.push_back("-ptx");
                }
                if (print_debug) {
                    printf("nvJitLink options:\n");
                    for (auto o : lopts) {
                        printf("%s\n", o);
                    }
                }
                if (!check_nvjitlink(handle, nvJitLinkCreate(&handle, lopts.size(), lopts.data()))) {
                    res = nvrtcResult(-1);
                }
                // Links
                if (std::getenv("WARP_DUMP_LTOIR")) {
                    write_file(output.data(), output.size(), "nvrtc_output.ltoir", "wb");
                }
                if (!check_nvjitlink(
                        handle,
                        nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, output.data(), output.size(), "nvrtc_output")
                    ))  // NVRTC business
                {
                    res = nvrtcResult(-1);
                }
                for (size_t ltoidx = 0; ltoidx < num_ltoirs; ltoidx++) {
                    nvJitLinkInputType input_type = static_cast<nvJitLinkInputType>(ltoir_input_types[ltoidx]);
                    const char* ext = ".unknown";
                    switch (input_type) {
                    case NVJITLINK_INPUT_CUBIN:
                        ext = ".cubin";
                        break;
                    case NVJITLINK_INPUT_LTOIR:
                        ext = ".ltoir";
                        break;
                    case NVJITLINK_INPUT_FATBIN:
                        ext = ".fatbin";
                        break;
                    default:
                        break;
                    }
                    if (std::getenv("WARP_DUMP_LTOIR")) {
                        write_file(
                            ltoirs[ltoidx], ltoir_sizes[ltoidx],
                            std::string("lto_online_") + std::to_string(ltoidx) + ext, "wb"
                        );
                    }
                    if (!check_nvjitlink(
                            handle,
                            nvJitLinkAddData(handle, input_type, ltoirs[ltoidx], ltoir_sizes[ltoidx], "lto_online")
                        ))  // External LTOIR
                    {
                        res = nvrtcResult(-1);
                    }
                }
                if (!check_nvjitlink(handle, nvJitLinkComplete(handle))) {
                    res = nvrtcResult(-1);
                } else {
                    if (use_ptx) {
                        size_t ptx_size = 0;
                        check_nvjitlink(handle, nvJitLinkGetLinkedPtxSize(handle, &ptx_size));
                        std::vector<char> ptx(ptx_size);
                        check_nvjitlink(handle, nvJitLinkGetLinkedPtx(handle, ptx.data()));
                        output = ptx;
                    } else {
                        size_t cubin_size = 0;
                        check_nvjitlink(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
                        std::vector<char> cubin(cubin_size);
                        check_nvjitlink(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
                        output = cubin;
                    }
                }
                check_nvjitlink(handle, nvJitLinkDestroy(&handle));
#else
                fprintf(stderr, "Warp error: num_ltoirs > 0 but Warp was not built with MathDx support\n");
                return size_t(-1);
#endif
            }

            if (!write_file(output.data(), output.size(), output_path, output_mode)) {
                res = nvrtcResult(-1);
            }
        }
    }

    check_nvrtc(nvrtcDestroyProgram(&prog));

    return res;
#endif
}

#if WP_ENABLE_MATHDX
bool check_cufftdx_result(commondxStatusType result, const char* file, int line)
{
    if (result != commondxStatusType::COMMONDX_SUCCESS) {
        fprintf(stderr, "libmathdx cuFFTDx error: %d on %s:%d\n", (int)result, file, line);
        return false;
    } else {
        return true;
    }
}

bool check_cublasdx_result(commondxStatusType result, const char* file, int line)
{
    if (result != commondxStatusType::COMMONDX_SUCCESS) {
        fprintf(stderr, "libmathdx cuBLASDx error: %d on %s:%d\n", (int)result, file, line);
        return false;
    } else {
        return true;
    }
}

bool check_cusolver_result(commondxStatusType result, const char* file, int line)
{
    if (result != commondxStatusType::COMMONDX_SUCCESS) {
        fprintf(stderr, "libmathdx cuSOLVER error: %d on %s:%d\n", (int)result, file, line);
        return false;
    } else {
        return true;
    }
}

bool wp_cuda_compile_fft(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int size,
    int elements_per_thread,
    int direction,
    int precision,
    int* shared_memory_size
)
{

    CHECK_ANY(ltoir_output_path != nullptr);
    CHECK_ANY(symbol_name != nullptr);
    CHECK_ANY(shared_memory_size != nullptr);
    // Includes currently unused
    CHECK_ANY(include_dirs == nullptr);
    CHECK_ANY(mathdx_include_dir == nullptr);
    CHECK_ANY(num_include_dirs == 0);

    bool res = true;
    cufftdxDescriptor h;
    CHECK_CUFFTDX(cufftdxCreateDescriptor(&h));

    // CUFFTDX_API_LMEM means each thread starts with a subset of the data
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_LMEM));
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(
        h, cufftdxOperatorType::CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK
    ));
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_SIZE, (long long)size));
    CHECK_CUFFTDX(
        cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_DIRECTION, (cufftdxDirection)direction)
    );
    CHECK_CUFFTDX(
        cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_PRECISION, (commondxPrecision)precision)
    );
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_SM, (long long)(arch * 10)));
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(
        h, cufftdxOperatorType::CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, (long long)(elements_per_thread)
    ));
    CHECK_CUFFTDX(cufftdxSetOperatorInt64(h, cufftdxOperatorType::CUFFTDX_OPERATOR_FFTS_PER_BLOCK, 1));

    CHECK_CUFFTDX(cufftdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

    size_t lto_size = 0;
    CHECK_CUFFTDX(cufftdxGetLTOIRSize(h, &lto_size));

    std::vector<char> lto(lto_size);
    CHECK_CUFFTDX(cufftdxGetLTOIR(h, lto.size(), lto.data()));

    long long int smem = 0;
    CHECK_CUFFTDX(cufftdxGetTraitInt64(h, cufftdxTraitType::CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &smem));
    *shared_memory_size = (int)smem;

    if (!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
        res = false;
    }

    CHECK_CUFFTDX(cufftdxDestroyDescriptor(h));

    return res;
}

bool wp_cuda_compile_dot(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int precision_A,
    int precision_B,
    int precision_C,
    int type,
    int arrangement_A,
    int arrangement_B,
    int arrangement_C,
    int num_threads,
    int lda,
    int ldb,
    int ldc,
    int alignment_A,
    int alignment_B,
    int alignment_C,
    int enable_static_block_dim,
    int suppress_errors
)
{

    CHECK_ANY(ltoir_output_path != nullptr);
    CHECK_ANY(symbol_name != nullptr);
    // Includes currently unused
    CHECK_ANY(include_dirs == nullptr);
    CHECK_ANY(mathdx_include_dir == nullptr);
    CHECK_ANY(num_include_dirs == 0);

    bool res = true;
    cublasdxDescriptor h;
    CHECK_CUBLASDX(cublasdxCreateDescriptor(&h));

    CHECK_CUBLASDX(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_MM
    ));
    CHECK_CUBLASDX(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK
    ));
    CHECK_CUBLASDX(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_SMEM)
    );
    std::array<long long int, 3> precisions = { precision_A, precision_B, precision_C };
    CHECK_CUBLASDX(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, 3, precisions.data())
    );
    CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SM, (long long)(arch * 10)));
    CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_TYPE, (cublasdxType)type));
    std::array<long long int, 3> block_dim = { num_threads, 1, 1 };
    CHECK_CUBLASDX(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()
    ));
    std::array<long long int, 3> size = { M, N, K };
    CHECK_CUBLASDX(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data())
    );
    std::array<long long int, 3> arrangement = { arrangement_A, arrangement_B, arrangement_C };
    CHECK_CUBLASDX(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_ARRANGEMENT, arrangement.size(), arrangement.data()
    ));

    // non-positive values leave the leading dimensions at their dense defaults
    if (lda > 0 && ldb > 0 && ldc > 0) {
        std::array<long long int, 3> ld = { lda, ldb, ldc };
        CHECK_CUBLASDX(cublasdxSetOperatorInt64s(
            h, cublasdxOperatorType::CUBLASDX_OPERATOR_LEADING_DIMENSION, ld.size(), ld.data()
        ));
    }

    // any non-positive value leaves cuBLASDx at its default (each value type's natural alignment)
    if (alignment_A > 0 && alignment_B > 0 && alignment_C > 0) {
        std::array<long long int, 3> align = { alignment_A, alignment_B, alignment_C };
        CHECK_CUBLASDX(
            cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_ALIGNMENT, align.size(), align.data())
        );
    }

    if (enable_static_block_dim) {
        CHECK_CUBLASDX(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_STATIC_BLOCK_DIM, 1));
    }

    CHECK_CUBLASDX(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

    // generating the LTO is where cuBLASDx compiles the GEMM; a rejected configuration is
    // expected when the caller probes an alignment it can fall back from, so do not report it then
    size_t lto_size = 0;
    commondxStatusType lto_status = cublasdxGetLTOIRSize(h, &lto_size);
    if (lto_status != commondxStatusType::COMMONDX_SUCCESS) {
        if (!suppress_errors) {
            check_cublasdx(lto_status);
        }
        cublasdxDestroyDescriptor(h);
        return false;
    }

    std::vector<char> lto(lto_size);
    CHECK_CUBLASDX(cublasdxGetLTOIR(h, lto.size(), lto.data()));

    if (!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
        res = false;
    }

    CHECK_CUBLASDX(cublasdxDestroyDescriptor(h));

    return res;
}

bool wp_cuda_compile_solver(
    const char* fatbin_output_path,
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int function,
    int side,
    int diag,
    int precision,
    int arrangement_A,
    int arrangement_B,
    int fill_mode,
    int num_threads
)
{

    CHECK_ANY(ltoir_output_path != nullptr);
    CHECK_ANY(symbol_name != nullptr);
    CHECK_ANY(mathdx_include_dir == nullptr);
    CHECK_ANY(num_include_dirs == 0);
    CHECK_ANY(include_dirs == nullptr);

    bool res = true;

    cusolverdxDescriptor h { 0 };
    CHECK_CUSOLVER(cusolverdxCreateDescriptor(&h));
    std::array<long long int, 3> size = { M, N, K };
    CHECK_CUSOLVER(
        cusolverdxSetOperatorInt64s(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_SIZE, size.size(), size.data())
    );
    std::array<long long int, 3> block_dim = { num_threads, 1, 1 };
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64s(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()
    ));
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_TYPE, cusolverdxType::CUSOLVERDX_TYPE_REAL
    ));
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_API, cusolverdxApi::CUSOLVERDX_API_SMEM
    ));
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_FUNCTION, (cusolverdxFunction)function
    ));
    if (side >= 0) {
        CHECK_CUSOLVER(
            cusolverdxSetOperatorInt64(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_SIDE, (cusolverdxSide)side)
        );
    }
    if (diag >= 0) {
        CHECK_CUSOLVER(
            cusolverdxSetOperatorInt64(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_DIAG, (cusolverdxDiag)diag)
        );
    }
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK
    ));
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_PRECISION, (commondxPrecision)precision
    ));
    std::array<long long int, 2> arrangement = { arrangement_A, arrangement_B };
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64s(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_ARRANGEMENT, arrangement.size(), arrangement.data()
    ));
    CHECK_CUSOLVER(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_FILL_MODE, (cusolverdxFillMode)fill_mode
    ));
    CHECK_CUSOLVER(
        cusolverdxSetOperatorInt64(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_SM, (long long)(arch * 10))
    );

    CHECK_CUSOLVER(cusolverdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name));

    size_t lto_size = 0;
    CHECK_CUSOLVER(cusolverdxGetLTOIRSize(h, &lto_size));

    std::vector<char> lto(lto_size);
    CHECK_CUSOLVER(cusolverdxGetLTOIR(h, lto.size(), lto.data()));

    // This fatbin is universal, ie it is the same for any instantiations of a cusolver device function
    size_t fatbin_size = 0;
    CHECK_CUSOLVER(cusolverdxGetUniversalFATBINSize(h, &fatbin_size));

    std::vector<char> fatbin(fatbin_size);
    CHECK_CUSOLVER(cusolverdxGetUniversalFATBIN(h, fatbin.size(), fatbin.data()));

    if (!write_file(lto.data(), lto.size(), ltoir_output_path, "wb")) {
        res = false;
    }

    if (!write_file(fatbin.data(), fatbin.size(), fatbin_output_path, "wb")) {
        res = false;
    }

    CHECK_CUSOLVER(cusolverdxDestroyDescriptor(h));

    return res;
}

#endif

#if !defined(__HIP_PLATFORM_AMD__)
static const char* apic_nvptx_error_string(nvPTXCompileResult result)
{
    switch (result) {
    case NVPTXCOMPILE_SUCCESS:
        return "success";
    case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
        return "invalid compiler handle";
    case NVPTXCOMPILE_ERROR_INVALID_INPUT:
        return "invalid input";
    case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
        return "compilation failure";
    case NVPTXCOMPILE_ERROR_INTERNAL:
        return "internal error";
    case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
        return "incomplete compiler invocation";
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
        return "unsupported PTX version";
    default:
        return "unknown error";
    }
}
#endif

static std::string apic_cuda_error_string(CUresult result)
{
    const char* name = nullptr;
    const char* description = nullptr;
    ignore_cu_result(cuGetErrorName_f(result, &name));
    ignore_cu_result(cuGetErrorString_f(result, &description));
    std::string message = name ? name : "CUDA_ERROR_UNKNOWN";
    if (description) {
        message += ": ";
        message += description;
    }
    return message;
}

// Load PTX or CUBIN bytes without printing. APIC uses this primitive to try
// packaged and cached candidates before reporting one combined diagnostic.
static bool apic_cuda_load_module_data(
    void* context, const uint8_t* data, size_t size, bool load_ptx, CUmodule* module, std::string& diagnostic
)
{
    if (!data || size == 0 || !module) {
        diagnostic = "module data is empty";
        return false;
    }

    ContextGuard guard(context);
    *module = nullptr;

    std::vector<uint8_t> terminated_input;
    const void* input = data;
    size_t input_size = size;
    if (load_ptx && data[size - 1] != 0) {
        terminated_input.assign(data, data + size);
        terminated_input.push_back(0);
        input = terminated_input.data();
        input_size = terminated_input.size();
    }

    if (load_ptx) {
#if defined(__HIP_PLATFORM_AMD__)
        // 1.19's APIC loader treats a module image as a HIP code object on this platform. There is no PTX to JIT or compile.
        CUresult hip_result = cuModuleLoadDataEx_f(module, input, 0, nullptr, nullptr);
        if (hip_result != CUDA_SUCCESS) {
            diagnostic = apic_cuda_error_string(hip_result) + " (PTX is not supported on HIP; a module here is a code object)";
            return false;
        }
        hip_print_attach((hipModule_t)*module);
        return true;
#else
        int driver_cuda_version = 0;
        CUresult driver_version_result = cuDriverGetVersion_f(&driver_cuda_version);
        if (driver_version_result == CUDA_SUCCESS && driver_cuda_version >= CUDA_VERSION) {
            CUjit_option options[2];
            void* option_values[2];
            char error_log[8192] = "";
            options[0] = CU_JIT_ERROR_LOG_BUFFER;
            option_values[0] = error_log;
            options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            option_values[1] = reinterpret_cast<void*>(sizeof(error_log));
            CUresult result = cuModuleLoadDataEx_f(module, input, 2, options, option_values);
            if (result != CUDA_SUCCESS) {
                diagnostic = apic_cuda_error_string(result);
                if (error_log[0]) {
                    diagnostic += "\n";
                    diagnostic += error_log;
                }
                return false;
            }
            return true;
        }

        ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
        if (!context_info || !context_info->device_info) {
            diagnostic = "could not determine the CUDA device architecture";
            return false;
        }

        char arch_option[64];
        snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%d", context_info->device_info->arch);
        const char* compiler_options[] = { arch_option };
        nvPTXCompilerHandle compiler = nullptr;
        nvPTXCompileResult result = nvPTXCompilerCreate(&compiler, input_size, reinterpret_cast<const char*>(input));
        if (result == NVPTXCOMPILE_SUCCESS) {
            result = nvPTXCompilerCompile(compiler, 1, compiler_options);
        }
        if (result != NVPTXCOMPILE_SUCCESS) {
            diagnostic = std::string("nvPTXCompiler: ") + apic_nvptx_error_string(result);
            if (compiler) {
                size_t log_size = 0;
                if (nvPTXCompilerGetErrorLogSize(compiler, &log_size) == NVPTXCOMPILE_SUCCESS && log_size > 1) {
                    std::vector<char> log(log_size);
                    if (nvPTXCompilerGetErrorLog(compiler, log.data()) == NVPTXCOMPILE_SUCCESS) {
                        diagnostic += "\n";
                        diagnostic += log.data();
                    }
                }
                nvPTXCompilerDestroy(&compiler);
            }
            return false;
        }

        size_t cubin_size = 0;
        result = nvPTXCompilerGetCompiledProgramSize(compiler, &cubin_size);
        std::vector<uint8_t> cubin(cubin_size);
        if (result == NVPTXCOMPILE_SUCCESS) {
            result = nvPTXCompilerGetCompiledProgram(compiler, reinterpret_cast<char*>(cubin.data()));
        }
        nvPTXCompilerDestroy(&compiler);
        if (result != NVPTXCOMPILE_SUCCESS) {
            diagnostic = std::string("nvPTXCompiler output: ") + apic_nvptx_error_string(result);
            return false;
        }

        CUresult load_result = cuModuleLoadDataEx_f(module, cubin.data(), 0, nullptr, nullptr);
        if (load_result != CUDA_SUCCESS) {
            diagnostic = apic_cuda_error_string(load_result);
            return false;
        }
        return true;
#endif
    }

    CUresult result = cuModuleLoadDataEx_f(module, input, 0, nullptr, nullptr);
    if (result != CUDA_SUCCESS) {
        diagnostic = apic_cuda_error_string(result);
        return false;
    }
    return true;
}

// Compile generated Warp CUDA source to memory with the semantic options
// captured in the .wrp. External includes, link inputs, PCH, and tracing are
// intentionally excluded from the portable fallback contract.
static bool apic_cuda_compile_program_to_memory(
    const std::string& cuda_source,
    const char* program_name,
    int arch,
    const char* arch_suffix,
    const char* include_dir,
    bool use_ptx,
    const APICCudaCompileRecipe& recipe,
    std::vector<uint8_t>& output,
    std::string& diagnostic
)
{
    if (!include_dir || !include_dir[0]) {
        diagnostic = "the Warp native include directory was not provided";
        return false;
    }
    if (strlen(include_dir) > 4096) {
        diagnostic = "the Warp native include path is too long";
        return false;
    }

#if defined(__HIP_PLATFORM_AMD__)
    // Compile the saved module for this device's gfx target with the options Warp's own compile uses. A HIP module is a code
    // object, never PTX, and an arch suffix such as sm_90a or sm_100f means nothing here.
    if (!(arch & WP_APIC_HIP_ARCH_FLAG) || use_ptx || (arch_suffix && arch_suffix[0])) {
        diagnostic = "a HIP device compiles a saved module only for its own gfx target, as a code object";
        return false;
    }
    const std::string gfx_arch = hip_apic_arch_name(arch);
    const std::vector<std::string> stored_options = hip_compile_options(
        gfx_arch.c_str(), include_dir, 0, nullptr, recipe.debug, recipe.optimization_level, recipe.verify_fp,
        recipe.fast_math, recipe.fuse_fp, recipe.lineinfo
    );
    std::vector<const char*> options;
    for (const std::string& option : stored_options)
        options.push_back(option.c_str());
#else
    std::string arch_option = std::string("--gpu-architecture=") + (use_ptx ? "compute_" : "sm_") + std::to_string(arch)
        + (arch_suffix ? arch_suffix : "");
    std::string include_option = std::string("--include-path=") + include_dir;
    std::vector<std::string> stored_options = { arch_option, include_option };
    std::vector<const char*> options;
    options.push_back(stored_options[0].c_str());
    options.push_back(stored_options[1].c_str());
    append_common_nvrtc_compile_options(
        options, use_ptx, arch, recipe.debug, recipe.optimization_level, recipe.verify_fp, recipe.fast_math,
        recipe.fuse_fp, recipe.lineinfo
    );
#endif

    nvrtcProgram program = nullptr;
    nvrtcResult result = nvrtcCreateProgram(
        &program, cuda_source.c_str(), program_name ? program_name : "apic_module.cu", 0, nullptr, nullptr
    );
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
    }
    if (result != NVRTC_SUCCESS) {
        diagnostic = std::string("NVRTC: ") + nvrtcGetErrorString(result);
        if (program) {
            size_t log_size = 0;
            if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1) {
                std::vector<char> log(log_size);
                if (nvrtcGetProgramLog(program, log.data()) == NVRTC_SUCCESS) {
                    diagnostic += "\n";
                    diagnostic += log.data();
                }
            }
            nvrtcDestroyProgram(&program);
        }
        return false;
    }

#if defined(__HIP_PLATFORM_AMD__)
    // hip_util.h maps nvrtcGetPTX to hiprtcGetCode, which returns the loadable code object, while nvrtcGetCUBIN returns LLVM bitcode
    // there.
    const bool code_through_ptx_names = true;
#else
    const bool code_through_ptx_names = use_ptx;
#endif
    size_t output_size = 0;
    result = code_through_ptx_names ? nvrtcGetPTXSize(program, &output_size) : nvrtcGetCUBINSize(program, &output_size);
    if (result == NVRTC_SUCCESS) {
        output.resize(output_size);
        result = code_through_ptx_names ? nvrtcGetPTX(program, reinterpret_cast<char*>(output.data()))
                                        : nvrtcGetCUBIN(program, reinterpret_cast<char*>(output.data()));
    }
    nvrtcDestroyProgram(&program);
    if (result != NVRTC_SUCCESS) {
        output.clear();
        diagnostic = std::string("NVRTC output: ") + nvrtcGetErrorString(result);
        return false;
    }
    return true;
}

// Whether a module binary's path or recorded filename names a PTX module. The
// file extension decides, matching how Warp names its compiled modules.
static bool cuda_module_path_is_ptx(const char* path)
{
    const char* ext = strrchr(path, '.');
    return ext && strcmp(ext + 1, "ptx") == 0;
}

// Load a CUDA module from an in-memory image. A PTX image must end with a NUL
// byte, which `size` includes. The driver copies the image, so it only needs
// to stay valid for the call.
static void* cuda_load_module_image(void* context, const char* image, size_t size, bool is_ptx)
{
    ContextGuard guard(context);

    int driver_cuda_version = 0;
    CUmodule module = NULL;

    if (is_ptx) {
#if defined(__HIP_PLATFORM_AMD__)
        // On this platform a module image is a HIP code object. 1.19 loads every module from an image, which is the path read
        // before.
        if (!check_cu(cuModuleLoadDataEx_f(&module, image, 0, NULL, NULL))) {
            fprintf(stderr, "Warp error: Failed to load a HIP code object. PTX is not supported on HIP; use HSACO.\n");
            return NULL;
        }
#else
        if (check_cu(cuDriverGetVersion_f(&driver_cuda_version)) && driver_cuda_version >= CUDA_VERSION) {
            // let the driver compile the PTX

            CUjit_option options[2];
            void* option_vals[2];
            char error_log[8192] = "";
            unsigned int log_size = 8192;
            // Set up loader options
            // Pass a buffer for error message
            options[0] = CU_JIT_ERROR_LOG_BUFFER;
            option_vals[0] = (void*)error_log;
            // Pass the size of the error buffer
            options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            option_vals[1] = (void*)(size_t)log_size;

            if (!check_cu(cuModuleLoadDataEx_f(&module, image, 2, options, option_vals))) {
                fprintf(stderr, "Warp error: Loading PTX module failed\n");
                // print error log if not empty
                if (*error_log)
                    fprintf(stderr, "PTX loader error:\n%s\n", error_log);
                return NULL;
            }
        } else {
            // manually compile the PTX and load as CUBIN

            ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
            if (!context_info || !context_info->device_info) {
                fprintf(stderr, "Warp error: Failed to determine target architecture\n");
                return NULL;
            }

            int arch = context_info->device_info->arch;

            char arch_opt[128];
            sprintf(arch_opt, "--gpu-name=sm_%d", arch);

            const char* compiler_options[] = { arch_opt };

            nvPTXCompilerHandle compiler = NULL;
            if (!check_nvptx(nvPTXCompilerCreate(&compiler, size, image)))
                return NULL;

            if (!check_nvptx(nvPTXCompilerCompile(
                    compiler, sizeof(compiler_options) / sizeof(*compiler_options), compiler_options
                )))
                return NULL;

            size_t cubin_size = 0;
            if (!check_nvptx(nvPTXCompilerGetCompiledProgramSize(compiler, &cubin_size)))
                return NULL;

            std::vector<char> cubin(cubin_size);
            if (!check_nvptx(nvPTXCompilerGetCompiledProgram(compiler, cubin.data())))
                return NULL;

            check_nvptx(nvPTXCompilerDestroy(&compiler));

            if (!check_cu(cuModuleLoadDataEx_f(&module, cubin.data(), 0, NULL, NULL))) {
                fprintf(stderr, "Warp CUDA error: Loading module failed\n");
                return NULL;
            }
        }
#endif
    } else {
        // load CUBIN
        if (!check_cu(cuModuleLoadDataEx_f(&module, image, 0, NULL, NULL))) {
            fprintf(stderr, "Warp CUDA error: Loading module failed\n");
            return NULL;
        }
    }

#if defined(__HIP_PLATFORM_AMD__)
    if (module) {
        g_hip_module_vgprs[module] = hip_reserved_vgprs(image, size);
        // Its prints go to the HIP print buffer. CUDA also flushes printf output when a module loads.
        hip_print_flush();
        hip_print_attach((hipModule_t)module);
    }
#endif
    return module;
}

void* wp_cuda_load_module(void* context, const char* path)
{
    std::vector<char> input;

    FILE* file = fopen(path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        size_t length = ftell(file);
        fseek(file, 0, SEEK_SET);

        input.resize(length + 1);
        if (fread(input.data(), 1, length, file) != length) {
            fprintf(stderr, "Warp error: Failed to read input file '%s'\n", path);
            fclose(file);
            return NULL;
        }
        fclose(file);

        input[length] = '\0';
    } else {
        fprintf(stderr, "Warp error: Failed to open input file '%s'\n", path);
        return NULL;
    }

    return cuda_load_module_image(context, input.data(), input.size(), cuda_module_path_is_ptx(path));
}

void wp_cuda_unload_module(void* context, void* module)
{
    release_or_defer(ReleaseKind::Module, context, (uint64_t)module);
}


int wp_cuda_get_max_shared_memory(void* context)
{
    ContextInfo* info = get_context_info(context);
    if (!info)
        return -1;

    int max_smem_bytes = info->device_info->max_smem_bytes;
    return max_smem_bytes;
}

bool wp_cuda_configure_kernel_shared_memory(void* kernel, int size)
{
    int requested_smem_bytes = size;

    // configure shared memory
    CUresult res = cuFuncSetAttribute_f(
        (CUfunction)kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_smem_bytes
    );
    if (res != CUDA_SUCCESS)
        return false;

    return true;
}

static int get_cuda_kernel_attribute(void* context, void* kernel, CUfunction_attribute attribute)
{
    if (!kernel)
        return -1;

    ContextGuard guard(context);

    int value = 0;
    if (!check_cu(cuFuncGetAttribute_f(&value, attribute, (CUfunction)kernel)))
        return -1;

    return value;
}

int wp_cuda_get_kernel_static_shared_memory(void* context, void* kernel)
{
    return get_cuda_kernel_attribute(context, kernel, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
}

bool wp_cuda_get_kernel_properties(void* context, void* kernel, int* properties, int property_count)
{
    constexpr int kernel_property_count = 2;
    if (!kernel || !properties || property_count != kernel_property_count)
        return false;

    ContextGuard guard(context);
    int queried_properties[kernel_property_count] = {};

    if (!check_cu(cuFuncGetAttribute_f(&queried_properties[0], CU_FUNC_ATTRIBUTE_NUM_REGS, (CUfunction)kernel)))
        return false;
    if (!check_cu(cuFuncGetAttribute_f(&queried_properties[1], CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, (CUfunction)kernel)))
        return false;

#if defined(__HIP_PLATFORM_AMD__)
    {
        // Use the number of registers the GPU reserves, which is never fewer than the code uses (see g_hip_kernel_vgprs above).
        auto held = g_hip_kernel_vgprs.find((CUfunction)kernel);
        if (held != g_hip_kernel_vgprs.end() && held->second.second > queried_properties[0])
            queried_properties[0] = held->second.second;
    }
#endif

    properties[0] = queried_properties[0];
    properties[1] = queried_properties[1];
    return true;
}

bool wp_cuda_set_kernel_cluster_attrs(void* kernel, int cx, int cy, int cz)
{
    if (!kernel)
        return false;

    int total = cx * cy * cz;
    if (total <= 1)
        return true;  // (1,1,1) and other unit shapes are pure no-ops.

#if defined(__HIP_PLATFORM_AMD__)
    // On HIP, clusters are formed inside the kernel itself (WP_CLUSTER_INIT in tile.h), so there is no attribute to set.
    (void)total;
    return true;
#else
    if (total > 8) {
        // CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED == 14
        // (defined in CUDA 12.0 headers; declared as a constant here to
        // decouple from any future header reorg).
        const CUfunction_attribute non_portable_attr = (CUfunction_attribute)14;
        CUresult res = cuFuncSetAttribute_f((CUfunction)kernel, non_portable_attr, 1);
        if (res != CUDA_SUCCESS)
            return false;
    }

    return true;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

int wp_cuda_get_max_cluster_dim(void* context, void* kernel, int block_dim, int dynamic_smem_bytes)
{
    if (!kernel || !context)
        return 1;

#if defined(__HIP_PLATFORM_AMD__)
    // Clusters formed in software can be as large as on NVIDIA's largest parts (16 blocks on Hopper and Blackwell desktop GPUs), but
    // never larger than what the GPU can hold at once for this kernel at this block size, because a cluster's blocks are meant to
    // run together.
    {
        ContextGuard guard(context);
        int per_cu = 0, cus = 0, device = 0;
        if (hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, (hipFunction_t)kernel, block_dim, dynamic_smem_bytes)
                != hipSuccess
            || hipGetDevice(&device) != hipSuccess
            || hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess) {
            ignore_cuda_error(hipGetLastError());
            return 1;
        }
        const int resident = per_cu * cus;
        return resident >= 16 ? 16 : (resident >= 1 ? resident : 1);
    }
#else
    ContextGuard guard(context);

    // Opt into non-portable cluster sizes so probes above 8 are valid on
    // hardware that supports them. The attribute is a no-op on cluster sizes
    // <= 8. CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED == 14.
    //
    // This is a query helper, so save and restore the attribute to avoid
    // permanently opting the caller's kernel into non-portable cluster sizes
    // as a side effect of probing. If the attribute can't be read back,
    // prior_non_portable stays 0 (disabled), which is the value we restore.
    const CUfunction_attribute non_portable_attr = (CUfunction_attribute)14;
    int prior_non_portable = 0;
    cuFuncGetAttribute_f(&prior_non_portable, non_portable_attr, (CUfunction)kernel);
    cuFuncSetAttribute_f((CUfunction)kernel, non_portable_attr, 1);

    // Descending probe via cuOccupancyMaxActiveClusters: return the largest
    // cluster_dim the driver reports as launchable. For plain kernels this
    // reveals the device's true cluster limit (8 on integrated Hopper+ SoCs
    // like Thor sm_101 and DGX Spark sm_121, up to 16 on Hopper/Blackwell
    // desktop parts). For kernels with __cluster_dims__ baked in, the driver
    // only accepts the declared size, so we get the declared value back (or
    // 1 if the device can't support it).
    CUlaunchAttribute cluster_attr = {};
    cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attr.value.clusterDim.y = 1;
    cluster_attr.value.clusterDim.z = 1;

    CUlaunchConfig config = {};
    config.gridDimY = 1;
    config.gridDimZ = 1;
    config.blockDimX = (unsigned int)block_dim;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = (unsigned int)dynamic_smem_bytes;
    config.hStream = 0;
    config.attrs = &cluster_attr;
    config.numAttrs = 1;

    int max_cluster_dim = 1;
    for (int cd = 16; cd >= 2; --cd) {
        cluster_attr.value.clusterDim.x = (unsigned int)cd;
        config.gridDimX = (unsigned int)cd;
        int num_clusters = 0;
        CUresult res = cuOccupancyMaxActiveClusters_f(&num_clusters, (CUfunction)kernel, &config);
        if (res == CUDA_SUCCESS && num_clusters >= 1) {
            max_cluster_dim = cd;
            break;
        }
    }

    // Restore the prior value (disabled when the read-back above failed) so
    // probing leaves no observable side effect on the caller's kernel.
    cuFuncSetAttribute_f((CUfunction)kernel, non_portable_attr, prior_non_portable);

    return max_cluster_dim;
#endif  // defined(__HIP_PLATFORM_AMD__)
}

void* wp_cuda_get_kernel(void* context, void* module, const char* name)
{
    ContextGuard guard(context);

    CUfunction kernel = NULL;
    if (!check_cu(cuModuleGetFunction_f(&kernel, (CUmodule)module, name))) {
        fprintf(stderr, "Warp CUDA error: Failed to lookup kernel function %s in module\n", name);
        return NULL;
    }

    g_kernel_names[kernel] = name;
#if defined(__HIP_PLATFORM_AMD__)
    {
        auto held = g_hip_module_vgprs.find((CUmodule)module);
        if (held != g_hip_module_vgprs.end()) {
            auto entry = held->second.find(name);
            if (entry != held->second.end())
                g_hip_kernel_vgprs[kernel] = { (CUmodule)module, entry->second };
        }
    }
#endif
    return kernel;
}

size_t wp_cuda_launch_kernel(
    void* context,
    void* kernel,
    size_t dim,
    int max_blocks,
    int block_dim,
    int grid_stride,
    int cluster_dim,
    int shared_memory_bytes,
    void** args,
    void* stream,
    const APICLaunchInfo* apic_info
)
{
    ContextGuard guard(context);

    if (block_dim <= 0) {
#if defined(_DEBUG)
        fprintf(stderr, "Warp warning: Launch got block_dim %d. Setting to 256.\n", block_dim);
#endif
        block_dim = 256;
    }

    if (cluster_dim <= 0) {
        cluster_dim = 1;
    }
    if (cluster_dim > 1) {
        ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
#if defined(__HIP_PLATFORM_AMD__)
        // Clusters are formed in software on AMD GPUs (WP_CLUSTER_INIT in tile.h), so the grid below is kept to whole clusters.
        if (!context_info || !context_info->device_info) {
#else
        if (!context_info || !context_info->device_info || context_info->device_info->arch < 90) {
#endif
            cluster_dim = 1;
        }
    }

#if defined(__HIP_PLATFORM_AMD__)
    // Refuse a workgroup that could never be placed rather than dispatching it. HIP checks a launch's dynamic LDS against the most a
    // block may hold, but not together with the kernel's static LDS. A dispatch whose dynamic and static LDS together exceed the
    // device's LDS is never scheduled, and its queue waits forever (test_tile_reduce_axis_tier2_3d hung in hipDeviceSynchronize).
    // NVIDIA's driver refuses such a launch, and so does this code, reporting the numbers.
    // HIP does refuse a dynamic request larger than the LDS itself with "invalid argument", as CUDA's driver does. It only misses
    // the case where the two together are too large.
    if (shared_memory_bytes > 0) {
        ContextInfo* context_info = get_context_info(static_cast<CUcontext>(context));
        const int lds = (context_info && context_info->device_info) ? context_info->device_info->max_smem_bytes : 0;
        int static_bytes = 0;
        if (lds > 0 && shared_memory_bytes <= lds
            && cuFuncGetAttribute_f(&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, (CUfunction)kernel) == CUDA_SUCCESS
            && static_bytes + shared_memory_bytes > lds) {
            wp::set_error_string(
                "Warp CUDA error: invalid argument: the kernel asks for %d bytes of dynamic shared memory beside its %d bytes of static "
                "shared memory, past the %d bytes a workgroup can hold on this card; such a launch could never run",
                shared_memory_bytes, static_bytes, lds
            );
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
#endif

    unsigned int grid_x, grid_y = 1, grid_z = 1;

    if (grid_stride) {
        // Grid-stride loop kernel: a 1D grid suffices because the loop covers every work item.
        // max_blocks (when set) caps the block count; otherwise use the CUDA gridDim.x max (2**31-1).
        // A clustered kernel needs gridDim.x to be a multiple of cluster_dim: an untruncated grid must
        // already be aligned (pad dim), while a grid truncated by max_blocks rounds down to a whole
        // number of clusters (the loop still covers every work item).
        int natural_grid_dim = (dim + block_dim - 1) / block_dim;
        int grid_dim = natural_grid_dim;
        int cap = (max_blocks > 0) ? max_blocks : 2147483647;
        if (grid_dim < 0)
            grid_dim = cap;
        else if (grid_dim > cap)
            grid_dim = cap;
        if (cluster_dim > 1 && grid_dim > 0) {
            if (grid_dim == natural_grid_dim) {
                if (grid_dim % cluster_dim != 0) {
                    wp::set_error_string(
                        "Warp CUDA error: clustered kernel launch requires the block count to be a multiple of "
                        "cluster_dim (got %d blocks, cluster_dim=%d); pad dim to a whole number of clusters",
                        grid_dim, cluster_dim
                    );
                    return CUDA_ERROR_INVALID_VALUE;
                }
            } else {
                grid_dim = (grid_dim / cluster_dim) * cluster_dim;
                if (grid_dim == 0) {
                    wp::set_error_string(
                        "Warp CUDA error: clustered kernel launch requires max_blocks to be 0 or at least cluster_dim "
                        "(got max_blocks=%d, cluster_dim=%d)",
                        max_blocks, cluster_dim
                    );
                    return CUDA_ERROR_INVALID_VALUE;
                }
            }
        }
#if defined(__HIP_PLATFORM_AMD__)
        // On HIP, the total number of work items along x (gridDim.x * blockDim.x) must not exceed 2^32-1. The grid-stride
        // loop covers every work item, so clamping the block count here is safe.
        {
            const int hip_max_grid = (int)(0xFFFFFFFFu / (unsigned)block_dim);
            if (grid_dim > hip_max_grid)
                grid_dim = cluster_dim > 1 ? (hip_max_grid / cluster_dim) * cluster_dim : hip_max_grid;
        }
#endif
        grid_x = (unsigned int)(grid_dim <= 0 ? 1 : grid_dim);
    } else {
        // Lean 3D kernel (no loop): spread blocks across a 3D grid so the launch can exceed the
        // gridDim.x limit. Cap grid.x so gridDim.x*blockDim.x stays within uint32 (matching the
        // lean kernel template's index math on the IMAD.WIDE.U32 fast path), then spill the
        // remaining blocks into grid.y and grid.z (each capped at 65535).
        //
        // Clusters group cluster_dim blocks along x, but a lean block early-returns (before any
        // cluster barrier) when _idx >= dim.size. To keep every cluster all-run or all-return, require
        // the block count to be a whole number of clusters (pad dim) and keep grid.x a multiple of
        // cluster_dim, so the run/early-return boundary lands on a cluster boundary.
        size_t total_blocks = (dim + block_dim - 1) / block_dim;
        if (cluster_dim > 1 && (total_blocks % (size_t)cluster_dim) != 0) {
            wp::set_error_string(
                "Warp CUDA error: clustered kernel launch requires the block count to be a multiple of "
                "cluster_dim (got %zu blocks, cluster_dim=%d); pad dim to a whole number of clusters",
                total_blocks, cluster_dim
            );
            return CUDA_ERROR_INVALID_VALUE;
        }
        unsigned int max_grid_x = (1u << 24) / (unsigned int)block_dim;
        if (cluster_dim > 1) {
            max_grid_x = (max_grid_x / (unsigned int)cluster_dim) * (unsigned int)cluster_dim;
            if (max_grid_x == 0)
                max_grid_x = (unsigned int)cluster_dim;
        }
        unsigned int gx = (unsigned int)(total_blocks < (size_t)max_grid_x ? total_blocks : (size_t)max_grid_x);
        if (gx == 0)
            gx = 1;
        size_t remaining = (total_blocks + gx - 1) / gx;
        unsigned int gy = (unsigned int)(remaining < 65535u ? remaining : 65535u);
        if (gy == 0)
            gy = 1;
        unsigned int gz = (unsigned int)((remaining + gy - 1) / gy);
        if (gz == 0)
            gz = 1;
        if (gz > 65535u)
            gz = 65535u;
        grid_x = gx;
        grid_y = gy;
        grid_z = gz;
    }

    WP_HIP_ISSUE(stream);
    begin_cuda_range(WP_TIMING_KERNEL, stream, context, get_cuda_kernel_name(kernel));

    // Skip the launch for an empty grid (no work): the dummy gridDim.x=1 fallback is not a multiple
    // of cluster_dim, so an empty clustered launch (e.g. a recorded lean launch resized via
    // set_dim(0)) would otherwise be rejected by CUDA.
    CUresult res = CUDA_SUCCESS;
    if (dim > 0)
        res = cuLaunchKernel_f(
            (CUfunction)kernel, grid_x, grid_y, grid_z, block_dim, 1, 1, shared_memory_bytes,
            static_cast<CUstream>(stream), args, 0
        );

    check_cu(res);

    end_cuda_range(WP_TIMING_KERNEL, stream);

    // APIC recording: record kernel launch to byte stream if capturing
    if (apic_info) {
        APICState* state = wp_apic_get_cuda_recording_state();
        if (state) {
            // Read shape and size from launch_bounds_t<N> in args[0].
            int ndim = apic_info->kernel_dim;
            if (ndim < 1)
                ndim = 1;
            if (ndim > APIC_LAUNCH_MAX_DIMS)
                ndim = APIC_LAUNCH_MAX_DIMS;

            uint32_t shape[APIC_LAUNCH_MAX_DIMS] = {};
            uint64_t launch_size = dim;
            if (args && args[0]) {
                const uint32_t* bounds_shape = static_cast<const uint32_t*>(args[0]);
                for (int d = 0; d < ndim; d++)
                    shape[d] = bounds_shape[d];

                const size_t size_offset = apic_detail::launch_bounds_size_offset(ndim);
                const uint8_t* bounds_bytes = static_cast<const uint8_t*>(args[0]);
                launch_size = *reinterpret_cast<const size_t*>(bounds_bytes + size_offset);
            } else {
                shape[0] = static_cast<uint32_t>(dim);
            }

            apic_record_kernel_launch(
                state, apic_info->kernel_key, apic_info->module_hash, apic_info->is_forward, shape, ndim, launch_size,
                max_blocks, block_dim, grid_stride, cluster_dim, shared_memory_bytes, apic_info->params,
                apic_info->num_params, apic_info->adj_params, apic_info->relocs, apic_info->num_relocs,
                apic_info->value_data, apic_info->value_data_size
            );
        }
    }

    return res;
}

bool wp_cuda_get_suggested_block_size(
    void* context, void* kernel, int shared_memory_bytes, int* block_size_out, int* min_grid_size_out
)
{
    ContextGuard guard(context);

    int min_grid_size = 0;
    int block_size = 0;
    CUresult res = cuOccupancyMaxPotentialBlockSize_f(
        &min_grid_size, &block_size, (CUfunction)kernel, NULL, shared_memory_bytes, 0
    );

    if (!check_cu(res))
        return false;

    *block_size_out = block_size;
    *min_grid_size_out = min_grid_size;
    return true;
}

bool wp_cuda_graphics_map(void* context, void* resource)
{
    ContextGuard guard(context);

    return check_cu(cuGraphicsMapResources_f(1, (CUgraphicsResource*)resource, get_current_stream()));
}

void wp_cuda_graphics_unmap(void* context, void* resource)
{
    ContextGuard guard(context);

    check_cu(cuGraphicsUnmapResources_f(1, (CUgraphicsResource*)resource, get_current_stream()));
}

void wp_cuda_graphics_device_ptr_and_size(void* context, void* resource, uint64_t* ptr, size_t* size)
{
    ContextGuard guard(context);

    CUdeviceptr device_ptr;
    size_t bytes;
    check_cu(cuGraphicsResourceGetMappedPointer_f(&device_ptr, &bytes, *(CUgraphicsResource*)resource));

#if defined(__HIP_PLATFORM_AMD__)
    *ptr = reinterpret_cast<uint64_t>(device_ptr);
#else
    *ptr = device_ptr;
#endif
    *size = bytes;
}

void* wp_cuda_graphics_register_gl_buffer(void* context, uint32_t gl_buffer, unsigned int flags)
{
    ContextGuard guard(context);

    CUgraphicsResource* resource = new CUgraphicsResource;
    bool success = check_cu(cuGraphicsGLRegisterBuffer_f(resource, gl_buffer, flags));
    if (!success) {
        delete resource;
        return NULL;
    }

    return resource;
}

void* wp_cuda_graphics_register_gl_image(void* context, uint32_t image, uint32_t target, unsigned int flags)
{
    ContextGuard guard(context);

    CUgraphicsResource* resource = new CUgraphicsResource;
    bool success = check_cu(cuGraphicsGLRegisterImage_f(resource, image, target, flags));
    if (!success) {
        delete resource;
        return NULL;
    }

    return resource;
}

uint64_t wp_cuda_graphics_sub_resource_get_mapped_array(
    void* context, void* resource, unsigned int array_index, unsigned int mip_level
)
{
    ContextGuard guard(context);

    CUarray cuda_array = NULL;
    check_cu(
        cuGraphicsSubResourceGetMappedArray_f(&cuda_array, *(CUgraphicsResource*)resource, array_index, mip_level)
    );
    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_cuda_graphics_unregister_resource(void* context, void* resource)
{
    ContextGuard guard(context);

    CUgraphicsResource* res = (CUgraphicsResource*)resource;
    check_cu(cuGraphicsUnregisterResource_f(*res));
    delete res;
}

void wp_cuda_timing_begin(int flags) { g_cuda_timing_state = new CudaTimingState(flags, g_cuda_timing_state); }

int wp_cuda_timing_get_result_count()
{
    if (g_cuda_timing_state)
        return int(g_cuda_timing_state->ranges.size());
    return 0;
}

void wp_cuda_timing_end(timing_result_t* results, int size)
{
    if (!g_cuda_timing_state)
        return;

    // number of results to write to the user buffer
    int count = std::min(wp_cuda_timing_get_result_count(), size);

    // compute timings and write results
    for (int i = 0; i < count; i++) {
        const CudaTimingRange& range = g_cuda_timing_state->ranges[i];
        timing_result_t& result = results[i];
        result.context = range.context;
        result.name = range.name;
        result.flag = range.flag;
        check_cuda(cudaEventElapsedTime(&result.elapsed, range.start, range.end));
    }

    // release events
    for (CudaTimingRange& range : g_cuda_timing_state->ranges) {
        check_cu(cuEventDestroy_f(range.start));
        check_cu(cuEventDestroy_f(range.end));
    }

    // restore previous state
    CudaTimingState* parent_state = g_cuda_timing_state->parent;
    delete g_cuda_timing_state;
    g_cuda_timing_state = parent_state;
}

// #include "spline.inl"
// #include "volume.inl"

// APIC (API Capture) implementation
#include "apic.cu"
