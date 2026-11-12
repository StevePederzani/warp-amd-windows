// SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel printf on HIP.
//
// HIP's device printf is printed on the host by the C runtime that amdhip64_7.dll links statically. Its stdout is the handle the
// process had when that DLL loaded, so on Windows a kernel's output never follows a later redirection (Warp's tests capture stdout
// with os.dup2, which repoints STD_OUTPUT_HANDLE), and it is written whenever that runtime flushes. CUDA flushes device printf at
// the next synchronization, through the process's stdout. Here, a kernel's printf writes a record with its format and its typed
// arguments into a host-mapped buffer shared by every module, and the host formats the records and writes them to the process's
// current stdout at each synchronization (hip_print_flush in warp.cu), as CUDA does.
//
// This header also holds a kernel's trap, which reports through the buffer (see trap below), and on gfx11 the clock behind
// clock64(), whose scale the host writes into each module as it loads, beside the buffer's address (see the end of this file).
//
// The layout is shared by the device (this header, compiled into every kernel module) and the host (warp.cu).
//   The header is 64 bytes: u64 write (bytes reserved, updated with system-scope atomics), u64 capacity, u32 dropped, and so on.
//   Each record is 8-byte aligned. It holds u32 len (the whole record, stored last with release semantics, where 0 means not yet
//     written), u32 nargs, u32 fmt_bytes (including the NUL), u32 0, the format padded to 8, and then each argument as u32 tag,
//     u32 payload bytes, and the payload padded to 8 (a string's bytes with their NUL, or otherwise one 8-byte value).
//   A reservation that doesn't fit leaves a skip record (nargs = WP_HIP_PRINT_SKIP) if its header fits, and counts as dropped.

#define WP_HIP_PRINT_HEADER 64u
// The header's u32 at this offset is set when a kernel traps (see trap below).
#define WP_HIP_PRINT_TRAPPED 24u
#define WP_HIP_PRINT_SKIP 0xffffffffu
#define WP_HIP_PRINT_I32 1u
#define WP_HIP_PRINT_U32 2u
#define WP_HIP_PRINT_I64 3u
#define WP_HIP_PRINT_U64 4u
#define WP_HIP_PRINT_F64 5u
#define WP_HIP_PRINT_STR 6u
#define WP_HIP_PRINT_PTR 7u
#define WP_HIP_PRINT_MAX_STR 1024u

// This is only used in kernel modules compiled by hipRTC (Warp's modules, including AOT ones), which Warp loads and attaches.
// warp.dll's own device code, built with hipcc, keeps HIP's printf.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIPCC_RTC__) && !defined(WP_HIP_NATIVE_PRINTF)

// The buffer's device address, written into each module when it loads. It is 0 until then, and prints are dropped. It is weak
// because a module linked with the HIP device library (warp.cu) includes this header in each of its units, and two strong
// definitions fail to link. With a weak one, the linker keeps one copy and the loader still finds it by name.
extern "C" __device__ __attribute__((weak)) unsigned long long wp_hip_print_buffer = 0;

namespace wp {
namespace hip_print {

struct Arg {
    unsigned tag;
    unsigned long long v;
    const char* s;
};

__device__ inline Arg arg(int x) { return { WP_HIP_PRINT_I32, (unsigned long long)(long long)x, nullptr }; }
__device__ inline Arg arg(unsigned int x) { return { WP_HIP_PRINT_U32, (unsigned long long)x, nullptr }; }
__device__ inline Arg arg(long x) { return { WP_HIP_PRINT_I64, (unsigned long long)(long long)x, nullptr }; }
__device__ inline Arg arg(unsigned long x) { return { WP_HIP_PRINT_U64, (unsigned long long)x, nullptr }; }
__device__ inline Arg arg(long long x) { return { WP_HIP_PRINT_I64, (unsigned long long)x, nullptr }; }
__device__ inline Arg arg(unsigned long long x) { return { WP_HIP_PRINT_U64, x, nullptr }; }
__device__ inline Arg arg(short x) { return arg(int(x)); }
__device__ inline Arg arg(unsigned short x) { return arg(int(x)); }
__device__ inline Arg arg(char x) { return arg(int(x)); }
__device__ inline Arg arg(signed char x) { return arg(int(x)); }
__device__ inline Arg arg(unsigned char x) { return arg(int(x)); }
__device__ inline Arg arg(bool x) { return arg(int(x)); }
__device__ inline Arg arg(double x) { return { WP_HIP_PRINT_F64, __builtin_bit_cast(unsigned long long, x), nullptr }; }
__device__ inline Arg arg(float x) { return arg(double(x)); }
__device__ inline Arg arg(const char* x) { return { WP_HIP_PRINT_STR, 0ull, x }; }
__device__ inline Arg arg(char* x) { return arg((const char*)x); }
template <typename T> __device__ inline Arg arg(T* x) { return { WP_HIP_PRINT_PTR, (unsigned long long)x, nullptr }; }
template <typename T> __device__ inline Arg arg(const T* x) { return { WP_HIP_PRINT_PTR, (unsigned long long)x, nullptr }; }

__device__ inline unsigned str_bytes(const char* s)
{
    if (!s)
        return 7u;  // "(null)" and its NUL
    unsigned n = 0;
    while (n < WP_HIP_PRINT_MAX_STR - 1u && s[n])
        ++n;
    return n + 1u;
}

__device__ inline unsigned pad8(unsigned n) { return (n + 7u) & ~7u; }

__device__ inline void put_str(char* dst, const char* s, unsigned bytes)
{
    if (!s) {
        const char* null_text = "(null)";
        for (unsigned i = 0; i < 7u; ++i)
            dst[i] = null_text[i];
        return;
    }
    for (unsigned i = 0; i + 1u < bytes; ++i)
        dst[i] = s[i];
    dst[bytes - 1u] = 0;
}

__device__ inline int write_record(const char* fmt, const Arg* args, unsigned nargs)
{
    char* const base = reinterpret_cast<char*>(wp_hip_print_buffer);
    if (!base)
        return -1;
    const unsigned fmt_bytes = str_bytes(fmt);
    unsigned len = 16u + pad8(fmt_bytes);
    for (unsigned i = 0; i < nargs; ++i)
        len += 8u + (args[i].tag == WP_HIP_PRINT_STR ? pad8(str_bytes(args[i].s)) : 8u);
    unsigned long long* const write = reinterpret_cast<unsigned long long*>(base);
    const unsigned long long capacity = reinterpret_cast<const unsigned long long*>(base)[1];
    const unsigned long long off
        = __hip_atomic_fetch_add(write, (unsigned long long)len, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    char* const rec = base + WP_HIP_PRINT_HEADER + off;
    if (off + len > capacity) {
        __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(base + 16), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (off + 16u <= capacity) {
            reinterpret_cast<unsigned*>(rec)[1] = WP_HIP_PRINT_SKIP;
            __hip_atomic_store(
                reinterpret_cast<unsigned*>(rec), (unsigned)(capacity - off), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM
            );
        }
        return -1;
    }
    unsigned* const head = reinterpret_cast<unsigned*>(rec);
    head[1] = nargs;
    head[2] = fmt_bytes;
    head[3] = 0u;
    char* p = rec + 16;
    put_str(p, fmt, fmt_bytes);
    p += pad8(fmt_bytes);
    for (unsigned i = 0; i < nargs; ++i) {
        unsigned* const a = reinterpret_cast<unsigned*>(p);
        a[0] = args[i].tag;
        if (args[i].tag == WP_HIP_PRINT_STR) {
            const unsigned bytes = str_bytes(args[i].s);
            a[1] = bytes;
            put_str(p + 8, args[i].s, bytes);
            p += 8 + pad8(bytes);
        } else {
            a[1] = 8u;
            *reinterpret_cast<unsigned long long*>(p + 8) = args[i].v;
            p += 16;
        }
    }
    __hip_atomic_store(head, len, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    return (int)nargs;
}

template <typename... Args> __device__ inline int printf_(const char* fmt, Args... args)
{
    const Arg list[sizeof...(Args) + 1] = { arg(args)..., Arg { 0u, 0ull, nullptr } };
    return write_record(fmt, list, (unsigned)sizeof...(Args));
}

// A kernel's __trap(). CUDA aborts the kernel, and the next synchronization reports error 719. HIP's own trap (__builtin_trap) under
// Windows ends the kernel and reports nothing: the synchronization after it and every later one succeed (measured on the Radeon
// 8060S, where the driver reports that it could not set up a trap handler), so a kernel's failed check, such as Warp's zero slice
// step, went unnoticed.
// Here the trap writes its error into the trapped word in the buffer's header (WP_HIP_PRINT_TRAPPED), unless an earlier trap wrote
// one, the wave ends, and the host reports that error at the next synchronization and every one after it (hip_print_flush in
// hip_print_host.h), as a CUDA context does: 719 (unspecified launch failure) for __trap(), 710 (device-side assert triggered) for a
// failed assert (crt.h). Before the buffer is attached, HIP's own trap is used.
#define WP_HIP_TRAP_ERROR 719u
#define WP_HIP_ASSERT_ERROR 710u

__device__ __attribute__((noreturn)) inline void trap(unsigned error = WP_HIP_TRAP_ERROR)
{
    char* const base = reinterpret_cast<char*>(wp_hip_print_buffer);
    if (base) {
        unsigned none = 0u;
        __hip_atomic_compare_exchange_strong(
            reinterpret_cast<unsigned*>(base + WP_HIP_PRINT_TRAPPED), &none, error, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_SYSTEM
        );
        __builtin_amdgcn_endpgm();
    }
    __builtin_trap();
}

}  // namespace hip_print
}  // namespace wp

#define printf(...) ::wp::hip_print::printf_(__VA_ARGS__)

#endif  // __HIP_DEVICE_COMPILE__ && __HIPCC_RTC__ && !WP_HIP_NATIVE_PRINTF

// clock64() and clock() on gfx11. CUDA's clock64() is a 64-bit count of the SM's cycles. HIP's is __builtin_readcyclecounter(),
// which on gfx11 reads SHADER_CYCLES, a 20-bit counter: on the Radeon 8060S it wraps every 2^20 cycles (0.36 ms at the 2.9 GHz the
// card ran at, measured against the real-time counter), so a kernel that waits a number of cycles, as Warp's test_sync_errors does,
// never stops waiting. gfx11's only 64-bit counter is the real-time counter wall_clock64() reads, at a constant 100 MHz. Here
// clock64() is that counter in shader cycles: it is multiplied by the device's clock rate over its wall-clock rate, which the host
// writes into each module as it loads (wp_hip_clock_scale, hip_print_attach), so a count of cycles divided by
// hipDeviceAttributeClockRate is the time it took, as on CUDA. It is 64 bits wide and never goes backwards, and it advances in steps
// of the scale (29 cycles on the 8060S). Before the scale is written, it counts real-time ticks.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIPCC_RTC__) && defined(__GFX11__)

extern "C" __device__ __attribute__((weak)) unsigned wp_hip_clock_scale = 0;

namespace wp {
namespace hip_print {

__device__ inline long long cycles()
{
    const unsigned scale = wp_hip_clock_scale;
    return wall_clock64() * (long long)(scale ? scale : 1u);
}

}  // namespace hip_print
}  // namespace wp

#define clock64() ::wp::hip_print::cycles()
#define clock() ::wp::hip_print::cycles()

#endif  // __HIP_DEVICE_COMPILE__ && __HIPCC_RTC__ && __GFX11__
