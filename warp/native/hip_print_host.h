// SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The host half of kernel printf on HIP. The device half and the record layout are in hip_print.h. Only warp.cu includes this.
//   hip_print_attach(module)  gives a newly loaded module the buffer's device address in its wp_hip_print_buffer, creating the
//                             buffer on first use.
//   hip_print_flush()         formats every committed record and writes it through the process's current stdout handle, in the
//                             order the kernels reserved them, and returns the error a trapped kernel left (0 if none has). It is
//                             called after each synchronization, where CUDA flushes its printf and reports a trapped kernel.

#if defined(__HIP_PLATFORM_AMD__)

#include "hip_print.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX  // windows.h's min and max macros would break std::min and std::max in warp.cu.
#endif
#include <windows.h>
#endif

namespace {

constexpr unsigned long long k_hip_print_capacity = 8ull << 20;  // CUDA's default printf FIFO is 1 MiB, and this one is 8 MiB.

std::mutex g_hip_print_mutex;
char* g_hip_print_host = nullptr;          // Points to the buffer, which is host-mapped and coherent.
unsigned long long g_hip_print_device = 0;  // Holds the buffer's device address.
unsigned long long g_hip_print_read = 0;    // Counts the bytes the host has consumed.
bool g_hip_print_failed = false;

// These are guarded by g_hip_print_mutex.
bool hip_print_buffer_ready()
{
    if (g_hip_print_host)
        return true;
    if (g_hip_print_failed)
        return false;
    g_hip_print_failed = true;
    void* p = nullptr;
    const size_t bytes = size_t(WP_HIP_PRINT_HEADER + k_hip_print_capacity);
    if (hipHostMalloc(&p, bytes, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess || !p) {
        (void)hipGetLastError();
        fprintf(stderr, "Warp HIP warning: kernel prints are off (their buffer could not be made)\n");
        return false;
    }
    memset(p, 0, bytes);
    reinterpret_cast<unsigned long long*>(p)[1] = k_hip_print_capacity;
    void* d = nullptr;
    if (hipHostGetDevicePointer(&d, p, 0) != hipSuccess || !d) {
        (void)hipGetLastError();
        (void)hipHostFree(p);
        fprintf(stderr, "Warp HIP warning: kernel prints are off (their buffer has no device address)\n");
        return false;
    }
    g_hip_print_host = static_cast<char*>(p);
    g_hip_print_device = reinterpret_cast<unsigned long long>(d);
    g_hip_print_failed = false;
    return true;
}

void hip_print_write(const std::string& text)
{
    if (text.empty())
        return;
#if defined(_WIN32)
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out && out != INVALID_HANDLE_VALUE) {
        // Write each newline as CR LF, as a text-mode C runtime stdout does on Windows, which is what a CUDA kernel's printf gives there.
        std::string crlf;
        crlf.reserve(text.size() + text.size() / 8 + 1);
        for (const char c : text) {
            if (c == '\n')
                crlf += '\r';
            crlf += c;
        }
        DWORD written = 0;
        WriteFile(out, crlf.data(), DWORD(crlf.size()), &written, nullptr);
        return;
    }
#endif
    fwrite(text.data(), 1, text.size(), stdout);
    fflush(stdout);
}

// Formats one conversion of a format string, with its flags, width, and precision as written, for one recorded argument.
void hip_print_one(std::string& out, const std::string& spec_prefix, char conv, unsigned tag, unsigned long long v, const char* s)
{
    char buf[512];
    std::string spec = spec_prefix;  // "%" + flags + width + precision, without a length modifier
    int n = -1;
    switch (conv) {
    case 'd':
    case 'i':
        if (tag == WP_HIP_PRINT_I32 || tag == WP_HIP_PRINT_U32)
            n = snprintf(buf, sizeof(buf), (spec + 'd').c_str(), int(unsigned(v)));
        else
            n = snprintf(buf, sizeof(buf), (spec + "lld").c_str(), (long long)v);
        break;
    case 'u':
    case 'o':
    case 'x':
    case 'X':
        if (tag == WP_HIP_PRINT_I32 || tag == WP_HIP_PRINT_U32)
            n = snprintf(buf, sizeof(buf), (spec + conv).c_str(), unsigned(v));
        else
            n = snprintf(buf, sizeof(buf), (spec + "ll" + conv).c_str(), (unsigned long long)v);
        break;
    case 'c':
        n = snprintf(buf, sizeof(buf), (spec + 'c').c_str(), int(unsigned(v)));
        break;
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A': {
        double d = 0.0;
        if (tag == WP_HIP_PRINT_F64)
            memcpy(&d, &v, sizeof(d));
        else if (tag == WP_HIP_PRINT_I32)
            d = double(int(unsigned(v)));
        else
            d = double((long long)v);
        n = snprintf(buf, sizeof(buf), (spec + conv).c_str(), d);
        break;
    }
    case 's':
        if (tag == WP_HIP_PRINT_STR) {
            const int need = snprintf(nullptr, 0, (spec + 's').c_str(), s ? s : "(null)");
            if (need > 0) {
                std::string tmp(size_t(need) + 1, '\0');
                snprintf(&tmp[0], tmp.size(), (spec + 's').c_str(), s ? s : "(null)");
                tmp.resize(size_t(need));
                out += tmp;
            }
            return;
        }
        n = snprintf(buf, sizeof(buf), "%llx", (unsigned long long)v);
        break;
    case 'p':
        n = snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)v);
        break;
    default:
        return;
    }
    if (n > 0)
        out.append(buf, size_t(n) < sizeof(buf) ? size_t(n) : sizeof(buf) - 1);
}

// Formats one record as text, the way the C runtime's printf would have written it.
void hip_print_format(const char* rec, std::string& out)
{
    const unsigned* head = reinterpret_cast<const unsigned*>(rec);
    const unsigned nargs = head[1];
    const unsigned fmt_bytes = head[2];
    const char* fmt = rec + 16;
    const char* p = rec + 16 + ((fmt_bytes + 7u) & ~7u);
    unsigned used = 0;
    auto next = [&](unsigned& tag, unsigned long long& v, const char*& s) {
        if (used >= nargs) {
            tag = 0;
            v = 0;
            s = nullptr;
            return false;
        }
        const unsigned* a = reinterpret_cast<const unsigned*>(p);
        tag = a[0];
        if (tag == WP_HIP_PRINT_STR) {
            s = p + 8;
            v = 0;
            p += 8 + ((a[1] + 7u) & ~7u);
        } else {
            memcpy(&v, p + 8, sizeof(v));
            s = nullptr;
            p += 16;
        }
        ++used;
        return true;
    };
    const char* end = fmt + (fmt_bytes ? fmt_bytes - 1 : 0);
    for (const char* f = fmt; f < end;) {
        if (*f != '%') {
            const char* lit = f;
            while (f < end && *f != '%')
                ++f;
            out.append(lit, size_t(f - lit));
            continue;
        }
        if (f + 1 < end && f[1] == '%') {
            out += '%';
            f += 2;
            continue;
        }
        std::string spec = "%";
        ++f;
        while (f < end && strchr("-+ #0", *f))
            spec += *f++;
        unsigned tag;
        unsigned long long v;
        const char* s;
        if (f < end && *f == '*') {
            next(tag, v, s);
            spec += std::to_string(int(unsigned(v)));
            ++f;
        } else {
            while (f < end && *f >= '0' && *f <= '9')
                spec += *f++;
        }
        if (f < end && *f == '.') {
            spec += *f++;
            if (f < end && *f == '*') {
                next(tag, v, s);
                spec += std::to_string(int(unsigned(v)));
                ++f;
            } else {
                while (f < end && *f >= '0' && *f <= '9')
                    spec += *f++;
            }
        }
        while (f < end && strchr("hljztLq", *f))
            ++f;  // The recorded type decides the width, not the length modifier, since a device long is 64 bits and a Windows long is 32.
        if (f >= end)
            break;
        const char conv = *f++;
        if (conv == 'n')
            continue;
        next(tag, v, s);
        hip_print_one(out, spec, conv, tag, v, s);
    }
}

}  // namespace

// Creates the buffer when Warp starts (cuda_init), never inside a capture, where host allocation is refused.
static void hip_print_init()
{
    std::lock_guard<std::mutex> lock(g_hip_print_mutex);
    (void)hip_print_buffer_ready();
}

constexpr int k_hip_attach_devices = 64;

// The scale behind clock64() on gfx11 (hip_print.h): the device's clock rate over its wall-clock rate, rounded, at least 1. It is
// read once per device. Guarded by g_hip_print_mutex.
unsigned g_hip_clock_scales[k_hip_attach_devices] = {};

unsigned hip_clock_scale(int device)
{
    if (device < 0 || device >= k_hip_attach_devices)
        return 1u;
    if (!g_hip_clock_scales[device]) {
        int clock_khz = 0;
        int wall_khz = 0;
        unsigned scale = 1u;
        if (hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, device) == hipSuccess
            && hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, device) == hipSuccess && clock_khz > 0
            && wall_khz > 0) {
            const long long rounded = ((long long)clock_khz + wall_khz / 2) / wall_khz;
            scale = rounded > 1 ? (unsigned)rounded : 1u;
        } else {
            (void)hipGetLastError();
        }
        g_hip_clock_scales[device] = scale;
    }
    return g_hip_clock_scales[device];
}

// Gives a newly loaded module what it reads from the host: its wp_hip_print_buffer points at the buffer, and on gfx11 its
// wp_hip_clock_scale holds its device's clock scale. A module without a symbol goes without it: it prints nothing of its own, or its
// clock64() counts real-time ticks. A module may load inside a capture, since Warp loads a module at its first launch, so the values
// are written asynchronously on a stream of the module's device owned by this thread and waited for, with the thread's capture mode
// relaxed. A synchronous copy would run on the null stream, which a capturing stream may not touch.
static void hip_print_attach(hipModule_t module)
{
    if (!module)
        return;
    hipDeviceptr_t print_global = 0;
    hipDeviceptr_t clock_global = 0;
    size_t size = 0;
    if (hipModuleGetGlobal(&print_global, &size, module, "wp_hip_print_buffer") != hipSuccess || size != sizeof(unsigned long long)) {
        (void)hipGetLastError();
        print_global = 0;
    }
    if (hipModuleGetGlobal(&clock_global, &size, module, "wp_hip_clock_scale") != hipSuccess || size != sizeof(unsigned)) {
        (void)hipGetLastError();
        clock_global = 0;
    }
    if (!print_global && !clock_global)
        return;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= k_hip_attach_devices) {
        (void)hipGetLastError();
        return;
    }
    unsigned long long buffer = 0;
    unsigned scale = 1u;
    {
        std::lock_guard<std::mutex> lock(g_hip_print_mutex);
        if (print_global) {
            if (hip_print_buffer_ready())
                buffer = g_hip_print_device;
            else
                print_global = 0;
        }
        if (clock_global)
            scale = hip_clock_scale(device);
    }
    if (!print_global && !clock_global)
        return;
    static thread_local hipStream_t streams[k_hip_attach_devices] = {};
    hipStream_t& s = streams[device];
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    if (!s && hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) {
        (void)hipGetLastError();
        s = nullptr;
    }
    // The copies read buffer and scale from this frame, so whatever was queued is waited for before returning.
    bool queued = false;
    bool ok = s != nullptr;
    if (ok && print_global) {
        ok = hipMemcpyHtoDAsync(print_global, &buffer, sizeof(buffer), s) == hipSuccess;
        queued = queued || ok;
    }
    if (ok && clock_global) {
        ok = hipMemcpyHtoDAsync(clock_global, &scale, sizeof(scale), s) == hipSuccess;
        queued = queued || ok;
    }
    if (!ok)
        (void)hipGetLastError();
    if (queued && hipStreamSynchronize(s) != hipSuccess)
        (void)hipGetLastError();
    (void)hipThreadExchangeStreamCaptureMode(&mode);
}

// The error a trapped kernel left in this process (hip_print.h trap): 719 (hipErrorLaunchFailure) for __trap(), 710 (hipErrorAssert)
// for a failed device assert, 0 while none has. Once a kernel has trapped, every synchronization reports its error, as a CUDA
// context does after a kernel traps in it.
int g_hip_trap_error = 0;

// Writes out every committed record in order, and starts the buffer over once everything in it has been written. Returns the error
// a trapped kernel left, or 0.
static int hip_print_flush()
{
    std::lock_guard<std::mutex> lock(g_hip_print_mutex);
    char* const base = g_hip_print_host;
    if (!base)
        return g_hip_trap_error;
    volatile unsigned long long* const write = reinterpret_cast<volatile unsigned long long*>(base);
    const unsigned long long capacity = reinterpret_cast<unsigned long long*>(base)[1];
    const unsigned long long reserved = *write;
    const unsigned long long end = reserved < capacity ? reserved : capacity;
    std::string out;
    while (g_hip_print_read + 16u <= end) {
        char* const rec = base + WP_HIP_PRINT_HEADER + g_hip_print_read;
        const unsigned len = __atomic_load_n(reinterpret_cast<unsigned*>(rec), __ATOMIC_ACQUIRE);
        if (len == 0u || g_hip_print_read + len > capacity)
            break;  // Reserved but not yet written, because a kernel on another stream is still running.
        if (reinterpret_cast<const unsigned*>(rec)[1] != WP_HIP_PRINT_SKIP)
            hip_print_format(rec, out);
        g_hip_print_read += len;
    }
    unsigned* const dropped = reinterpret_cast<unsigned*>(base + 16);
    const unsigned lost = __atomic_exchange_n(dropped, 0u, __ATOMIC_ACQ_REL);
    hip_print_write(out);
    if (lost)
        fprintf(stderr, "Warp HIP warning: %u kernel print(s) were dropped (the print buffer holds %llu bytes)\n", lost,
                (unsigned long long)capacity);
    if (g_hip_print_read >= end) {
        // Everything was written, so start over, unless a kernel reserved more in the meantime, in which case the next flush
        // continues from here.
        unsigned long long expected = reserved;
        const unsigned long long cleared = end;
        memset(base + WP_HIP_PRINT_HEADER, 0, size_t(cleared));
        if (__atomic_compare_exchange_n(reinterpret_cast<unsigned long long*>(base), &expected, 0ull, false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE))
            g_hip_print_read = 0;
    }
    if (!g_hip_trap_error) {
        const unsigned code = __atomic_load_n(reinterpret_cast<unsigned*>(base + WP_HIP_PRINT_TRAPPED), __ATOMIC_ACQUIRE);
        if (code)
            g_hip_trap_error = static_cast<int>(code);
    }
    return g_hip_trap_error;
}

#endif  // __HIP_PLATFORM_AMD__
