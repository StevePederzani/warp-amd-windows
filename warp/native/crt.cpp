// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.

#include "crt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>  // for __debugbreak()
#elif defined(__has_builtin)
#if __has_builtin(__builtin_debugtrap)
#define WP_HAS_DEBUGTRAP 1
#endif
#endif

// CPU kernel printf, written to the process's current stdout. CPU kernels resolve printf to warp-clang.dll's own C runtime (/MT),
// whose stdout is the handle the process had when that DLL loaded, so a later redirect (os.dup2, which Warp's tests use) never
// reaches it, and its buffer is written at exit. On Windows a CPU module calls this function instead, through the hook its
// prelude declares (_wp_printf_hook in crt.h, set when the module loads). It formats with this DLL's runtime and writes to
// STD_OUTPUT_HANDLE as it is now, the same way error.cpp does for STD_ERROR_HANDLE.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstdarg>
#include <string>

extern "C" WP_API int _wp_printf(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    va_list again;
    va_copy(again, args);
    char small[1024];
    const int n = vsnprintf(small, sizeof(small), format, args);
    va_end(args);
    std::string large;
    const char* text = small;
    if (n >= int(sizeof(small))) {
        large.resize(size_t(n) + 1);
        vsnprintf(&large[0], large.size(), format, again);
        text = large.c_str();
    }
    va_end(again);
    if (n <= 0)
        return n;
#if defined(_WIN32)
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != NULL && out != INVALID_HANDLE_VALUE) {
        // Write each newline as CR LF, as a text-mode C runtime stdout does on Windows, because Warp's print tests expect os.linesep.
        std::string crlf;
        crlf.reserve(size_t(n) + 16);
        for (int i = 0; i < n; ++i) {
            if (text[i] == '\n')
                crlf += '\r';
            crlf += text[i];
        }
        DWORD written = 0;
        if (WriteFile(out, crlf.data(), DWORD(crlf.size()), &written, NULL))
            return n;
    }
#endif
    fwrite(text, 1, size_t(n), stdout);
    fflush(stdout);
    return n;
}

extern "C" WP_API int _wp_isfinite(double x) { return std::isfinite(x); }

extern "C" WP_API int _wp_isnan(double x) { return std::isnan(x); }

extern "C" WP_API int _wp_isinf(double x) { return std::isinf(x); }

extern "C" WP_API void _wp_assert(const char* expression, const char* file, unsigned int line)
{
    fflush(stdout);
    fprintf(
        stderr,
        "Assertion failed: '%s'\n"
        "At '%s:%u'\n",
        expression, file, line
    );
    fflush(stderr);

    // Break into debugger if attached, then abort.
    // We don't use assert() because crt.cpp may be compiled with NDEBUG defined,
    // which would make assert() a no-op.
#if defined(_MSC_VER)
    __debugbreak();
#elif defined(WP_HAS_DEBUGTRAP)
    __builtin_debugtrap();  // Clang - breakpoint, can continue
#elif defined(__GNUC__)
    __builtin_trap();  // GCC - causes SIGILL, does not return
#endif

    abort();
}
