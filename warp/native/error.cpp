// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified by Steve Pederzani in 2026 for AMD GPUs (HIP). See NOTICE.md.

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace wp {
static char g_error_buffer[4096] = "";
static bool g_error_output_enabled = true;
static FILE* g_error_stream = stderr;

// On Windows warp.dll carries its own C runtime (/MT), whose stderr is the handle it found when the DLL loaded. When Python
// redirects stderr (os.dup2, as the StdErrCapture in Warp's tests does), Windows moves the process's error handle and Python's
// runtime follows it, but this runtime does not, so errors printed by Warp went past the redirect. Errors are written to the
// process's error handle as it is now, so they go wherever the process's stderr goes.
static void emit(const char* text)
{
#if defined(_WIN32)
    if (g_error_stream == stderr) {
        HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
        if (h != NULL && h != INVALID_HANDLE_VALUE) {
            fflush(stderr);
            DWORD n = 0;
            if (WriteFile(h, text, DWORD(strlen(text)), &n, NULL) && WriteFile(h, "\n", 1, &n, NULL))
                return;
        }
    }
#endif
    fputs(text, g_error_stream);
    fputc('\n', g_error_stream);
    fflush(g_error_stream);
}

const char* get_error_string() { return g_error_buffer; }

void set_error_string(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buffer, sizeof(g_error_buffer), fmt, args);
    if (g_error_output_enabled) {
        // note: we deliberately avoid vfprintf() due to problems with runtime glibc mismatch
        emit(g_error_buffer);
    }
    va_end(args);
}

void append_error_string(const char* fmt, ...)
{
    size_t offset = strlen(g_error_buffer);
    if (offset + 2 > sizeof(g_error_buffer))
        return;
    g_error_buffer[offset++] = '\n';
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buffer + offset, sizeof(g_error_buffer) - offset, fmt, args);
    if (g_error_output_enabled) {
        // note: we deliberately avoid vfprintf() due to problems with runtime glibc mismatch
        emit(g_error_buffer + offset);
    }
    va_end(args);
}

void set_error_output_enabled(bool enable) { g_error_output_enabled = enable; }

bool is_error_output_enabled() { return g_error_output_enabled; }

}  // end of namespace wp
