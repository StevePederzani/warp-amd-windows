// SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
// SPDX-License-Identifier: Apache-2.0
//
// Parallel CPU kernel launches.
//
// Upstream Warp runs a CPU kernel's launch dimension serially on the calling thread. Here, the launch dimension of a one-lane
// kernel that uses no tiles is split into chunks, which a persistent pool of worker threads and the calling thread take
// dynamically from one atomic counter. Codegen marks such a kernel "<name>_cpu_parallel" in the module meta and emits
// "<name>_cpu_forward_range" and "<name>_cpu_backward_range". CPU atomics are real atomics in builtin.h when the CPU code is
// compiled by clang (the JIT), so kernels that accumulate with wp.atomic_add stay correct, and the order of float accumulation
// varies between runs, as it does on any GPU.
//
// A launch ends when its last chunk is done, without waiting for every worker to look at it, because a worker the system has
// preempted (a pool thread can lose its CPU for a time slice) must not hold up launches it took no part in. A worker enters a
// launch by raising the job's `active` count and then checking the job's tag. The launcher retires a job (marks its tag invalid
// and waits for `active` to reach 0) before writing the next one, so a late worker either works on the job it saw or leaves it
// untouched.
//
// The pool is sized by wp_cpu_parallel_launch's num_threads argument. 0 means as many logical CPUs as the process's affinity
// mask allows, so a process pinned to N cores gets N threads. Workers are created on first use, spin briefly after each launch
// (spin_us) so a burst of launches doesn't pay for a wake-up each time, and then sleep on a condition variable. Only one
// parallel launch runs at a time, and a launch that finds the pool busy (from another Python thread) runs serially.

#include "warp.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <immintrin.h>
#include <intrin.h>
#define WP_CPU_PAUSE() _mm_pause()
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define WP_CPU_HAVE_MWAITX_INTRINSICS 1
#endif
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#include <sched.h>
#define WP_CPU_PAUSE() _mm_pause()
#else
#include <sched.h>
#define WP_CPU_PAUSE() std::this_thread::yield()
#endif

typedef void (*wp_cpu_range_fn)(void* dim, void* args, void* adj_args, size_t begin, size_t end);

namespace {

int process_cpu_count()
{
#if defined(_WIN32)
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask) {
        int n = 0;
        for (DWORD_PTR m = process_mask; m; m &= (m - 1))
            ++n;
        return n;
    }
#elif defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0)
            return n;
    }
#endif
    unsigned n = std::thread::hardware_concurrency();
    return n ? int(n) : 1;
}

constexpr uint64_t kRetired = ~uint64_t(0);

// AMD's user-level MONITORX/MWAITX instructions (Zen, CPUID 0x80000001 ECX bit 29) let a waiting worker sleep in the core until
// the launch counter's cache line is written or a timer runs out, instead of spinning. That uses almost no power, and the
// worker's SMT sibling, often the Python thread that launches, gets the whole core. Without them, workers spin with PAUSE.
bool cpu_has_mwaitx()
{
#if defined(WP_CPU_HAVE_MWAITX_INTRINSICS)
    int r[4];
    __cpuid(r, int(0x80000000));
    if (unsigned(r[0]) < 0x80000001u)
        return false;
    __cpuid(r, int(0x80000001));
    return ((unsigned(r[2]) >> 29) & 1u) != 0;
#else
    return false;
#endif
}

const bool g_mwaitx = cpu_has_mwaitx();

// Waits briefly, at most about timeout_cycles of the TSC, for *addr to change from `seen`.
inline void wait_for_change(const std::atomic<uint64_t>* addr, uint64_t seen)
{
#if defined(WP_CPU_HAVE_MWAITX_INTRINSICS)
    if (g_mwaitx) {
        _mm_monitorx(const_cast<std::atomic<uint64_t>*>(addr), 0, 0);
        if (addr->load(std::memory_order_acquire) == seen)
            _mm_mwaitx(2 /* timer on */, 0 /* C1 */, 20000 /* TSC cycles, ~5-10 us */);
        return;
    }
#endif
    (void)addr;
    (void)seen;
    for (int i = 0; i < 16; ++i)
        WP_CPU_PAUSE();
}

class CpuPool {
public:
    ~CpuPool() { resize(0); }

    // Number of threads taking part (the workers plus the caller) as of the last resize.
    int threads() const { return int(workers_.size()) + 1; }

    double spin_us = 5000.0;

    // Run fn over [0, size) in chunks of `chunk`, on `nthreads` threads (the caller included). Returns the number of
    // threads in the pool, or 0 when the pool was busy (the caller then runs the range itself).
    int run(wp_cpu_range_fn fn, void* dim, void* args, void* adj_args, size_t size, int nthreads, size_t chunk)
    {
        std::unique_lock<std::mutex> launch_lock(launch_mutex_, std::try_to_lock);
        if (!launch_lock.owns_lock())
            return 0;

        if (nthreads != threads())
            resize(nthreads - 1);

        // Retire the previous job, so no worker is inside it while its fields change.
        tag_.store(kRetired, std::memory_order_seq_cst);
        while (active_.load(std::memory_order_seq_cst) != 0)
            WP_CPU_PAUSE();

        chunk = chunk ? chunk : 1;
        fn_ = fn;
        dim_ = dim;
        args_ = args;
        adj_args_ = adj_args;
        size_ = size;
        chunk_ = chunk;
        next_.store(0, std::memory_order_relaxed);
        remaining_.store((size + chunk - 1) / chunk, std::memory_order_relaxed);

        const uint64_t g = generation_.load(std::memory_order_relaxed) + 1;
        tag_.store(g, std::memory_order_seq_cst);
        // Publish with seq_cst. It pairs with the sleepers' increment-then-check, so no wake-up is lost.
        generation_.store(g, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(sleep_mutex_);
            sleep_cv_.notify_all();
        }

        participate(g);

        // The launch is over when its last chunk is done, whichever thread ran it.
        while (remaining_.load(std::memory_order_acquire) != 0)
            WP_CPU_PAUSE();

        return threads();
    }

private:
    void participate(uint64_t g)
    {
        active_.fetch_add(1, std::memory_order_seq_cst);
        if (tag_.load(std::memory_order_seq_cst) == g) {
            for (;;) {
                const size_t begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
                if (begin >= size_)
                    break;
                const size_t end = (size_ - begin) < chunk_ ? size_ : begin + chunk_;
                fn_(dim_, args_, adj_args_, begin, end);
                remaining_.fetch_sub(1, std::memory_order_release);
            }
        }
        active_.fetch_sub(1, std::memory_order_seq_cst);
    }

    // `seen` is the generation when the worker was created, read by resize() before the launch that created the pool publishes.
    // If it were read inside the thread, a worker that started late would treat the first launch's generation as already seen
    // and hang.
    void worker_main(uint64_t seen)
    {
        for (;;) {
            uint64_t g = generation_.load(std::memory_order_acquire);
            if (g == seen && !stop_.load(std::memory_order_relaxed)) {
                const auto t0 = std::chrono::steady_clock::now();
                // Spinning without MWAITX costs a core's power and slows its SMT sibling, so it is kept short.
                const auto spin = std::chrono::duration<double, std::micro>(g_mwaitx ? spin_us : (spin_us < 50.0 ? spin_us : 50.0));
                unsigned k = 0;
                while ((g = generation_.load(std::memory_order_acquire)) == seen && !stop_.load(std::memory_order_relaxed)) {
                    wait_for_change(&generation_, seen);
                    if ((++k & 3) == 0 && std::chrono::steady_clock::now() - t0 > spin) {
                        std::unique_lock<std::mutex> lk(sleep_mutex_);
                        sleepers_.fetch_add(1, std::memory_order_seq_cst);
                        sleep_cv_.wait(lk, [&] {
                            return generation_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_seq_cst);
                        });
                        sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                        g = generation_.load(std::memory_order_acquire);
                        break;
                    }
                }
            }
            if (stop_.load(std::memory_order_acquire))
                return;
            if (g == seen)
                continue;
            seen = g;
            participate(g);
        }
    }

    void resize(int nworkers)
    {
        if (nworkers < 0)
            nworkers = 0;
        if (!workers_.empty()) {
            stop_.store(true, std::memory_order_seq_cst);
            {
                std::lock_guard<std::mutex> lk(sleep_mutex_);
                sleep_cv_.notify_all();
            }
            for (auto& t : workers_)
                t.join();
            workers_.clear();
            stop_.store(false, std::memory_order_seq_cst);
        }
        workers_.reserve(size_t(nworkers));
        const uint64_t current = generation_.load(std::memory_order_seq_cst);
        for (int i = 0; i < nworkers; ++i)
            workers_.emplace_back([this, current] { worker_main(current); });
    }

    std::mutex launch_mutex_;
    std::vector<std::thread> workers_;

    alignas(64) std::atomic<uint64_t> generation_ { 0 };  // On its own cache line, which is what MWAITX watches.
    alignas(64) std::atomic<int> sleepers_ { 0 };
    std::atomic<bool> stop_ { false };
    std::mutex sleep_mutex_;
    std::condition_variable sleep_cv_;

    // The current job. Its fields only change while it is retired (tag kRetired) and no worker is active in it.
    std::atomic<uint64_t> tag_ { kRetired };
    std::atomic<int> active_ { 0 };
    wp_cpu_range_fn fn_ = nullptr;
    void* dim_ = nullptr;
    void* args_ = nullptr;
    void* adj_args_ = nullptr;
    size_t size_ = 0;
    size_t chunk_ = 1;
    std::atomic<size_t> next_ { 0 };
    std::atomic<size_t> remaining_ { 0 };
};

CpuPool& pool()
{
    // Never destroyed, because worker threads must not be joined from a DLL's static destructors (the loader lock on Windows).
    static CpuPool* p = new CpuPool();
    return *p;
}

}  // namespace

extern "C" {

// Returns the number of logical CPUs this process may run on (its affinity mask), which is the default thread count.
WP_API int wp_cpu_parallel_default_threads() { return process_cpu_count(); }

// Returns the number of threads in the pool's last launch (the workers plus the caller).
WP_API int wp_cpu_parallel_pool_threads() { return pool().threads(); }

WP_API void wp_cpu_parallel_set_spin_us(double us) { pool().spin_us = us < 0.0 ? 0.0 : us; }

// Returns 1 if idle workers wait with MONITORX/MWAITX and 0 if they spin with PAUSE.
WP_API int wp_cpu_parallel_uses_mwaitx() { return g_mwaitx ? 1 : 0; }

// Launches a range entry point over [0, size). num_threads 0 means process_cpu_count(), and 1 means serially on the caller.
// min_chunk is the smallest chunk a thread takes. The chunk is size / (threads * 8), rounded up and never below min_chunk.
// Returns the number of threads used, which is 1 when the launch ran serially on the caller because one thread was requested,
// the launch was too small for two chunks, or the pool was busy.
WP_API int wp_cpu_parallel_launch(
    void* fn, void* dim, void* args, void* adj_args, size_t size, int num_threads, size_t min_chunk
)
{
    wp_cpu_range_fn range = reinterpret_cast<wp_cpu_range_fn>(fn);
    int n = num_threads > 0 ? num_threads : process_cpu_count();
    if (min_chunk == 0)
        min_chunk = 1;
    size_t chunk = (size + size_t(n) * 8 - 1) / (size_t(n) * 8);
    if (chunk < min_chunk)
        chunk = min_chunk;
    if (n <= 1 || size <= chunk) {
        range(dim, args, adj_args, 0, size);
        return 1;
    }
    int used = pool().run(range, dim, args, adj_args, size, n, chunk);
    if (used == 0) {
        range(dim, args, adj_args, 0, size);
        return 1;
    }
    return used;
}

}  // extern "C"
