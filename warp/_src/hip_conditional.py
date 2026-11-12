# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

"""Emulated conditional graph capture on HIP.

HIP has no conditional graph nodes (no ``hipGraphConditionalHandle``), so upstream Warp refuses ``wp.capture_while`` and
``wp.capture_if`` inside a HIP graph capture, and a library that captures a solver loop, such as MuJoCo Warp's ``solve``,
falls back to running the loop's full iteration count in the graph.

This module cuts a capture into segments when it reaches a conditional. The graph captured so far is closed as one segment,
the loop body or each branch is captured as its own program of segments (recursively, since conditionals can nest), and the
capture continues in a new segment. The :class:`~warp.Graph` keeps the program, and :func:`~warp.capture_launch` runs it.
Each segment is one native graph launch, and before each pass of a loop the host reads the condition with a copy to pinned
memory and a stream synchronization, exactly as ``capture_while`` does outside a capture. The loop runs as many times as its
condition says and no more, at the cost of one host round trip per pass.

``warp.config.hip_conditional_check_every`` (``k``, default 1) makes the host read the condition once every ``k`` passes of
a loop body instead. That is only correct for a body that changes nothing once its condition is 0. MuJoCo Warp's solver
iteration is such a body, since its kernels skip converged worlds, which is also why its own fallback can run the full
iteration count. With the default, the semantics are exactly those of a conditional node.

Under APIC recording (``capture_begin(apic=True)``), a conditional is recorded the way CUDA records it (see below). Using a
previously captured Graph as a branch is not supported there yet and raises an error, but a callable works.
"""

from __future__ import annotations

import atexit
import ctypes
import weakref
from collections.abc import Callable

import warp
import warp.config

# Graphs that hold segments, freed at exit while the runtime is still up. Graph.__del__ at interpreter shutdown runs after the
# import machinery is gone, and leaving a graph executable to the HIP runtime's own teardown gives it one more thing to clean up.
_live = weakref.WeakSet()


class _Segment:
    """One native graph of a program, instantiated on first launch."""

    __slots__ = ("exec", "graph", "nodes")

    def __init__(self, graph):
        self.graph = graph
        self.exec = None
        self.nodes = -1


class _While:
    __slots__ = ("body", "cond", "host", "last_passes", "launches", "total_passes")

    def __init__(self, cond, body, host):
        self.cond = cond
        self.body = body
        self.host = host
        self.last_passes = 0
        self.total_passes = 0
        self.launches = 0


class _If:
    __slots__ = ("cond", "host", "on_false", "on_true")

    def __init__(self, cond, on_true, on_false, host):
        self.cond = cond
        self.on_true = on_true
        self.on_false = on_false
        self.host = host


# Pinned condition words. Freeing pinned memory (hipHostFree) while any capture is open invalidates that capture (error 901 at
# its next launch), and a graph is often destroyed while another capture is open, for example in a loop that captures again.
# So a destroyed graph's words are kept only while some capture is open, reused first by a new conditional, and freed as soon
# as no capture is open, either when the graph is destroyed or when a capture ends. Nothing outlives its graph past that point
# (Warp's own test_radix_sort_pairs_capture_if_forked_stream counts the live allocations once the graph is gone). No launch
# leaves a copy into a word in flight, because the host reads each word after a stream synchronization before it continues.
_word_pool = []


def _pinned():
    if _word_pool:
        return _word_pool.pop()
    return warp.empty(1, dtype=warp.int32, device="cpu", pinned=True)


def _capturing_anywhere() -> bool:
    try:
        return any(d.captures for d in _ctx().runtime.cuda_devices)
    except (AttributeError, TypeError):
        return True  # If it is unknown, keep the words, which is the safe choice.


def _free_kept_words():
    """Release the kept words when no capture is open. Each word's array frees its pinned memory with its last reference."""
    if _word_pool and not _capturing_anywhere():
        _word_pool.clear()


def _return_words(prog, top=True):
    for step in prog or ():
        if isinstance(step, (_While, _If)):
            if step.host is not None:
                _word_pool.append(step.host)
                step.host = None
            if isinstance(step, _While):
                _return_words(step.body, False)
            else:
                _return_words(step.on_true, False)
                _return_words(step.on_false, False)
    if top:
        _free_kept_words()


def _copy_condition(host, cond, stream):
    """Capture a copy of the condition into its pinned host word, as a node of the segment being captured.

    The host reads the word after the segment has run, following a stream synchronization. Reading the condition with a
    memcpy outside the graphs fails on HIP when the condition was allocated inside the capture, which is how MuJoCo Warp's
    solve() allocates its `nsolving`. hipMemcpy rejects a graph allocation's pointer with "invalid argument", while graph
    nodes such as kernels and this copy can reach it.
    """
    warp.copy(host, cond, stream=stream)


class _GraphRef:
    """A previously captured Graph used as a loop body or branch."""

    __slots__ = ("graph",)

    def __init__(self, graph):
        self.graph = graph


def enabled() -> bool:
    from warp._src.context import runtime  # noqa: PLC0415

    return bool(runtime is not None and runtime.is_hip and warp.config.hip_conditional_emulation)


_context_module = None


def _ctx():
    global _context_module  # noqa: PLW0603
    if _context_module is None:
        from warp._src import context  # noqa: PLC0415

        _context_module = context
    return _context_module


# ---------------------------------------------------------------- Arrays Made During a HIP Capture
# An array allocated while a HIP graph is being captured would be a graph allocation, that is, a memory node of the graph being
# captured. On HIP such memory is not reliably usable from a different graph or from API calls (hipMemcpy rejects its pointer,
# and reused memory showed through in another graph), and an emulated conditional cuts the capture into several graphs. So
# during any HIP capture, an array is allocated once outside the capture, by pausing the capture and resuming it into the same
# graph around the allocation, and it is kept until both the array and every graph captured with it are gone. That is how
# CUDA's graph allocations behave for a single graph, with the same address at every launch.
_owned = {}  # ptr -> [context, graph count alive, array alive]


def capturing_graph(device):
    """Return the Graph being captured on ``device``'s current stream, when its arrays must be allocated outside the capture."""
    from warp._src.context import runtime  # noqa: PLC0415

    if not (runtime.is_hip and warp.config.hip_conditional_emulation and warp.config.hip_capture_persistent_arrays):
        return None
    graph = device.captures.get(device.stream)
    if graph is None or getattr(graph, "_apic_capture", None) is not None:
        return None
    return graph


BODY_ALLOC = "Conditional body graph contains an unsupported operation (memory allocation)"
CHILD_ALLOC = "Child graph contains an unsupported operation (memory allocation)"


def note_capture_alloc(device):
    """Handle an array being allocated on ``device`` while it is captured. Inside an emulated conditional's body the allocation
    is refused the way CUDA refuses it, because a conditional node's body cannot allocate and code that ran here would not run
    on an NVIDIA GPU (Warp's own test_error_alloc_* tests check this). Elsewhere the graph is marked as one that allocates, so it
    is refused later if it is used as a body (CHILD_ALLOC)."""
    if not enabled():
        return
    graph = device.captures.get(device.stream)
    if graph is None:
        return
    if len(getattr(graph, "_hip_stack", None) or []) > 1:
        raise RuntimeError(BODY_ALLOC)
    graph._hip_allocates = True


def persistent_alloc(graph, device, size_in_bytes):
    ctx = _ctx()
    stream = device.stream
    ctx.capture_pause(stream=stream, _suspend_apic_recording=False)
    try:
        ptr = ctx.runtime.core.wp_alloc_device_async(device.context, size_in_bytes, stream.cuda_stream, None)
    finally:
        ctx.capture_resume(graph, stream=stream, _resume_apic_recording=False)
    if not ptr:
        raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on device '{device}' (during a HIP capture)")
    if getattr(graph, "_hip_arrays", None) is None:
        graph._hip_arrays = []
        _live.add(graph)
    graph._hip_arrays.append(ptr)
    _owned[ptr] = [device.context, 1, True]
    return ptr


def release_array(ptr) -> bool:
    """Release an array's memory. Returns True when a graph still holds it, in which case it is freed with the last such graph."""
    entry = _owned.get(ptr)
    if entry is None:
        return False
    entry[2] = False
    if entry[1] > 0:
        return True
    del _owned[ptr]
    return False  # No graph holds it anymore, so the caller frees it now.


def _release_graph_arrays(graph):
    ctx = _context_module
    for ptr in getattr(graph, "_hip_arrays", None) or ():
        entry = _owned.get(ptr)
        if entry is None:
            continue
        entry[1] -= 1
        if entry[1] <= 0 and not entry[2]:
            del _owned[ptr]
            if ctx is not None and ctx.runtime is not None and ctx.runtime.core is not None:
                ctx.runtime.core.wp_free_device_async(entry[0], ptr, None)
    graph._hip_arrays = None


def _program_of(graph):
    """Return the program being appended to, creating the graph's program on first use.

    While a body is being captured, this is the body's program."""
    if getattr(graph, "_hip_program", None) is None:
        graph._hip_program = []
        graph._hip_stack = [graph._hip_program]
        graph._hip_segments = []
    return graph._hip_stack[-1]


def _close_segment(graph, stream) -> _Segment:
    ctx = _ctx()
    ctx.capture_pause(stream=stream, _suspend_apic_recording=False)
    seg = _Segment(graph.graph)
    graph._hip_segments.append(seg)
    graph.graph = None
    return seg


def _open_segment(graph, stream):
    ctx = _ctx()
    device = stream.device
    g = ctypes.c_void_p()
    if not ctx.runtime.core.wp_cuda_graph_create_empty(device.context, ctypes.byref(g)):
        raise RuntimeError(ctx.runtime.get_error_string())
    graph.graph = g
    ctx.capture_resume(graph, stream=stream, _resume_apic_recording=False)


def _capture_program(graph, stream, body, kwargs, tail=None, apic=None) -> list:
    """Capture ``body`` (a callable or a Graph) as its own program. The parent's segment must already be closed.

    ``tail`` is a (host word, condition) pair to copy at the end of the body, for a loop's next check.
    """
    ctx = _ctx()
    if body is None:
        return []
    if not isinstance(body, ctx.Graph) and not isinstance(body, Callable):
        raise TypeError("a conditional body must be a Callable or a Graph")
    prog = []
    graph._hip_stack.append(prog)
    _open_segment(graph, stream)
    try:
        if isinstance(body, ctx.Graph):
            # A previously captured Graph used as the body is launched from its own segment program, followed by the tail copy. A
            # graph whose capture allocated an array is refused, the way CUDA refuses a child graph with a memory node.
            if getattr(body, "_hip_allocates", False):
                raise RuntimeError(CHILD_ALLOC)
            prog.append(_close_segment(graph, stream))
            prog.append(_GraphRef(body))
            _open_segment(graph, stream)
        else:
            body(**kwargs)
        if tail is not None:
            with _Unrecorded(apic):
                _copy_condition(tail[0], tail[1], stream)
    finally:
        prog.append(_close_segment(graph, stream))
        graph._hip_stack.pop()
    return prog


def _check(graph, stream):
    if stream.device.captures.get(stream) is not graph:
        raise RuntimeError("HIP conditional emulation: the capture is not on this stream")


# ---------------------------------------------------------------- APIC Recording of an Emulated Conditional
# Under an APIC capture (capture_begin(apic=True)), a conditional is recorded the way CUDA's capture_if and capture_while record
# it. Its condition's region is tracked, each branch's launches are recorded into an APIC branch sub-stream (wp_apic_begin_branch
# and wp_apic_end_branch), and the op is written last (wp_apic_record_conditional), while the conditional actually runs as the
# segments described here. The emulation's own copies of the condition into its host word are kept out of the recording,
# because they are how the HIP build runs a conditional and not part of the program a saved .wrp file carries.
def _apic_of(graph, device, bodies):
    ctx = _ctx()
    apic = ctx._get_apic_capture_for_device(device) if getattr(graph, "_apic_capture", None) is not None else None
    if apic is not None and any(b is not None and not isinstance(b, Callable) for b in bodies):
        raise NotImplementedError("APIC conditional capture with Graph branches is not yet implemented; pass a Callable instead")
    return apic


def _apic_cond(apic, condition):
    if apic is None:
        return -1, 0
    region, offset = apic.track_array(condition)
    if region < 0:
        raise RuntimeError("an APIC conditional capture's condition array could not be tracked (null pointer?)")
    return region, offset


class _Unrecorded:
    """Pause APIC recording around an emulation step, such as copying the condition into its host word."""

    __slots__ = ("apic", "ctx")

    def __init__(self, apic):
        self.apic = apic
        self.ctx = _ctx()

    def __enter__(self):
        if self.apic is not None:
            self.apic.end_recording()
            self.ctx.runtime._apic_capture = None
        return self

    def __exit__(self, *exc):
        if self.apic is not None:
            self.ctx.runtime._apic_capture = self.apic
            self.apic.begin_recording()
        return False


def _branch(graph, stream, apic, body, kwargs, tail=None):
    """Capture a body as its own program and, under APIC, record it as a branch. Returns (program, branch handle or None)."""
    core = _ctx().runtime.core
    if apic is None or body is None:
        return _capture_program(graph, stream, body, kwargs, tail=tail, apic=apic), None
    handle = ctypes.c_void_p()
    start = core.wp_apic_begin_branch(apic.apic_state)
    try:
        prog = _capture_program(graph, stream, body, kwargs, tail=tail, apic=apic)
    except Exception:
        handle.value = core.wp_apic_end_branch(apic.apic_state, start)
        if handle.value:
            core.wp_apic_free_branch_body(handle)
        raise
    handle.value = core.wp_apic_end_branch(apic.apic_state, start)
    return prog, handle


def _first_check(graph, stream, prog, condition, apic=None):
    """Close the segment before a conditional and copy its condition to a new pinned word in a short segment of its own.

    The word is allocated while no capture is open, because a pinned allocation inside a capture is not safe."""
    prog.append(_close_segment(graph, stream))
    host = _pinned()
    _open_segment(graph, stream)
    try:
        with _Unrecorded(apic):
            _copy_condition(host, condition, stream)
    finally:
        prog.append(_close_segment(graph, stream))
    return host


def capture_while(condition, while_body, stream, graph, **kwargs):
    _check(graph, stream)
    apic = _apic_of(graph, stream.device, (while_body,))
    region, offset = _apic_cond(apic, condition)
    prog = _program_of(graph)
    host = _first_check(graph, stream, prog, condition, apic)
    try:
        body, handle = _branch(graph, stream, apic, while_body, kwargs, tail=(host, condition))
    finally:
        _open_segment(graph, stream)
    prog.append(_While(condition, body, host))
    if apic is not None:
        from warp._src.apic.types import APIC_OP_WHILE  # noqa: PLC0415

        _ctx().runtime.core.wp_apic_record_conditional(apic.apic_state, APIC_OP_WHILE, region, offset, handle, None)


def capture_if(condition, on_true, on_false, stream, graph, **kwargs):
    _check(graph, stream)
    apic = _apic_of(graph, stream.device, (on_true, on_false))
    region, offset = _apic_cond(apic, condition)
    prog = _program_of(graph)
    host = _first_check(graph, stream, prog, condition, apic)
    t_handle = None
    try:
        t, t_handle = _branch(graph, stream, apic, on_true, kwargs)
        f, f_handle = _branch(graph, stream, apic, on_false, kwargs)
    except Exception:
        if t_handle is not None and t_handle.value:
            _ctx().runtime.core.wp_apic_free_branch_body(t_handle)
        raise
    finally:
        _open_segment(graph, stream)
    prog.append(_If(condition, t, f, host))
    if apic is not None:
        from warp._src.apic.types import APIC_OP_IF  # noqa: PLC0415

        _ctx().runtime.core.wp_apic_record_conditional(apic.apic_state, APIC_OP_IF, region, offset, t_handle, f_handle)


def end_capture(graph, native_graph) -> bool:
    """Close the last segment at capture_end and return whether the graph is a program. The end of any capture also frees
    the words kept while it was open."""
    if getattr(graph, "_hip_program", None) is None:
        _free_kept_words()
        return False
    seg = _Segment(native_graph)
    graph._hip_segments.append(seg)
    graph._hip_program.append(seg)
    graph.graph = None
    graph._hip_stack = None
    _live.add(graph)
    _ctx()  # Cache it now so it is never imported during shutdown.
    _free_kept_words()
    return True


def _read(step, stream) -> bool:
    """Return the condition as the last segment that ran left it in the step's pinned word. The copy into the word is a graph node."""
    warp.synchronize_stream(stream)
    return bool(ctypes.cast(step.host.ptr, ctypes.POINTER(ctypes.c_int32)).contents.value)


def _launch_segment(seg, stream):
    ctx = _ctx()
    if seg.nodes < 0:
        seg.nodes = ctx.runtime.core.wp_cuda_graph_node_count(seg.graph)
    if seg.nodes == 0:
        return
    if seg.exec is None:
        g = ctypes.c_void_p()
        if not ctx.runtime.core.wp_cuda_graph_create_exec(stream.device.context, stream.cuda_stream, seg.graph, ctypes.byref(g)):
            raise RuntimeError(f"Graph creation error: {ctx.runtime.get_error_string()}")
        seg.exec = g
    if not ctx.runtime.core.wp_cuda_graph_launch(seg.exec, stream.cuda_stream):
        raise RuntimeError(f"Graph launch error: {ctx.runtime.get_error_string()}")


def run(program, stream):
    ctx = _ctx()
    for step in program:
        if isinstance(step, _Segment):
            _launch_segment(step, stream)
        elif isinstance(step, _While):
            k = max(1, int(warp.config.hip_conditional_check_every))
            passes = 0
            while _read(step, stream):
                for _ in range(k):
                    run(step.body, stream)
                    passes += 1
            step.last_passes = passes
            step.total_passes += passes
            step.launches += 1
        elif isinstance(step, _If):
            run(step.on_true if _read(step, stream) else step.on_false, stream)
        elif isinstance(step, _GraphRef):
            ctx.capture_launch(step.graph, stream=stream)
        else:
            raise TypeError(f"unknown program step {step!r}")


def stats(graph) -> list[dict]:
    """Return each emulated loop of a graph in program order, with its passes at the last launch, its total passes, and its launches."""
    out = []

    def walk(prog):
        for step in prog:
            if isinstance(step, _While):
                out.append({"last_passes": step.last_passes, "total_passes": step.total_passes, "launches": step.launches})
                walk(step.body)
            elif isinstance(step, _If):
                walk(step.on_true)
                walk(step.on_false)

    if getattr(graph, "_hip_program", None) is not None:
        walk(graph._hip_program)
    return out


def destroy(graph):
    """Free the segments' native graphs and executables, and any arrays that only this graph still holds. Called from
    Graph.__del__ or at exit."""
    ctx = _context_module
    segs = getattr(graph, "_hip_segments", None) or ()
    graph._hip_segments = None
    try:
        _return_words(getattr(graph, "_hip_program", None))
    except (TypeError, AttributeError):
        pass
    if ctx is None or ctx.runtime is None or ctx.runtime.core is None:
        return
    try:
        _release_graph_arrays(graph)
    except (TypeError, AttributeError):
        pass
    for seg in segs:
        try:
            if seg.exec is not None:
                ctx.runtime.core.wp_cuda_graph_exec_destroy(graph.device.context, seg.exec)
                seg.exec = None
            if seg.graph:
                ctx.runtime.core.wp_cuda_graph_destroy(graph.device.context, seg.graph)
                seg.graph = None
        except (TypeError, AttributeError):
            pass


@atexit.register
def _destroy_all():
    ctx = _context_module
    if ctx is None or ctx.runtime is None:
        return
    for graph in list(_live):
        try:
            with graph.device.context_guard:
                destroy(graph)
        except Exception:  # noqa: BLE001
            pass
