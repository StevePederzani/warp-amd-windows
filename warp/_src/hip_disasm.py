# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

"""Read an AMD code object as text, listing its functions and the instructions the GPU runs, using AMD's Code Object Manager.

On NVIDIA, a PTX request (``use_ptx``) produces text that people and tests read, for example to check that an inline attribute
was honored or that a division is approximate. On AMD the code is the code object hipRTC built, and its text is its disassembly.
It is read here by ``amd_comgr``, the library hipRTC itself uses, which ships next to it in Warp's ``bin``, and never by an
outside tool. Each function symbol is written as ``<name>:`` followed by its instructions as the GPU decodes them, one at a time,
using comgr's per-instruction disassembler, because its whole-object disassembly actions are deprecated and refuse to run.
Handles are passed as the 64-bit integers they are, since a one-field struct of a uint64 travels in a register on both the
Windows x64 and System V ABIs.
"""

from __future__ import annotations

import ctypes
import os
import struct

_SUCCESS = 0
_DATA_KIND_EXECUTABLE = 0x8

_READ = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
_PRINT = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
_ANNOTATE = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p)

_COMGR = None


def _declare(lib) -> None:
    u64, ptr = ctypes.c_uint64, ctypes.POINTER
    sig = {
        "amd_comgr_create_data": [ctypes.c_int, ptr(u64)],
        "amd_comgr_set_data": [u64, ctypes.c_size_t, ctypes.c_char_p],
        "amd_comgr_release_data": [u64],
        "amd_comgr_get_data_isa_name": [u64, ptr(ctypes.c_size_t), ctypes.c_char_p],
        "amd_comgr_create_disassembly_info": [ctypes.c_char_p, _READ, _PRINT, _ANNOTATE, ptr(u64)],
        "amd_comgr_destroy_disassembly_info": [u64],
        "amd_comgr_disassemble_instruction": [u64, u64, ctypes.c_void_p, ptr(u64)],
    }
    for name, args in sig.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = ctypes.c_int


def _comgr():
    """Load amd_comgr from Warp's own bin, where the HIP build ships it next to hipRTC, or else from the loader's search path.

    Returns None when there is none."""
    global _COMGR
    if _COMGR is not None:
        return _COMGR or None
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
    names = ["amd_comgr.dll"] if os.name == "nt" else ["libamd_comgr.so.3", "libamd_comgr.so"]
    for name in names:
        for path in (os.path.join(here, name), name):
            try:
                lib = ctypes.CDLL(path)
                _declare(lib)
            except (OSError, AttributeError):
                continue
            _COMGR = lib
            return lib
    _COMGR = False
    return None


def _sections(blob: bytes) -> list[tuple[int, int, int]]:
    """Return (address, file offset, size) for each section of a 64-bit ELF that holds bytes in the file, skipping SHT_NOBITS."""
    (e_shoff,) = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3A)
    out = []
    for k in range(e_shnum):
        sh = e_shoff + k * e_shentsize
        sh_type, _flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IQQQQ", blob, sh + 4)
        if sh_type not in (0, 8) and sh_size:  # (SHT_NULL, SHT_NOBITS)
            out.append((sh_addr, sh_offset, sh_size))
    return out


def _functions(blob: bytes) -> list[tuple[int, int, str]]:
    """Return (address, size, name) for every defined function symbol (STT_FUNC) in the ELF's symbol tables, including local ones.
    A helper kept out of line is a local symbol, which comgr's own symbol walk, covering only exported symbols, doesn't visit."""
    (e_shoff,) = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3A)
    heads = []
    for k in range(e_shnum):
        sh = e_shoff + k * e_shentsize
        heads.append(struct.unpack_from("<IQQQQI", blob, sh + 4))  # (type, flags, addr, offset, size, link)
    seen: set[tuple[int, str]] = set()
    out = []
    for sh_type, _flags, _addr, sh_offset, sh_size, sh_link in heads:
        if sh_type not in (2, 11):  # (SHT_SYMTAB, SHT_DYNSYM)
            continue
        strings = heads[sh_link][3]
        for p in range(sh_offset, sh_offset + sh_size, 24):
            st_name, st_info, _other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", blob, p)
            if (st_info & 0xF) != 2 or st_shndx == 0 or not st_name:
                continue
            end = blob.index(b"\0", strings + st_name)
            name = bytes(blob[strings + st_name : end]).decode("utf-8", "replace")
            if (st_value, name) not in seen:
                seen.add((st_value, name))
                out.append((st_value, st_size, name))
    return sorted(out)


def text_of(blob: bytes) -> str | None:
    """Return a code object's functions and instructions as text, or None when comgr is missing or the blob is not an ELF code object."""
    lib = _comgr()
    if lib is None or blob[:4] != b"\x7fELF":
        return None
    data = ctypes.c_uint64(0)
    if lib.amd_comgr_create_data(_DATA_KIND_EXECUTABLE, ctypes.byref(data)) != _SUCCESS:
        return None
    try:
        if lib.amd_comgr_set_data(data.value, len(blob), blob) != _SUCCESS:
            return None
        n = ctypes.c_size_t(0)
        if lib.amd_comgr_get_data_isa_name(data.value, ctypes.byref(n), None) != _SUCCESS or not n.value:
            return None
        isa = ctypes.create_string_buffer(n.value)
        if lib.amd_comgr_get_data_isa_name(data.value, ctypes.byref(n), isa) != _SUCCESS:
            return None
    finally:
        lib.amd_comgr_release_data(data.value)
    try:
        funcs = _functions(blob)
    except (struct.error, ValueError, IndexError):
        return None

    sections = _sections(blob)
    pieces: list[str] = []

    def read(addr: int, to, size: int, _user) -> int:
        for s_addr, s_offset, s_size in sections:
            if s_addr <= addr < s_addr + s_size:
                k = min(size, s_addr + s_size - addr)
                start = s_offset + (addr - s_addr)
                ctypes.memmove(to, blob[start : start + k], k)
                return k
        return 0

    def on_instruction(text: bytes, _user) -> None:
        pieces.append(text.decode("utf-8", "replace"))

    def on_address(addr: int, _user) -> None:
        pieces.append(f" <0x{addr:x}>")

    read_c, print_c, annotate_c = _READ(read), _PRINT(on_instruction), _ANNOTATE(on_address)
    info = ctypes.c_uint64(0)
    if lib.amd_comgr_create_disassembly_info(isa.value, read_c, print_c, annotate_c, ctypes.byref(info)) != _SUCCESS:
        return None
    lines = [f"; {isa.value.decode('ascii', 'replace')}: {len(funcs)} functions, read by amd_comgr"]
    try:
        for value, size, name in sorted(funcs):
            lines.append("")
            lines.append(f"{value:016x} <{name}>:")
            addr = value
            while addr < value + size:
                pieces.clear()
                used = ctypes.c_uint64(0)
                status = lib.amd_comgr_disassemble_instruction(info.value, addr, None, ctypes.byref(used))
                if not used.value:
                    break
                text = "".join(pieces).strip() if status == _SUCCESS else "<not an instruction>"
                lines.append(f"\t{text}\t// {addr:012X}")
                addr += used.value
    finally:
        lib.amd_comgr_destroy_disassembly_info(info.value)
    return "\n".join(lines) + "\n"
