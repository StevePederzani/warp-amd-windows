# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

"""Read an AMD code object's own description of its kernels.

The AMDGPU compiler writes an ELF note (``NT_AMDGPU_METADATA``, owner ``AMDGPU``) into every code object it emits. The note holds
a MessagePack map of the code object's kernels and what each asks of the GPU, including its static LDS
(``.group_segment_fixed_size``), its largest block (``.max_flat_workgroup_size``), its registers, and its private memory. Warp
reads it to learn a kernel's needs from the binary alone, before loading it or without loading it at all. The virtual shared
memory plan in ``context.py`` uses it to weigh a kernel's static and dynamic shared memory together, and a test reads a kernel's
bounds as the compiler recorded them. Only the MessagePack types the metadata uses are read, and anything else is refused rather
than guessed.
"""

from __future__ import annotations

import struct

NT_AMDGPU_METADATA = 32
SHT_NOTE = 7


def _unpack(b: bytes, i: int):
    """Read one MessagePack value at ``i`` and return (value, the offset after it). Multi-byte numbers are big-endian, as the format defines."""
    t = b[i]
    if t <= 0x7F:
        return t, i + 1
    if 0x80 <= t <= 0x8F:
        return _map(b, i + 1, t & 0x0F)
    if 0x90 <= t <= 0x9F:
        return _array(b, i + 1, t & 0x0F)
    if 0xA0 <= t <= 0xBF:
        n = t & 0x1F
        return b[i + 1 : i + 1 + n].decode("utf-8", "replace"), i + 1 + n
    if t >= 0xE0:
        return t - 0x100, i + 1
    if t == 0xC0:
        return None, i + 1
    if t == 0xC2:
        return False, i + 1
    if t == 0xC3:
        return True, i + 1
    sized = {0xC4: ">B", 0xC5: ">H", 0xC6: ">I"}
    if t in sized:
        (n,) = struct.unpack_from(sized[t], b, i + 1)
        at = i + 1 + struct.calcsize(sized[t])
        return bytes(b[at : at + n]), at + n
    strs = {0xD9: ">B", 0xDA: ">H", 0xDB: ">I"}
    if t in strs:
        (n,) = struct.unpack_from(strs[t], b, i + 1)
        at = i + 1 + struct.calcsize(strs[t])
        return b[at : at + n].decode("utf-8", "replace"), at + n
    numbers = {
        0xCA: ">f", 0xCB: ">d",
        0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q",
        0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q",
    }  # fmt: skip
    if t in numbers:
        (v,) = struct.unpack_from(numbers[t], b, i + 1)
        return v, i + 1 + struct.calcsize(numbers[t])
    if t in (0xDC, 0xDD):
        fmt = ">H" if t == 0xDC else ">I"
        (n,) = struct.unpack_from(fmt, b, i + 1)
        return _array(b, i + 1 + struct.calcsize(fmt), n)
    if t in (0xDE, 0xDF):
        fmt = ">H" if t == 0xDE else ">I"
        (n,) = struct.unpack_from(fmt, b, i + 1)
        return _map(b, i + 1 + struct.calcsize(fmt), n)
    raise ValueError(f"MessagePack type 0x{t:02x} at {i} is not one the AMDGPU metadata uses")


def _array(b: bytes, i: int, n: int):
    out = []
    for _ in range(n):
        v, i = _unpack(b, i)
        out.append(v)
    return out, i


def _map(b: bytes, i: int, n: int):
    out = {}
    for _ in range(n):
        k, i = _unpack(b, i)
        v, i = _unpack(b, i)
        out[k] = v
    return out, i


def metadata(blob: bytes) -> dict | None:
    """Return the code object's AMDGPU metadata map, or None when ``blob`` is not a 64-bit ELF that carries one."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return None
    (e_shoff,) = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3A)
    for k in range(e_shnum):
        sh = e_shoff + k * e_shentsize
        if sh + 0x28 > len(blob):
            break
        (sh_type,) = struct.unpack_from("<I", blob, sh + 4)
        if sh_type != SHT_NOTE:
            continue
        sh_offset, sh_size = struct.unpack_from("<QQ", blob, sh + 0x18)
        p, end = sh_offset, min(len(blob), sh_offset + sh_size)
        while p + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", blob, p)
            name = bytes(blob[p + 12 : p + 12 + namesz]).rstrip(b"\0")
            desc_at = p + 12 + ((namesz + 3) & ~3)
            if ntype == NT_AMDGPU_METADATA and name == b"AMDGPU":
                value, _ = _unpack(blob, desc_at)
                return value if isinstance(value, dict) else None
            p = desc_at + ((descsz + 3) & ~3)
    return None


def _symbols(blob: bytes) -> dict[str, tuple[int, int]]:
    """Return every named symbol of a 64-bit ELF with its file offset and size, mapping its address through the section that holds it."""
    (e_shoff,) = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3A)
    sections = []
    for k in range(e_shnum):
        sh = e_shoff + k * e_shentsize
        sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link = struct.unpack_from("<IQQQQI", blob, sh + 4)
        sections.append((sh_type, sh_addr, sh_offset, sh_size, sh_link))
    out: dict[str, tuple[int, int]] = {}
    for sh_type, _addr, sh_offset, sh_size, sh_link in sections:
        if sh_type not in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            continue
        str_offset = sections[sh_link][2]
        for p in range(sh_offset, sh_offset + sh_size, 24):
            st_name, _info, _other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", blob, p)
            if not st_name or st_shndx == 0 or st_shndx >= len(sections):
                continue
            end = blob.index(b"\0", str_offset + st_name)
            name = bytes(blob[str_offset + st_name : end]).decode("utf-8", "replace")
            _t, s_addr, s_offset, _s, _l = sections[st_shndx]
            out[name] = (s_offset + (st_value - s_addr), st_size)
    return out


def _allocated_vgprs(blob: bytes, symbols: dict[str, tuple[int, int]], name: str) -> int:
    """Return the vector registers the hardware reserves for each thread of kernel ``name``, from its kernel descriptor
    (``<name>.kd``). The count is COMPUTE_PGM_RSRC1's granulated count at offset 48 of the 64-byte descriptor. A granule is 8
    registers on RDNA (GFX10 to GFX12) in 32-wide waves and 4 in 64-wide waves (see AMDGPU's documentation of compute_pgm_rsrc1),
    and bit 10 of the descriptor's kernel_code_properties says which applies. Even an empty kernel holds one granule. Returns 0
    when the descriptor can't be read."""
    at = symbols.get(name + ".kd")
    if not at or at[0] + 64 > len(blob):
        return 0
    (rsrc1,) = struct.unpack_from("<I", blob, at[0] + 48)
    (code_props,) = struct.unpack_from("<H", blob, at[0] + 56)
    granule = 8 if code_props & (1 << 10) else 4
    return ((rsrc1 & 0x3F) + 1) * granule


def kernel_resources(blob: bytes) -> dict[str, dict[str, int]]:
    """Return each kernel's needs as its code object records them, by kernel name. These are ``static_lds`` (bytes of LDS fixed
    at compile time), ``max_threads`` (the largest block it was compiled for), ``vgprs`` (the registers its code uses),
    ``vgprs_allocated`` (the registers the hardware reserves for each of its threads), ``sgprs``, and ``private`` (bytes of
    scratch per thread)."""
    out: dict[str, dict[str, int]] = {}
    try:
        symbols = _symbols(blob) if blob[:4] == b"\x7fELF" else {}
    except (struct.error, ValueError, IndexError):
        symbols = {}
    for k in (metadata(blob) or {}).get("amdhsa.kernels", []) or []:
        if not isinstance(k, dict) or not k.get(".name"):
            continue
        name = str(k[".name"])
        out[name] = {
            "static_lds": int(k.get(".group_segment_fixed_size", 0) or 0),
            "max_threads": int(k.get(".max_flat_workgroup_size", 0) or 0),
            "vgprs": int(k.get(".vgpr_count", 0) or 0),
            "vgprs_allocated": _allocated_vgprs(blob, symbols, name),
            "sgprs": int(k.get(".sgpr_count", 0) or 0),
            "private": int(k.get(".private_segment_fixed_size", 0) or 0),
        }
    return out


def kernel_resources_of_file(path: str) -> dict[str, dict[str, int]]:
    with open(path, "rb") as f:
        return kernel_resources(f.read())
