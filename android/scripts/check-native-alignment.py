#!/usr/bin/env python3
"""Check every packaged 64-bit ELF and uncompressed ZIP entry for 16 KB alignment."""
import struct
import sys
from zipfile import ZipFile, ZIP_STORED

apk = sys.argv[1]
failures = []
checked = 0
with ZipFile(apk) as archive, open(apk, "rb") as raw:
    for entry in archive.infolist():
        if not entry.filename.endswith(".so"):
            continue
        elf = archive.read(entry)
        if elf[:4] != b"\x7fELF":
            failures.append(f"{entry.filename}: invalid ELF")
            continue
        if elf[4] != 2:  # Android's 16 KB requirement applies to 64-bit ABIs.
            continue
        endian = "<" if elf[5] == 1 else ">"
        phoff = struct.unpack_from(endian + "Q", elf, 32)[0]
        phsize, phnum = struct.unpack_from(endian + "HH", elf, 54)
        loads = 0
        for index in range(phnum):
            kind, flags, offset, address, physical, filesz, memsz, alignment = struct.unpack_from(
                endian + "IIQQQQQQ", elf, phoff + index * phsize
            )
            if kind == 1:
                loads += 1
                if alignment < 16384 or (address - offset) % 16384:
                    failures.append(f"{entry.filename}: LOAD alignment {alignment}")
        if not loads:
            failures.append(f"{entry.filename}: no LOAD segments")
        if entry.compress_type == ZIP_STORED:
            raw.seek(entry.header_offset + 26)
            name_size, extra_size = struct.unpack("<HH", raw.read(4))
            if (entry.header_offset + 30 + name_size + extra_size) % 16384:
                failures.append(f"{entry.filename}: ZIP data is not 16 KB aligned")
        checked += 1
        print(entry.filename)
if not checked:
    failures.append("No 64-bit libraries found")
if failures:
    sys.exit("\n".join(failures))
print(f"PASS: {checked} packaged 64-bit libraries satisfy 16 KB ELF/ZIP alignment")
