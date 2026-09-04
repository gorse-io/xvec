#!/usr/bin/env python3
"""Normalize clang assembly for GoAT's amd64 parser."""

from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile


def encode_constant_blocks(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not re.match(r"^LCPI[0-9_]+:$", line):
            output.append(line)
            index += 1
            continue
        output.append(line)
        data = bytearray()
        index += 1
        while index < len(lines):
            stripped = lines[index].strip().split("#", 1)[0].strip()
            if re.match(r"^(?:LCPI[0-9_]+:|\.section|\.text|\.globl)", stripped):
                break
            if stripped.startswith(".zero"):
                values = [int(value.strip(), 0) for value in stripped[5:].split(",")]
                data.extend(bytes([values[1] if len(values) > 1 else 0]) * values[0])
            elif stripped.startswith(".byte"):
                data.extend(int(value.strip(), 0) & 0xff for value in stripped[5:].split(","))
            elif stripped.startswith((".short", ".value")):
                values = stripped.split(None, 1)[1].split(",")
                for value in values:
                    data.extend(struct.pack("<H", int(value.strip(), 0) & 0xffff))
            elif stripped.startswith(".long"):
                for value in stripped[5:].split(","):
                    data.extend(struct.pack("<I", int(value.strip(), 0) & 0xffffffff))
            elif stripped.startswith(".quad"):
                for value in stripped[5:].split(","):
                    data.extend(struct.pack("<Q", int(value.strip(), 0) & 0xffffffffffffffff))
            index += 1
        escaped = "".join(f"\\{byte:03o}" for byte in data)
        output.append(f'\t.ascii\t"{escaped}"')
    return "\n".join(output) + "\n"


def normalize(path: Path) -> None:
    source = path.read_text().replace(".LCPI", "LCPI")
    source = encode_constant_blocks(source)
    output: list[str] = []
    constant_ref = re.compile(r"(LCPI[0-9_]+)\(%rip\)")
    for line in source.splitlines():
        match = constant_ref.search(line)
        if match is not None and not line.lstrip().startswith("leaq"):
            symbol = match.group(1)
            output.extend(
                [
                    "\tpushq\t%r11",
                    f"\tleaq\t{symbol}(%rip), %r11",
                    constant_ref.sub("(%r11)", line),
                    "\tpopq\t%r11",
                ]
            )
        else:
            output.append(line)
    source = "\n".join(output) + "\n"
    source = source.replace("\n\t.text\n", "\n\t.section\t.text,\"ax\",@progbits\n")
    path.write_text(source)


args = sys.argv[1:]
if "-c" in args and "-S" not in args and "-o" in args and any(arg.endswith(".c") for arg in args):
    output = Path(args[args.index("-o") + 1])
    with tempfile.TemporaryDirectory() as directory:
        assembly = Path(directory) / "source.s"
        asm_args = list(args)
        asm_args[asm_args.index("-c")] = "-S"
        asm_args[asm_args.index("-o") + 1] = str(assembly)
        result = subprocess.run(["/usr/bin/clang", *asm_args], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        normalize(assembly)
        raise SystemExit(subprocess.run(["/usr/bin/clang", "-c", str(assembly), "-o", str(output)], check=False).returncode)

result = subprocess.run(["/usr/bin/clang", *args], check=False)
if result.returncode == 0 and "-S" in args and "-o" in args:
    normalize(Path(args[args.index("-o") + 1]))
raise SystemExit(result.returncode)
