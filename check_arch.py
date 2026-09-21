"""CPU-build guard that checks CUDA kernels embedded in the torch wheel."""

from __future__ import annotations

import pathlib
import re
import sys

import torch

CHUNK = 32 * 1024 * 1024


def compiled_archs() -> set[str]:
    lib_dir = pathlib.Path(torch.__file__).parent / "lib"
    libraries = sorted(lib_dir.glob("libtorch_cuda*.so*"))
    if not libraries:
        raise SystemExit(f"no libtorch_cuda found under {lib_dir}")
    pattern = re.compile(rb"sm_(\d{2,3})[af]?")
    found: set[str] = set()
    for library in libraries:
        with library.open("rb") as source:
            tail = b""
            while block := source.read(CHUNK):
                found.update(f"sm_{match.group(1).decode()}" for match in pattern.finditer(tail + block))
                tail = block[-16:]
    return found


def main() -> None:
    required = sys.argv[1:] or ["sm_89", "sm_120"]
    archs = compiled_archs()
    missing = [arch for arch in required if arch not in archs]
    print(f"torch={torch.__version__} cuda={torch.version.cuda} archs={' '.join(sorted(archs))}")
    if missing:
        raise SystemExit(f"torch wheel is missing kernels for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
