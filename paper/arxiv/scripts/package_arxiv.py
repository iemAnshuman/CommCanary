#!/usr/bin/env python3
"""Package only the LaTeX inputs needed by arXiv, at the ZIP root.

Build first so main.bbl corresponds to the current manuscript. The larger
package_source.py archive retains the analysis scripts and evidence separately.
"""

from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_arxiv.py OUTPUT.zip")
    main_tex = (SOURCE / "main.tex").read_bytes()
    files = {"main.tex": main_tex, "references.bib": (SOURCE / "references.bib").read_bytes()}
    for relative in re.findall(r"\\input\{([^}]+)\}", main_tex.decode()):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe TeX input: {relative}")
        files[relative] = (SOURCE / path).read_bytes()
    bbl = ROOT / "output/pdf/main.bbl"
    if not bbl.is_file():
        raise SystemExit("main.bbl is missing; run paper/arxiv/scripts/build.sh first")
    newest_source = max((SOURCE / name).stat().st_mtime for name in files)
    if bbl.stat().st_mtime < newest_source:
        raise SystemExit("main.bbl predates a manuscript input; rebuild before packaging")
    files["main.bbl"] = bbl.read_bytes()
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise SystemExit("arXiv ZIP failed inventory or integrity verification")
        for name, data in files.items():
            if archive.read(name) != data:
                raise SystemExit(f"arXiv ZIP content mismatch: {name}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"verified arXiv upload ZIP: {len(files)} files, sha256={digest}")


if __name__ == "__main__":
    main()
