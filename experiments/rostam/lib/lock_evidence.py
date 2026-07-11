"""Emit hashed lock lines and contract evidence from a downloaded wheelhouse.

This tool writes no contract state itself. It turns a reviewed ``pip
download`` wheelhouse into the two evidence artifacts the environment
contract expects — a complete ``--require-hashes`` lock and the
``wheel_artifacts`` inventory — plus observed interpreter/platform values,
so the operator pastes reviewed output instead of hand-hashing wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class LockEvidenceError(RuntimeError):
    """A wheelhouse cannot be turned into reviewable lock evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_name_version(filename: str) -> Tuple[str, str]:
    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) < 3:
        raise LockEvidenceError(f"unparseable wheel filename: {filename}")
    return parts[0].replace("_", "-").lower(), parts[1]


def _collect_wheels(wheelhouse: Path) -> List[Path]:
    if not wheelhouse.is_dir():
        raise LockEvidenceError(f"wheelhouse is not a directory: {wheelhouse}")
    non_wheels = sorted(p.name for p in wheelhouse.iterdir() if p.is_file() and not p.name.endswith(".whl"))
    if non_wheels:
        raise LockEvidenceError(
            "wheelhouse contains non-wheel artifacts; the reviewed inventory accepts wheels only: "
            + ", ".join(non_wheels)
        )
    wheels = sorted((p for p in wheelhouse.iterdir() if p.name.endswith(".whl")), key=lambda p: p.name.lower())
    if not wheels:
        raise LockEvidenceError(f"wheelhouse holds no wheels: {wheelhouse}")
    return wheels


def build_evidence(wheelhouse: Path, resolver_report: Optional[Path]) -> Dict[str, object]:
    lock_lines: List[str] = []
    inventory: List[Dict[str, object]] = []
    seen: Dict[str, str] = {}
    for wheel in _collect_wheels(wheelhouse):
        name, version = _wheel_name_version(wheel.name)
        if name in seen:
            raise LockEvidenceError(f"duplicate distribution {name!r} in wheelhouse ({seen[name]} and {wheel.name})")
        seen[name] = wheel.name
        digest = _sha256_file(wheel)
        lock_lines.append(f"{name}=={version} --hash=sha256:{digest}")
        inventory.append({"filename": wheel.name, "sha256": digest, "size_bytes": wheel.stat().st_size})
    lock_text = "\n".join(sorted(lock_lines, key=str.lower)) + "\n"
    evidence: Dict[str, object] = {
        "lock_text_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        "wheel_artifacts": inventory,
        "observed_target": {
            "implementation": sys.implementation.name,
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "platform": sysconfig.get_platform(),
            "abi_tag": sysconfig.get_config_var("SOABI"),
        },
    }
    if resolver_report is not None:
        if not resolver_report.is_file():
            raise LockEvidenceError(f"resolver report is not a file: {resolver_report}")
        evidence["resolver_report_sha256"] = _sha256_file(resolver_report)
    evidence["lock_text"] = lock_text
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--resolver-report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = build_evidence(args.wheelhouse, args.resolver_report)
    except LockEvidenceError as error:
        print(f"lock evidence error: {error}", file=sys.stderr)
        return 1
    lock_text = str(evidence.pop("lock_text"))
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_output.write_text(lock_text, encoding="utf-8")
    evidence["lock_path"] = str(args.lock_output)
    json.dump(evidence, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
