"""Detached Ed25519 signatures for owner-issued exchange artifacts.

The optional signing surface shells out to OpenSSL instead of introducing a
runtime Python dependency or reading private-key bytes into the process.  A
receiver must still pin an independently obtained owner public key: a valid
self-contained signature proves integrity, not the signer's identity.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..errors import CommCanaryError, SchemaError

Pathish = Union[str, Path]
DETACHED_SIGNATURE_FORMAT = "commcanary.detached_ed25519_signature.v1"
ED25519_ALGORITHM = "Ed25519"
MAX_SIGNED_BYTES = 16 * 1024 * 1024
MAX_KEY_BYTES = 64 * 1024
OPENSSL_TIMEOUT_SECONDS = 20


def sign_ed25519_payload(
    payload: bytes,
    *,
    private_key: Pathish,
    signed_artifact: str,
    expected_public_key: Optional[Pathish] = None,
) -> Dict[str, Any]:
    """Sign exact bytes and embed the normalized public verification key."""

    _validate_payload(payload)
    _canonical_artifact_name(signed_artifact)
    private_path = _safe_key_path(private_key, "private key")
    public_pem = _derive_public_pem(private_path)
    public_der = _public_der_from_bytes(public_pem)
    if expected_public_key is not None:
        expected_der = _public_der_from_path(_safe_key_path(expected_public_key, "expected public key"))
        if not _constant_time_equal(public_der, expected_der):
            raise CommCanaryError("Ed25519 private key does not match the expected owner public key")

    with tempfile.TemporaryDirectory(prefix="commcanary-sign-") as temporary:
        payload_path = Path(temporary) / "payload"
        _write_private_temp(payload_path, payload)
        signature = _openssl(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_path),
                "-in",
                str(payload_path),
            ],
            failure="OpenSSL could not create an Ed25519 signature",
        )
    if len(signature) != 64:
        raise CommCanaryError(f"OpenSSL returned an invalid Ed25519 signature length: {len(signature)}")
    return {
        "format": DETACHED_SIGNATURE_FORMAT,
        "algorithm": ED25519_ALGORITHM,
        "signed_artifact": signed_artifact,
        "signed_sha256": hashlib.sha256(payload).hexdigest(),
        "public_key": {
            "encoding": "subject-public-key-info-pem",
            "sha256": hashlib.sha256(public_der).hexdigest(),
            "pem": public_pem.decode("ascii"),
        },
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_ed25519_payload(
    payload: bytes,
    signature_document: Mapping[str, Any],
    *,
    expected_signed_artifact: str,
    trusted_public_key: Optional[Pathish] = None,
) -> str:
    """Verify exact bytes and return the embedded public-key SHA-256.

    When ``trusted_public_key`` is omitted, this verifies only internal
    integrity. Private exchange callers should always pass a separately
    obtained trust anchor.
    """

    _validate_payload(payload)
    _canonical_artifact_name(expected_signed_artifact)
    expected_fields = {
        "format",
        "algorithm",
        "signed_artifact",
        "signed_sha256",
        "public_key",
        "signature_base64",
    }
    if set(signature_document) != expected_fields:
        raise SchemaError("detached signature fields are not closed")
    if signature_document.get("format") != DETACHED_SIGNATURE_FORMAT:
        raise SchemaError("detached signature format is unsupported")
    if signature_document.get("algorithm") != ED25519_ALGORITHM:
        raise SchemaError("detached signature algorithm is unsupported")
    if signature_document.get("signed_artifact") != expected_signed_artifact:
        raise SchemaError("detached signature names the wrong artifact")
    expected_payload_sha256 = hashlib.sha256(payload).hexdigest()
    if signature_document.get("signed_sha256") != expected_payload_sha256:
        raise SchemaError("detached signature payload SHA-256 does not match exact bytes")

    public = signature_document.get("public_key")
    if not isinstance(public, Mapping) or set(public) != {"encoding", "sha256", "pem"}:
        raise SchemaError("detached signature public_key is malformed")
    if public.get("encoding") != "subject-public-key-info-pem":
        raise SchemaError("detached signature public-key encoding is unsupported")
    public_pem_text = public.get("pem")
    if not isinstance(public_pem_text, str) or not public_pem_text.endswith("\n"):
        raise SchemaError("detached signature public key must be canonical PEM text")
    try:
        public_pem = public_pem_text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SchemaError("detached signature public key must contain ASCII PEM text") from exc
    if len(public_pem) > MAX_KEY_BYTES:
        raise SchemaError("detached signature public key exceeds its byte limit")
    public_der = _public_der_from_bytes(public_pem, verification=True)
    public_sha256 = hashlib.sha256(public_der).hexdigest()
    if public.get("sha256") != public_sha256:
        raise SchemaError("detached signature public-key SHA-256 does not recompute")
    normalized_pem = _public_pem_from_bytes(public_pem, verification=True)
    if normalized_pem != public_pem:
        raise SchemaError("detached signature public key is not canonical OpenSSL PEM")

    if trusted_public_key is not None:
        trusted_der = _public_der_from_path(_safe_key_path(trusted_public_key, "trusted public key"), verification=True)
        if not _constant_time_equal(public_der, trusted_der):
            raise SchemaError("detached signature key does not match the trusted owner public key")

    encoded_signature = signature_document.get("signature_base64")
    if not isinstance(encoded_signature, str):
        raise SchemaError("detached signature_base64 must be a string")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SchemaError("detached signature_base64 is invalid") from exc
    if len(signature) != 64 or base64.b64encode(signature).decode("ascii") != encoded_signature:
        raise SchemaError("detached Ed25519 signature must use canonical 64-byte base64")

    with tempfile.TemporaryDirectory(prefix="commcanary-verify-") as temporary:
        root = Path(temporary)
        payload_path = root / "payload"
        signature_path = root / "signature"
        public_path = root / "public.pem"
        _write_private_temp(payload_path, payload)
        _write_private_temp(signature_path, signature)
        _write_private_temp(public_path, public_pem)
        _openssl(
            [
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(public_path),
                "-sigfile",
                str(signature_path),
                "-in",
                str(payload_path),
            ],
            failure="detached Ed25519 signature verification failed",
            verification=True,
        )
    return public_sha256


def public_key_sha256(public_key: Pathish) -> str:
    """Return the SHA-256 of canonical SubjectPublicKeyInfo DER bytes."""

    return hashlib.sha256(_public_der_from_path(_safe_key_path(public_key, "public key"))).hexdigest()


def _validate_payload(payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("signed payload must be bytes")
    if not payload or len(payload) > MAX_SIGNED_BYTES:
        raise SchemaError(f"signed payload bytes must be in [1, {MAX_SIGNED_BYTES}]")


def _canonical_artifact_name(value: str) -> None:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise SchemaError("signed artifact must be a canonical basename")


def _safe_key_path(value: Pathish, label: str) -> Path:
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
        stat_result = resolved.stat()
    except OSError as exc:
        raise CommCanaryError(f"cannot read Ed25519 {label} {path}: {exc}") from exc
    if path.is_symlink() or not resolved.is_file() or stat_result.st_size < 1 or stat_result.st_size > MAX_KEY_BYTES:
        raise CommCanaryError(f"Ed25519 {label} is missing, unsafe, or outside its byte limit: {path}")
    return resolved


def _derive_public_pem(private_path: Path) -> bytes:
    return _openssl(
        ["pkey", "-in", str(private_path), "-pubout", "-outform", "PEM"],
        failure="OpenSSL could not derive the Ed25519 public key",
    )


def _public_der_from_path(path: Path, *, verification: bool = False) -> bytes:
    return _openssl(
        ["pkey", "-pubin", "-in", str(path), "-pubout", "-outform", "DER"],
        failure="OpenSSL could not parse the Ed25519 public key",
        verification=verification,
    )


def _public_der_from_bytes(value: bytes, *, verification: bool = False) -> bytes:
    with tempfile.TemporaryDirectory(prefix="commcanary-public-key-") as temporary:
        path = Path(temporary) / "public.pem"
        _write_private_temp(path, value)
        return _public_der_from_path(path, verification=verification)


def _public_pem_from_bytes(value: bytes, *, verification: bool = False) -> bytes:
    with tempfile.TemporaryDirectory(prefix="commcanary-public-key-") as temporary:
        path = Path(temporary) / "public.pem"
        _write_private_temp(path, value)
        return _openssl(
            ["pkey", "-pubin", "-in", str(path), "-pubout", "-outform", "PEM"],
            failure="OpenSSL could not normalize the Ed25519 public key",
            verification=verification,
        )


def _write_private_temp(path: Path, value: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


_OPENSSL_VERSIONS: Dict[str, str] = {}


def _require_openssl3(*, failure: str, verification: bool) -> None:
    """Refuse an ``openssl`` that cannot do raw Ed25519 before using it.

    ``pkeyutl -rawin`` with Ed25519 needs OpenSSL 3. The ``openssl`` a stock
    macOS puts on PATH is LibreSSL, which has no Ed25519 at all, and it failed
    here with "unable to load key" -- true, and no help in finding the cause.
    """

    error_type = SchemaError if verification else CommCanaryError
    resolved = shutil.which("openssl")
    if resolved is None:
        raise error_type(f"{failure}: openssl is not on PATH; Ed25519 signing needs OpenSSL 3.0 or newer")
    reported = _OPENSSL_VERSIONS.get(resolved)
    if reported is None:
        try:
            completed = subprocess.run(
                [resolved, "version"],
                check=False,
                capture_output=True,
                timeout=OPENSSL_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise error_type(f"{failure}: cannot run {resolved}: {type(exc).__name__}: {exc}") from exc
        reported = completed.stdout.decode("utf-8", errors="replace").strip()
        _OPENSSL_VERSIONS[resolved] = reported
    match = re.match(r"OpenSSL (\d+)\.", reported)
    if match is None or int(match.group(1)) < 3:
        raise error_type(
            f"{failure}: Ed25519 signing needs OpenSSL 3.0 or newer, but openssl on PATH "
            f"({resolved}) reports {reported or 'no version'!r}. The openssl that ships with "
            "macOS is LibreSSL; install OpenSSL 3 and put it first on PATH"
        )


def _openssl(
    arguments: Sequence[str],
    *,
    failure: str,
    verification: bool = False,
) -> bytes:
    _require_openssl3(failure=failure, verification=verification)
    try:
        completed = subprocess.run(
            ["openssl", *arguments],
            check=False,
            capture_output=True,
            timeout=OPENSSL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        error_type = SchemaError if verification else CommCanaryError
        raise error_type(f"{failure}: {type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[:1024].strip()
        error_type = SchemaError if verification else CommCanaryError
        raise error_type(f"{failure}: {detail or f'exit {completed.returncode}'}")
    return completed.stdout


def _constant_time_equal(left: bytes, right: bytes) -> bool:
    return hmac.compare_digest(left, right)


__all__ = [
    "DETACHED_SIGNATURE_FORMAT",
    "ED25519_ALGORITHM",
    "public_key_sha256",
    "sign_ed25519_payload",
    "verify_ed25519_payload",
]
