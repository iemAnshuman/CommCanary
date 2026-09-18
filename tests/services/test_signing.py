from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest

from commcanary.errors import SchemaError
from commcanary.workflows.signing import (
    public_key_sha256,
    sign_ed25519_payload,
    verify_ed25519_payload,
)


def _keypair(directory: Path, name: str) -> tuple[Path, Path]:
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL is not installed")
    reported = subprocess.run(["openssl", "version"], capture_output=True, text=True).stdout
    if not reported.startswith("OpenSSL ") or int(reported.split()[1].split(".")[0]) < 3:
        pytest.skip(f"Ed25519 needs OpenSSL 3; openssl reports {reported.strip()!r}")
    private = directory / f"{name}-private.pem"
    public = directory / f"{name}-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    return private, public


def test_detached_ed25519_signature_binds_exact_payload_and_trust_anchor(tmp_path: Path) -> None:
    private, public = _keypair(tmp_path, "owner")
    _other_private, other_public = _keypair(tmp_path, "other")
    payload = b'{"bundle_id":"exact"}\n'

    signature = sign_ed25519_payload(
        payload,
        private_key=private,
        expected_public_key=public,
        signed_artifact="manifest.json",
    )

    assert signature["public_key"]["sha256"] == public_key_sha256(public)
    assert verify_ed25519_payload(
        payload,
        signature,
        expected_signed_artifact="manifest.json",
        trusted_public_key=public,
    ) == public_key_sha256(public)
    with pytest.raises(SchemaError, match="payload SHA-256"):
        verify_ed25519_payload(
            payload + b" ",
            signature,
            expected_signed_artifact="manifest.json",
            trusted_public_key=public,
        )
    with pytest.raises(SchemaError, match="trusted owner"):
        verify_ed25519_payload(
            payload,
            signature,
            expected_signed_artifact="manifest.json",
            trusted_public_key=other_public,
        )

    forged = copy.deepcopy(signature)
    forged["signature_base64"] = "A" * 88
    with pytest.raises(SchemaError, match="canonical 64-byte base64"):
        verify_ed25519_payload(
            payload,
            forged,
            expected_signed_artifact="manifest.json",
            trusted_public_key=public,
        )


@pytest.mark.parametrize("name", [".", ".."])
def test_rejects_current_and_parent_as_artifact_name(tmp_path: Path, name: str) -> None:
    private, _ = _keypair(tmp_path, "owner")
    with pytest.raises(SchemaError, match="canonical basename"):
        sign_ed25519_payload(
            b"x",
            private_key=private,
            signed_artifact=name,
        )
    with pytest.raises(SchemaError, match="canonical basename"):
        verify_ed25519_payload(
            b"x",
            {},
            expected_signed_artifact=name,
        )


@pytest.mark.parametrize("reported", ["LibreSSL 3.3.6", "OpenSSL 1.1.1w  11 Sep 2023", ""])
def test_an_openssl_without_ed25519_is_named_before_it_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reported: str
) -> None:
    """A stock macOS openssl is LibreSSL, which has no Ed25519. Signing used
    to die inside it with "unable to load key", naming neither cause nor fix."""

    from commcanary.errors import CommCanaryError
    from commcanary.workflows import signing

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "openssl"
    fake.write_text(f"#!/bin/sh\necho '{reported}'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setattr(signing, "_OPENSSL_VERSIONS", {})
    key = tmp_path / "owner-private.pem"
    key.write_bytes(b"not a key")
    key.chmod(0o600)
    with pytest.raises(CommCanaryError, match="needs OpenSSL 3.0 or newer.*LibreSSL"):
        sign_ed25519_payload(b"payload", private_key=key, signed_artifact="bundle.json")
