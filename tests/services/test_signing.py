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
