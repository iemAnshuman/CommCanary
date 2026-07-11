"""Immutable models for a manifest-driven physical experiment campaign."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

from .canonical import (
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    safe_slug,
    sha256_hex,
    strict_json_loads,
)

CAMPAIGN_SCHEMA = "commcanary.experiment.campaign-spec.v1"
MANIFEST_SCHEMA = "commcanary.experiment.run-manifest.v1"
CELL_ID_MAX_LENGTH = 80
MAX_CAMPAIGN_REPETITIONS = 1_000
MAX_CAMPAIGN_MATRIX_CELLS = 100_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SCHEMA_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ManifestValidationError(ContractError):
    """Raised when a campaign or run manifest violates its contract."""


def _expect_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be an object")
    return value


def _strict_fields(
    value: Mapping[str, Any],
    *,
    field: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - allowed)
    if missing:
        raise ManifestValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ManifestValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _require_str(value: Any, field: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ManifestValidationError(f"{field} must be a non-empty string of at most {max_length} characters")
    return value


def _schema_id(value: Any, field: str) -> str:
    text = _require_str(value, field, max_length=160)
    if not _SCHEMA_ID_RE.fullmatch(text):
        raise ManifestValidationError(f"{field} is not a valid schema identifier")
    return text


def _external_token(value: Any, field: str, *, max_length: int = 128) -> str:
    """Validate an opaque scheduler identifier without changing its spelling."""

    text = _require_str(value, field, max_length=max_length)
    if text != text.strip() or _CONTROL_RE.search(text) or "/" in text or "\\" in text:
        raise ManifestValidationError(
            f"{field} may not contain surrounding whitespace, control characters, or path separators"
        )
    return text


def _sha256(value: Any, field: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ManifestValidationError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ManifestValidationError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


def campaign_matrix_size(configuration_count: int, workload_count: int, repetitions: int) -> int:
    """Preflight the campaign Cartesian product before any cell allocation."""

    for value, field in (
        (configuration_count, "configuration count"),
        (workload_count, "workload count"),
        (repetitions, "repetitions"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ManifestValidationError(f"campaign {field} must be a positive integer")
    if repetitions > MAX_CAMPAIGN_REPETITIONS:
        raise ManifestValidationError(
            f"campaign repetitions exceed MAX_CAMPAIGN_REPETITIONS={MAX_CAMPAIGN_REPETITIONS}"
        )
    if configuration_count > MAX_CAMPAIGN_MATRIX_CELLS // workload_count:
        raise ManifestValidationError(f"campaign matrix exceeds MAX_CAMPAIGN_MATRIX_CELLS={MAX_CAMPAIGN_MATRIX_CELLS}")
    axis_cells = configuration_count * workload_count
    if axis_cells > MAX_CAMPAIGN_MATRIX_CELLS // repetitions:
        raise ManifestValidationError(f"campaign matrix exceeds MAX_CAMPAIGN_MATRIX_CELLS={MAX_CAMPAIGN_MATRIX_CELLS}")
    return axis_cells * repetitions


@dataclass(frozen=True)
class FrozenJSON:
    """An arbitrary JSON value stored internally as immutable canonical bytes."""

    canonical: bytes

    @classmethod
    def from_value(cls, value: Any, field: str) -> "FrozenJSON":
        try:
            encoded = canonical_json_bytes(value)
        except ContractError as exc:
            raise ManifestValidationError(f"{field}: {exc}") from exc
        return cls(encoded)

    def to_value(self) -> Any:
        return strict_json_loads(self.canonical)

    def validate(self, field: str) -> None:
        try:
            value = strict_json_loads(self.canonical)
            if canonical_json_bytes(value) != self.canonical:
                raise ManifestValidationError(f"{field} is not stored as canonical JSON")
        except ContractError as exc:
            if isinstance(exc, ManifestValidationError):
                raise
            raise ManifestValidationError(f"{field}: {exc}") from exc


@dataclass(frozen=True)
class RepositoryState:
    commit: str
    dirty: bool
    patch_sha256: Optional[str]
    source_archive_sha256: Optional[str]

    @classmethod
    def from_dict(cls, raw: Any) -> "RepositoryState":
        data = _expect_object(raw, "repository")
        _strict_fields(
            data,
            field="repository",
            required=("commit", "dirty", "patch_sha256", "source_archive_sha256"),
        )
        commit = data["commit"]
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise ManifestValidationError("repository.commit must be a full lowercase 40-character Git SHA")
        dirty = data["dirty"]
        if not isinstance(dirty, bool):
            raise ManifestValidationError("repository.dirty must be boolean")
        patch = _sha256(data["patch_sha256"], "repository.patch_sha256", optional=True)
        archive = _sha256(
            data["source_archive_sha256"],
            "repository.source_archive_sha256",
            optional=True,
        )
        if dirty and patch is None:
            raise ManifestValidationError("a dirty repository requires repository.patch_sha256")
        if not dirty and patch is not None:
            raise ManifestValidationError("a clean repository must not declare repository.patch_sha256")
        return cls(commit=commit, dirty=dirty, patch_sha256=patch, source_archive_sha256=archive)

    def validate(self) -> None:
        if not _COMMIT_RE.fullmatch(self.commit):
            raise ManifestValidationError("repository.commit must be a full lowercase 40-character Git SHA")
        if not isinstance(self.dirty, bool):
            raise ManifestValidationError("repository.dirty must be boolean")
        _sha256(self.patch_sha256, "repository.patch_sha256", optional=True)
        _sha256(self.source_archive_sha256, "repository.source_archive_sha256", optional=True)
        if self.dirty != (self.patch_sha256 is not None):
            raise ManifestValidationError(
                "repository.patch_sha256 must be present exactly when the repository is dirty"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "patch_sha256": self.patch_sha256,
            "source_archive_sha256": self.source_archive_sha256,
        }


@dataclass(frozen=True)
class InputArtifact:
    id: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_dict(cls, raw: Any) -> "InputArtifact":
        data = _expect_object(raw, "input")
        _strict_fields(data, field="input", required=("id", "sha256", "size_bytes"))
        return cls(
            id=safe_slug(data["id"], field="input.id"),
            sha256=_sha256(data["sha256"], "input.sha256") or "",
            size_bytes=_integer(data["size_bytes"], "input.size_bytes", maximum=2**63 - 1),
        )

    def validate(self) -> None:
        safe_slug(self.id, field="input.id")
        _sha256(self.sha256, "input.sha256")
        _integer(self.size_bytes, "input.size_bytes", maximum=2**63 - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class Configuration:
    id: str
    environment: FrozenJSON
    parameters: FrozenJSON
    expected_runtime: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "Configuration":
        data = _expect_object(raw, "configuration")
        _strict_fields(
            data,
            field="configuration",
            required=("id",),
            optional=("environment", "parameters", "expected_runtime"),
        )
        environment = _expect_object(data.get("environment", {}), "configuration.environment")
        for key, value in environment.items():
            if not _ENV_KEY_RE.fullmatch(key):
                raise ManifestValidationError(f"configuration.environment has unsafe key {key!r}")
            if not isinstance(value, str):
                raise ManifestValidationError(f"configuration.environment.{key} must be a string")
        parameters = _expect_object(data.get("parameters", {}), "configuration.parameters")
        runtime = _expect_object(data.get("expected_runtime", {}), "configuration.expected_runtime")
        return cls(
            id=safe_slug(data["id"], field="configuration.id"),
            environment=FrozenJSON.from_value(environment, "configuration.environment"),
            parameters=FrozenJSON.from_value(parameters, "configuration.parameters"),
            expected_runtime=FrozenJSON.from_value(runtime, "configuration.expected_runtime"),
        )

    def validate(self) -> None:
        safe_slug(self.id, field="configuration.id")
        self.environment.validate("configuration.environment")
        environment = _expect_object(self.environment.to_value(), "configuration.environment")
        for key, value in environment.items():
            if not _ENV_KEY_RE.fullmatch(key) or not isinstance(value, str):
                raise ManifestValidationError(f"configuration.environment has invalid entry {key!r}")
        self.parameters.validate("configuration.parameters")
        _expect_object(self.parameters.to_value(), "configuration.parameters")
        self.expected_runtime.validate("configuration.expected_runtime")
        _expect_object(self.expected_runtime.to_value(), "configuration.expected_runtime")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "environment": self.environment.to_value(),
            "parameters": self.parameters.to_value(),
            "expected_runtime": self.expected_runtime.to_value(),
        }


@dataclass(frozen=True)
class Workload:
    id: str
    producer_schema: str
    measurement_schema: str
    parameters: FrozenJSON
    depends_on: Tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "Workload":
        data = _expect_object(raw, "workload")
        _strict_fields(
            data,
            field="workload",
            required=("id", "producer_schema", "measurement_schema"),
            optional=("parameters", "depends_on"),
        )
        raw_dependencies = data.get("depends_on", [])
        if not isinstance(raw_dependencies, list):
            raise ManifestValidationError("workload.depends_on must be an array")
        dependencies = tuple(sorted(safe_slug(item, field="workload.depends_on[]") for item in raw_dependencies))
        if len(set(dependencies)) != len(dependencies):
            raise ManifestValidationError("workload.depends_on contains duplicates")
        parameters = _expect_object(data.get("parameters", {}), "workload.parameters")
        return cls(
            id=safe_slug(data["id"], field="workload.id"),
            producer_schema=_schema_id(data["producer_schema"], "workload.producer_schema"),
            measurement_schema=_schema_id(data["measurement_schema"], "workload.measurement_schema"),
            parameters=FrozenJSON.from_value(parameters, "workload.parameters"),
            depends_on=dependencies,
        )

    def validate(self) -> None:
        safe_slug(self.id, field="workload.id")
        _schema_id(self.producer_schema, "workload.producer_schema")
        _schema_id(self.measurement_schema, "workload.measurement_schema")
        self.parameters.validate("workload.parameters")
        _expect_object(self.parameters.to_value(), "workload.parameters")
        if tuple(sorted(self.depends_on)) != self.depends_on or len(set(self.depends_on)) != len(self.depends_on):
            raise ManifestValidationError("workload.depends_on must be sorted and unique")
        for dependency in self.depends_on:
            safe_slug(dependency, field="workload.depends_on[]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "producer_schema": self.producer_schema,
            "measurement_schema": self.measurement_schema,
            "parameters": self.parameters.to_value(),
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class ExpectedSiteConstraints:
    """Pre-submission constraints only; observed scheduler state belongs to attempts."""

    site_id: str
    scheduler: str
    partition: str
    nodes: int
    exclusive: bool
    node_constraints: Tuple[str, ...]
    account: Optional[str]
    resources: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "ExpectedSiteConstraints":
        data = _expect_object(raw, "expected_site")
        _strict_fields(
            data,
            field="expected_site",
            required=("site_id", "scheduler", "partition", "nodes", "exclusive"),
            optional=("node_constraints", "account", "resources"),
        )
        exclusive = data["exclusive"]
        if not isinstance(exclusive, bool):
            raise ManifestValidationError("expected_site.exclusive must be boolean")
        raw_constraints = data.get("node_constraints", [])
        if not isinstance(raw_constraints, list):
            raise ManifestValidationError("expected_site.node_constraints must be an array")
        constraints = tuple(
            sorted(_external_token(item, "expected_site.node_constraints[]") for item in raw_constraints)
        )
        if len(set(constraints)) != len(constraints):
            raise ManifestValidationError("expected_site.node_constraints contains duplicates")
        account_raw = data.get("account")
        account = None if account_raw is None else _external_token(account_raw, "expected_site.account")
        resources = _expect_object(data.get("resources", {}), "expected_site.resources")
        return cls(
            site_id=safe_slug(data["site_id"], field="expected_site.site_id"),
            scheduler=safe_slug(data["scheduler"], field="expected_site.scheduler"),
            partition=_external_token(data["partition"], "expected_site.partition"),
            nodes=_integer(data["nodes"], "expected_site.nodes", minimum=1, maximum=1024),
            exclusive=exclusive,
            node_constraints=constraints,
            account=account,
            resources=FrozenJSON.from_value(resources, "expected_site.resources"),
        )

    def validate(self) -> None:
        safe_slug(self.site_id, field="expected_site.site_id")
        safe_slug(self.scheduler, field="expected_site.scheduler")
        _external_token(self.partition, "expected_site.partition")
        _integer(self.nodes, "expected_site.nodes", minimum=1, maximum=1024)
        if not isinstance(self.exclusive, bool):
            raise ManifestValidationError("expected_site.exclusive must be boolean")
        if tuple(sorted(self.node_constraints)) != self.node_constraints or len(set(self.node_constraints)) != len(
            self.node_constraints
        ):
            raise ManifestValidationError("expected_site.node_constraints must be sorted and unique")
        for constraint in self.node_constraints:
            _external_token(constraint, "expected_site.node_constraints[]")
        if self.account is not None:
            _external_token(self.account, "expected_site.account")
        self.resources.validate("expected_site.resources")
        _expect_object(self.resources.to_value(), "expected_site.resources")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "scheduler": self.scheduler,
            "partition": self.partition,
            "nodes": self.nodes,
            "exclusive": self.exclusive,
            "node_constraints": list(self.node_constraints),
            "account": self.account,
            "resources": self.resources.to_value(),
        }


def _ensure_unique(values: Sequence[Any], field: str) -> None:
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for item in values:
        if item.id in seen:
            duplicates.add(item.id)
        seen.add(item.id)
    if duplicates:
        raise ManifestValidationError(f"{field} contains duplicate ids: {', '.join(sorted(duplicates))}")


def _validate_workload_graph(workloads: Sequence[Workload]) -> None:
    known = {workload.id for workload in workloads}
    for workload in workloads:
        unknown = sorted(set(workload.depends_on) - known)
        if unknown:
            raise ManifestValidationError(
                f"workload {workload.id!r} depends on unknown workloads: {', '.join(unknown)}"
            )
        if workload.id in workload.depends_on:
            raise ManifestValidationError(f"workload {workload.id!r} cannot depend on itself")

    visiting: Set[str] = set()
    visited: Set[str] = set()
    dependencies = {workload.id: workload.depends_on for workload in workloads}

    def visit(workload_id: str) -> None:
        if workload_id in visited:
            return
        if workload_id in visiting:
            raise ManifestValidationError("workload dependency graph contains a cycle")
        visiting.add(workload_id)
        for dependency in dependencies[workload_id]:
            visit(dependency)
        visiting.remove(workload_id)
        visited.add(workload_id)

    for workload_id in sorted(known):
        visit(workload_id)


@dataclass(frozen=True)
class CampaignSpec:
    schema: str
    run_id: str
    campaign_id: str
    repository: RepositoryState
    inputs: Tuple[InputArtifact, ...]
    configurations: Tuple[Configuration, ...]
    workloads: Tuple[Workload, ...]
    repetitions: int
    policy: FrozenJSON
    expected_site: ExpectedSiteConstraints

    @classmethod
    def from_dict(cls, raw: Any) -> "CampaignSpec":
        data = _expect_object(raw, "campaign")
        _strict_fields(
            data,
            field="campaign",
            required=(
                "schema",
                "run_id",
                "campaign_id",
                "repository",
                "inputs",
                "axes",
                "policy",
                "expected_site",
            ),
        )
        if data["schema"] != CAMPAIGN_SCHEMA:
            raise ManifestValidationError(f"unsupported campaign schema {data['schema']!r}")
        raw_inputs = data["inputs"]
        if not isinstance(raw_inputs, list):
            raise ManifestValidationError("campaign.inputs must be an array")
        axes = _expect_object(data["axes"], "campaign.axes")
        _strict_fields(
            axes,
            field="campaign.axes",
            required=("configurations", "workloads", "repetitions"),
        )
        raw_configurations = axes["configurations"]
        raw_workloads = axes["workloads"]
        if not isinstance(raw_configurations, list) or not raw_configurations:
            raise ManifestValidationError("campaign.axes.configurations must be a non-empty array")
        if not isinstance(raw_workloads, list) or not raw_workloads:
            raise ManifestValidationError("campaign.axes.workloads must be a non-empty array")
        repetitions = _integer(
            axes["repetitions"],
            "campaign.axes.repetitions",
            minimum=1,
            maximum=MAX_CAMPAIGN_REPETITIONS,
        )
        campaign_matrix_size(len(raw_configurations), len(raw_workloads), repetitions)
        policy = _expect_object(data["policy"], "campaign.policy")
        result = cls(
            schema=CAMPAIGN_SCHEMA,
            run_id=safe_slug(data["run_id"], field="campaign.run_id", max_length=64),
            campaign_id=safe_slug(data["campaign_id"], field="campaign.campaign_id"),
            repository=RepositoryState.from_dict(data["repository"]),
            inputs=tuple(sorted((InputArtifact.from_dict(item) for item in raw_inputs), key=lambda item: item.id)),
            configurations=tuple(
                sorted((Configuration.from_dict(item) for item in raw_configurations), key=lambda item: item.id)
            ),
            workloads=tuple(sorted((Workload.from_dict(item) for item in raw_workloads), key=lambda item: item.id)),
            repetitions=repetitions,
            policy=FrozenJSON.from_value(policy, "campaign.policy"),
            expected_site=ExpectedSiteConstraints.from_dict(data["expected_site"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema != CAMPAIGN_SCHEMA:
            raise ManifestValidationError(f"unsupported campaign schema {self.schema!r}")
        safe_slug(self.run_id, field="campaign.run_id", max_length=64)
        safe_slug(self.campaign_id, field="campaign.campaign_id")
        self.repository.validate()
        if not self.configurations:
            raise ManifestValidationError("campaign must contain at least one configuration")
        if not self.workloads:
            raise ManifestValidationError("campaign must contain at least one workload")
        _integer(
            self.repetitions,
            "campaign.axes.repetitions",
            minimum=1,
            maximum=MAX_CAMPAIGN_REPETITIONS,
        )
        campaign_matrix_size(len(self.configurations), len(self.workloads), self.repetitions)
        for input_artifact in self.inputs:
            input_artifact.validate()
        for configuration in self.configurations:
            configuration.validate()
        for workload in self.workloads:
            workload.validate()
        self.policy.validate("campaign.policy")
        _expect_object(self.policy.to_value(), "campaign.policy")
        self.expected_site.validate()
        _ensure_unique(self.inputs, "campaign.inputs")
        _ensure_unique(self.configurations, "campaign.axes.configurations")
        _ensure_unique(self.workloads, "campaign.axes.workloads")
        _validate_workload_graph(self.workloads)
        if tuple(sorted(item.id for item in self.inputs)) != tuple(item.id for item in self.inputs):
            raise ManifestValidationError("campaign.inputs must be sorted by id")
        if tuple(sorted(item.id for item in self.configurations)) != tuple(item.id for item in self.configurations):
            raise ManifestValidationError("campaign configurations must be sorted by id")
        if tuple(sorted(item.id for item in self.workloads)) != tuple(item.id for item in self.workloads):
            raise ManifestValidationError("campaign workloads must be sorted by id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "repository": self.repository.to_dict(),
            "inputs": [item.to_dict() for item in self.inputs],
            "axes": {
                "configurations": [item.to_dict() for item in self.configurations],
                "workloads": [item.to_dict() for item in self.workloads],
                "repetitions": self.repetitions,
            },
            "policy": self.policy.to_value(),
            "expected_site": self.expected_site.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _abbreviate(value: str, length: int = 18) -> str:
    shortened = value[:length].rstrip(".-")
    return shortened or "x"


def cell_identity_payload(
    *,
    run_id: str,
    campaign_sha256: str,
    configuration_id: str,
    workload_id: str,
    repetition: int,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "campaign_sha256": campaign_sha256,
        "configuration_id": configuration_id,
        "workload_id": workload_id,
        "repetition": repetition,
    }


def derive_cell_identity(
    *,
    run_id: str,
    campaign_sha256: str,
    configuration_id: str,
    workload_id: str,
    repetition: int,
) -> Tuple[str, str]:
    payload = cell_identity_payload(
        run_id=run_id,
        campaign_sha256=campaign_sha256,
        configuration_id=configuration_id,
        workload_id=workload_id,
        repetition=repetition,
    )
    identity_sha256 = canonical_sha256(payload)
    cell_id = f"c-{_abbreviate(workload_id)}-{_abbreviate(configuration_id)}-r{repetition:06d}-{identity_sha256[:16]}"
    safe_slug(cell_id, field="cell.id", max_length=CELL_ID_MAX_LENGTH)
    return cell_id, identity_sha256


@dataclass(frozen=True)
class CellSpec:
    id: str
    identity_sha256: str
    configuration_id: str
    workload_id: str
    repetition: int
    dependencies: Tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "CellSpec":
        data = _expect_object(raw, "cell")
        _strict_fields(
            data,
            field="cell",
            required=(
                "id",
                "identity_sha256",
                "configuration_id",
                "workload_id",
                "repetition",
                "dependencies",
            ),
        )
        raw_dependencies = data["dependencies"]
        if not isinstance(raw_dependencies, list):
            raise ManifestValidationError("cell.dependencies must be an array")
        dependencies = tuple(
            sorted(
                safe_slug(item, field="cell.dependencies[]", max_length=CELL_ID_MAX_LENGTH) for item in raw_dependencies
            )
        )
        if len(set(dependencies)) != len(dependencies):
            raise ManifestValidationError("cell.dependencies contains duplicates")
        return cls(
            id=safe_slug(data["id"], field="cell.id", max_length=CELL_ID_MAX_LENGTH),
            identity_sha256=_sha256(data["identity_sha256"], "cell.identity_sha256") or "",
            configuration_id=safe_slug(data["configuration_id"], field="cell.configuration_id"),
            workload_id=safe_slug(data["workload_id"], field="cell.workload_id"),
            repetition=_integer(data["repetition"], "cell.repetition"),
            dependencies=dependencies,
        )

    def validate(self) -> None:
        safe_slug(self.id, field="cell.id", max_length=CELL_ID_MAX_LENGTH)
        _sha256(self.identity_sha256, "cell.identity_sha256")
        safe_slug(self.configuration_id, field="cell.configuration_id")
        safe_slug(self.workload_id, field="cell.workload_id")
        _integer(self.repetition, "cell.repetition")
        if tuple(sorted(self.dependencies)) != self.dependencies or len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise ManifestValidationError("cell.dependencies must be sorted and unique")
        for dependency in self.dependencies:
            safe_slug(dependency, field="cell.dependencies[]", max_length=CELL_ID_MAX_LENGTH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "identity_sha256": self.identity_sha256,
            "configuration_id": self.configuration_id,
            "workload_id": self.workload_id,
            "repetition": self.repetition,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class RunManifest:
    schema: str
    campaign_sha256: str
    campaign: CampaignSpec
    cells: Tuple[CellSpec, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "RunManifest":
        data = _expect_object(raw, "manifest")
        _strict_fields(
            data,
            field="manifest",
            required=("schema", "campaign_sha256", "campaign", "cells"),
        )
        if data["schema"] != MANIFEST_SCHEMA:
            raise ManifestValidationError(f"unsupported manifest schema {data['schema']!r}")
        raw_cells = data["cells"]
        if not isinstance(raw_cells, list):
            raise ManifestValidationError("manifest.cells must be an array")
        campaign = CampaignSpec.from_dict(data["campaign"])
        campaign_matrix_size(
            len(campaign.configurations),
            len(campaign.workloads),
            campaign.repetitions,
        )
        if len(raw_cells) > MAX_CAMPAIGN_MATRIX_CELLS:
            raise ManifestValidationError(
                f"manifest cells exceed MAX_CAMPAIGN_MATRIX_CELLS={MAX_CAMPAIGN_MATRIX_CELLS}"
            )
        result = cls(
            schema=MANIFEST_SCHEMA,
            campaign_sha256=_sha256(data["campaign_sha256"], "manifest.campaign_sha256") or "",
            campaign=campaign,
            cells=tuple(CellSpec.from_dict(item) for item in raw_cells),
        )
        result.validate()
        return result

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "RunManifest":
        return cls.from_dict(strict_json_loads(data))

    @property
    def run_id(self) -> str:
        return self.campaign.run_id

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ManifestValidationError(f"unsupported manifest schema {self.schema!r}")
        _sha256(self.campaign_sha256, "manifest.campaign_sha256")
        self.campaign.validate()
        if len(self.cells) > MAX_CAMPAIGN_MATRIX_CELLS:
            raise ManifestValidationError(
                f"manifest cells exceed MAX_CAMPAIGN_MATRIX_CELLS={MAX_CAMPAIGN_MATRIX_CELLS}"
            )
        if self.campaign_sha256 != self.campaign.sha256:
            raise ManifestValidationError("manifest campaign hash does not match its declarative campaign")

        for cell in self.cells:
            cell.validate()
        ids = [cell.id for cell in self.cells]
        if len(set(ids)) != len(ids):
            duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
            raise ManifestValidationError(f"manifest contains duplicate cell ids: {', '.join(duplicates)}")
        semantic_keys = [(cell.configuration_id, cell.workload_id, cell.repetition) for cell in self.cells]
        if len(set(semantic_keys)) != len(semantic_keys):
            raise ManifestValidationError("manifest contains duplicate semantic cells")
        if tuple(sorted(ids)) != tuple(ids):
            raise ManifestValidationError("manifest cells must be sorted by id")
        # Independently re-expand the declarative axes. This intentionally does
        # not trust the serialized cell list as its own completeness oracle.
        expected: Dict[Tuple[str, str, int], Tuple[str, str, Tuple[str, ...]]] = {}
        workloads = {workload.id: workload for workload in self.campaign.workloads}
        for repetition in range(self.campaign.repetitions):
            for configuration in self.campaign.configurations:
                for workload in self.campaign.workloads:
                    cell_id, identity_sha256 = derive_cell_identity(
                        run_id=self.run_id,
                        campaign_sha256=self.campaign_sha256,
                        configuration_id=configuration.id,
                        workload_id=workload.id,
                        repetition=repetition,
                    )
                    dependencies = []
                    for dependency_id in workloads[workload.id].depends_on:
                        dependency_cell_id, _ = derive_cell_identity(
                            run_id=self.run_id,
                            campaign_sha256=self.campaign_sha256,
                            configuration_id=configuration.id,
                            workload_id=dependency_id,
                            repetition=repetition,
                        )
                        dependencies.append(dependency_cell_id)
                    expected[(configuration.id, workload.id, repetition)] = (
                        cell_id,
                        identity_sha256,
                        tuple(sorted(dependencies)),
                    )

        actual = {(cell.configuration_id, cell.workload_id, cell.repetition): cell for cell in self.cells}
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        if missing or unknown:
            details: List[str] = []
            if missing:
                details.append(f"missing cells={missing!r}")
            if unknown:
                details.append(f"unknown cells={unknown!r}")
            raise ManifestValidationError("manifest matrix mismatch: " + "; ".join(details))

        for key, (expected_id, expected_identity, expected_dependencies) in expected.items():
            cell = actual[key]
            if cell.id != expected_id:
                raise ManifestValidationError(f"cell {key!r} has stale or forged id")
            if cell.identity_sha256 != expected_identity:
                raise ManifestValidationError(f"cell {cell.id!r} has stale or forged identity hash")
            if cell.dependencies != expected_dependencies:
                raise ManifestValidationError(f"cell {cell.id!r} has incorrect dependencies")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_sha256": self.campaign_sha256,
            "campaign": self.campaign.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_json_bytes())
