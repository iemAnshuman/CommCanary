"""Strict loader for the declarative Rostam site/config/workload catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

from ..harness import (
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    ContractError,
    read_bounded_bytes,
    strict_json_loads,
)
from ..harness.model import FrozenJSON

PathLike = Union[str, "Path"]
CATALOG_SCHEMA = "commcanary.rostam.catalog.v2"

_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SCHEMA_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_PLACEHOLDER_RE = re.compile(
    r"^\{(?:venv_python|venv_bin|experiment_dir|workspace|"
    r"dependency:[a-z0-9.-]+:[a-z0-9_-]+|input:[a-z0-9.-]+)\}(?:/[^\x00]*)?$"
)


class CatalogValidationError(ContractError):
    """Raised when a catalog is ambiguous or unsafe to bind into a manifest."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{field} must be an object")
    return value


def _fields(
    value: Mapping[str, Any],
    field: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise CatalogValidationError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise CatalogValidationError(f"{field} has unknown fields: {', '.join(unknown)}")


def _text(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise CatalogValidationError(f"{field} must be a non-empty NUL-free string of at most {maximum} characters")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _text(value, field, maximum=128)
    if not _ID_RE.fullmatch(text):
        raise CatalogValidationError(f"{field} is not a safe identifier")
    return text


def _schema(value: Any, field: str) -> str:
    text = _text(value, field, maximum=160)
    if not _SCHEMA_RE.fullmatch(text):
        raise CatalogValidationError(f"{field} is not a schema identifier")
    return text


def _positive_int(value: Any, field: str, *, maximum: int = 86_400) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CatalogValidationError(f"{field} must be an integer in [1, {maximum}]")
    return cast(int, value)


def _unique_ids(values: Sequence[Any], field: str) -> None:
    ids = [item.id for item in values]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise CatalogValidationError(f"{field} contains duplicate ids: {', '.join(duplicates)}")


def _relative_path(value: Any, field: str) -> str:
    text = _text(value, field, maximum=512)
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or any(part in {"", ".", ".."} for part in path.parts):
        raise CatalogValidationError(f"{field} must be a normalized relative POSIX path")
    return text


def _argv(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise CatalogValidationError(f"{field} must contain 1..256 string arguments")
    result = []
    for index, raw in enumerate(value):
        item = _text(raw, f"{field}[{index}]", maximum=4096)
        if "{" in item or "}" in item:
            # A placeholder may be the whole path prefix, followed by a fixed
            # relative suffix.  Embedded shell syntax and partial expansion
            # are intentionally impossible.
            if not _PLACEHOLDER_RE.fullmatch(item):
                raise CatalogValidationError(f"{field}[{index}] has an unsupported placeholder")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class Site:
    site_id: str
    scheduler: str
    partition: str
    nodes: int
    exclusive: bool
    node_constraints: Tuple[str, ...]
    account: Optional[str]
    python_module: str
    resources: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "Site":
        data = _object(raw, "catalog.site")
        _fields(
            data,
            "catalog.site",
            required=(
                "site_id",
                "scheduler",
                "partition",
                "nodes",
                "exclusive",
                "node_constraints",
                "account",
                "python_module",
                "resources",
            ),
        )
        if data["scheduler"] != "slurm" or data["site_id"] != "rostam":
            raise CatalogValidationError("this catalog must target the rostam/slurm site contract")
        exclusive = data["exclusive"]
        if not isinstance(exclusive, bool):
            raise CatalogValidationError("catalog.site.exclusive must be boolean")
        constraints_raw = data["node_constraints"]
        if not isinstance(constraints_raw, list) or not constraints_raw:
            raise CatalogValidationError("catalog.site.node_constraints must be a non-empty array")
        constraints = tuple(
            sorted(_text(item, "catalog.site.node_constraints[]", maximum=128) for item in constraints_raw)
        )
        if len(set(constraints)) != len(constraints):
            raise CatalogValidationError("catalog.site.node_constraints contains duplicates")
        account_raw = data["account"]
        account = None if account_raw is None else _text(account_raw, "catalog.site.account", maximum=128)
        resources = _object(data["resources"], "catalog.site.resources")
        return cls(
            site_id="rostam",
            scheduler="slurm",
            partition=_text(data["partition"], "catalog.site.partition", maximum=128),
            nodes=_positive_int(data["nodes"], "catalog.site.nodes", maximum=1024),
            exclusive=exclusive,
            node_constraints=constraints,
            account=account,
            python_module=_text(data["python_module"], "catalog.site.python_module", maximum=128),
            resources=FrozenJSON.from_value(resources, "catalog.site.resources"),
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        resources = self.resources.to_value()
        resources["python_module"] = self.python_module
        return {
            "site_id": self.site_id,
            "scheduler": self.scheduler,
            "partition": self.partition,
            "nodes": self.nodes,
            "exclusive": self.exclusive,
            "node_constraints": list(self.node_constraints),
            "account": self.account,
            "resources": resources,
        }


@dataclass(frozen=True)
class CatalogConfiguration:
    id: str
    venv: str
    environment: FrozenJSON
    expected_runtime: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "CatalogConfiguration":
        data = _object(raw, "catalog.configuration")
        _fields(data, "catalog.configuration", required=("id", "venv", "environment", "expected_runtime"))
        environment = _object(data["environment"], "catalog.configuration.environment")
        for key, value in environment.items():
            if not isinstance(key, str) or not _ENV_RE.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
                raise CatalogValidationError(f"catalog.configuration.environment has invalid entry {key!r}")
        expected_runtime = _object(data["expected_runtime"], "catalog.configuration.expected_runtime")
        _fields(
            expected_runtime,
            "catalog.configuration.expected_runtime",
            required=("nccl_version", "python_version", "runtime_nccl_version_code", "torch_version"),
        )
        _positive_int(
            expected_runtime["runtime_nccl_version_code"],
            "catalog.configuration.expected_runtime.runtime_nccl_version_code",
            maximum=99_999,
        )
        return cls(
            id=_identifier(data["id"], "catalog.configuration.id"),
            venv=_relative_path(data["venv"], "catalog.configuration.venv"),
            environment=FrozenJSON.from_value(environment, "catalog.configuration.environment"),
            expected_runtime=FrozenJSON.from_value(expected_runtime, "catalog.configuration.expected_runtime"),
        )


@dataclass(frozen=True)
class CatalogWorkload:
    id: str
    wrapper: str
    producer_schema: str
    measurement_schema: str
    timeout_seconds: int
    depends_on: Tuple[str, ...]
    parameters: FrozenJSON

    @classmethod
    def from_dict(cls, raw: Any) -> "CatalogWorkload":
        data = _object(raw, "catalog.workload")
        _fields(
            data,
            "catalog.workload",
            required=(
                "id",
                "wrapper",
                "producer_schema",
                "measurement_schema",
                "timeout_seconds",
                "depends_on",
                "parameters",
            ),
        )
        dependency_raw = data["depends_on"]
        if not isinstance(dependency_raw, list):
            raise CatalogValidationError("catalog.workload.depends_on must be an array")
        dependencies = tuple(sorted(_identifier(item, "catalog.workload.depends_on[]") for item in dependency_raw))
        if len(set(dependencies)) != len(dependencies):
            raise CatalogValidationError("catalog.workload.depends_on contains duplicates")
        parameters = _object(data["parameters"], "catalog.workload.parameters")
        _validate_physical_parameters(parameters, data["id"])
        return cls(
            id=_identifier(data["id"], "catalog.workload.id"),
            wrapper=_identifier(data["wrapper"], "catalog.workload.wrapper"),
            producer_schema=_schema(data["producer_schema"], "catalog.workload.producer_schema"),
            measurement_schema=_schema(data["measurement_schema"], "catalog.workload.measurement_schema"),
            timeout_seconds=_positive_int(data["timeout_seconds"], "catalog.workload.timeout_seconds"),
            depends_on=dependencies,
            parameters=FrozenJSON.from_value(parameters, "catalog.workload.parameters"),
        )


def _validate_physical_parameters(parameters: Mapping[str, Any], workload_id: Any) -> None:
    required = {"adapter", "operation", "world_size", "global_ranks"}
    missing = sorted(required - set(parameters))
    if missing:
        raise CatalogValidationError(f"workload {workload_id!r} parameters are missing: {', '.join(missing)}")
    adapter = parameters["adapter"]
    if adapter not in {"torch-json", "param-text", "capture"}:
        raise CatalogValidationError(f"workload {workload_id!r} has unsupported adapter {adapter!r}")
    if parameters["operation"] != "all_reduce":
        raise CatalogValidationError("the current physical contract supports only all_reduce")
    world_size = _positive_int(parameters["world_size"], "workload.parameters.world_size", maximum=1024)
    ranks = parameters["global_ranks"]
    if ranks != list(range(world_size)):
        raise CatalogValidationError("the current physical contract supports only a dense world process group")
    if adapter != "capture":
        _argv(parameters.get("command"), "workload.parameters.command")
    if "profile_command" in parameters:
        _argv(parameters["profile_command"], "workload.parameters.profile_command")
    if "transform_commands" in parameters:
        commands = parameters["transform_commands"]
        if not isinstance(commands, list) or not commands:
            raise CatalogValidationError("workload.parameters.transform_commands must be non-empty")
        for index, command in enumerate(commands):
            _argv(command, f"workload.parameters.transform_commands[{index}]")


@dataclass(frozen=True)
class CampaignProfile:
    id: str
    configuration_ids: Tuple[str, ...]
    workload_ids: Tuple[str, ...]
    required_input_ids: Tuple[str, ...]

    @classmethod
    def from_pair(cls, profile_id: Any, raw: Any) -> "CampaignProfile":
        identifier = _identifier(profile_id, "catalog.profiles key")
        data = _object(raw, f"catalog.profiles.{identifier}")
        _fields(
            data,
            f"catalog.profiles.{identifier}",
            required=("configuration_ids", "workload_ids", "required_input_ids"),
        )
        parsed = []
        for field in ("configuration_ids", "workload_ids", "required_input_ids"):
            values = data[field]
            if not isinstance(values, list) or not values:
                raise CatalogValidationError(f"catalog.profiles.{identifier}.{field} must be non-empty")
            items = tuple(_identifier(item, f"catalog.profiles.{identifier}.{field}[]") for item in values)
            if len(set(items)) != len(items):
                raise CatalogValidationError(f"catalog.profiles.{identifier}.{field} contains duplicates")
            parsed.append(items)
        return cls(identifier, parsed[0], parsed[1], parsed[2])


@dataclass(frozen=True)
class Catalog:
    schema: str
    site: Site
    configurations: Tuple[CatalogConfiguration, ...]
    workloads: Tuple[CatalogWorkload, ...]
    profiles: Tuple[CampaignProfile, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "Catalog":
        data = _object(raw, "catalog")
        _fields(data, "catalog", required=("schema", "site", "configurations", "workloads", "profiles"))
        if data["schema"] != CATALOG_SCHEMA:
            raise CatalogValidationError(f"unsupported catalog schema {data['schema']!r}")
        configurations_raw = data["configurations"]
        workloads_raw = data["workloads"]
        profiles_raw = _object(data["profiles"], "catalog.profiles")
        if not isinstance(configurations_raw, list) or not configurations_raw:
            raise CatalogValidationError("catalog.configurations must be a non-empty array")
        if not isinstance(workloads_raw, list) or not workloads_raw:
            raise CatalogValidationError("catalog.workloads must be a non-empty array")
        configurations = tuple(CatalogConfiguration.from_dict(item) for item in configurations_raw)
        workloads = tuple(CatalogWorkload.from_dict(item) for item in workloads_raw)
        profiles = tuple(CampaignProfile.from_pair(key, value) for key, value in sorted(profiles_raw.items()))
        _unique_ids(configurations, "catalog.configurations")
        _unique_ids(workloads, "catalog.workloads")
        _unique_ids(profiles, "catalog.profiles")
        result = cls(CATALOG_SCHEMA, Site.from_dict(data["site"]), configurations, workloads, profiles)
        result.validate_graph()
        return result

    def validate_graph(self) -> None:
        configuration_ids = {item.id for item in self.configurations}
        workload_ids = {item.id for item in self.workloads}
        dependencies = {item.id: set(item.depends_on) for item in self.workloads}
        for workload in self.workloads:
            unknown = sorted(set(workload.depends_on) - workload_ids)
            if unknown or workload.id in workload.depends_on:
                raise CatalogValidationError(f"workload {workload.id!r} has invalid dependencies: {unknown!r}")
        visiting = set()
        visited = set()

        def visit(workload_id: str) -> None:
            if workload_id in visited:
                return
            if workload_id in visiting:
                raise CatalogValidationError("catalog workload graph contains a cycle")
            visiting.add(workload_id)
            for dependency in dependencies[workload_id]:
                visit(dependency)
            visiting.remove(workload_id)
            visited.add(workload_id)

        for workload_id in sorted(workload_ids):
            visit(workload_id)
        for profile in self.profiles:
            unknown_configurations = sorted(set(profile.configuration_ids) - configuration_ids)
            unknown_workloads = sorted(set(profile.workload_ids) - workload_ids)
            if unknown_configurations or unknown_workloads:
                raise CatalogValidationError(
                    f"profile {profile.id!r} references unknown values: "
                    f"configurations={unknown_configurations!r}, workloads={unknown_workloads!r}"
                )
            selected = set(profile.workload_ids)
            for workload_id in profile.workload_ids:
                missing_dependencies = sorted(dependencies[workload_id] - selected)
                if missing_dependencies:
                    raise CatalogValidationError(
                        f"profile {profile.id!r} omits dependencies for {workload_id!r}: {missing_dependencies!r}"
                    )

    def profile(self, profile_id: str) -> CampaignProfile:
        identifier = _identifier(profile_id, "profile_id")
        try:
            return next(item for item in self.profiles if item.id == identifier)
        except StopIteration as exc:
            raise CatalogValidationError(f"unknown catalog profile {identifier!r}") from exc

    def selected_configurations(self, profile: CampaignProfile) -> Tuple[CatalogConfiguration, ...]:
        wanted = set(profile.configuration_ids)
        return tuple(sorted((item for item in self.configurations if item.id in wanted), key=lambda item: item.id))

    def selected_workloads(self, profile: CampaignProfile) -> Tuple[CatalogWorkload, ...]:
        wanted = set(profile.workload_ids)
        return tuple(sorted((item for item in self.workloads if item.id in wanted), key=lambda item: item.id))


def load_catalog(path: PathLike) -> Catalog:
    source = Path(path)
    try:
        encoded = read_bounded_bytes(
            source,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field="Rostam catalog",
        )
        raw = strict_json_loads(encoded)
    except CanonicalJSONError as exc:
        raise CatalogValidationError(f"cannot load Rostam catalog: {exc}") from exc
    catalog = Catalog.from_dict(raw)
    return catalog
