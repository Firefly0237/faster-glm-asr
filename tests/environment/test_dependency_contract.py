from __future__ import annotations

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).parents[2]


def test_published_hugging_face_constraints_accept_every_selected_version() -> None:
    fixture = json.loads(
        (ROOT / "tests/environment/fixtures/hf-stack-metadata.json").read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == "pypi-interoperability-metadata-v1"
    assert all(
        source.startswith("https://pypi.org/pypi/") and source.endswith("/json")
        for source in fixture["sources"].values()
    )
    selected = {
        canonicalize_name(name): Version(version)
        for name, version in fixture["selected_versions"].items()
    }
    for requirements in fixture["interoperability_constraints"].values():
        for raw_requirement in requirements:
            requirement = Requirement(raw_requirement)
            assert selected[canonicalize_name(requirement.name)] in requirement.specifier


def test_contract_common_requirements_and_pyproject_share_the_compatible_pins() -> None:
    contract = json.loads(
        (ROOT / "configs/environments/rtx3090-ddp-v1.json").read_text(encoding="utf-8")
    )
    expected = {
        "huggingface-hub": "1.5.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
    }
    assert {name: contract["packages"][name] for name in expected} == expected

    common = (ROOT / "requirements/rtx3090-common.in").read_text(encoding="utf-8")
    for name, version in expected.items():
        assert f"{name}=={version}" in common

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    for name, version in expected.items():
        if name == "tokenizers":
            continue
        assert Version(version) in declared[name].specifier
