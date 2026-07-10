import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evals" / "evaluation-dataset.json"

ALLOWED_PRIORITIES = {"P0", "P1", "P2"}

REQUIRED_CASE_FIELDS = {
    "id",
    "title",
    "category",
    "priority",
    "user_role",
    "expected_behavior",
    "required_facts",
    "forbidden_claims",
    "expected_source_ids",
    "pass_criteria",
}


def load_dataset() -> dict[str, Any]:
    """Загрузить синтетический AI evaluation dataset."""
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def get_test_cases() -> list[dict[str, Any]]:
    """Получить список evaluation-сценариев."""
    dataset = load_dataset()

    assert "test_cases" in dataset, "Dataset must contain test_cases"
    assert isinstance(dataset["test_cases"], list), (
        "test_cases must be a list"
    )
    assert dataset["test_cases"], (
        "Evaluation dataset must not be empty"
    )

    return dataset["test_cases"]


def test_dataset_contains_required_metadata() -> None:
    dataset = load_dataset()

    required_metadata = {
        "project",
        "release",
        "dataset_version",
        "description",
        "test_cases",
    }

    missing_fields = required_metadata - dataset.keys()

    assert not missing_fields, (
        "Dataset metadata is incomplete. "
        f"Missing fields: {sorted(missing_fields)}"
    )


def test_all_test_case_ids_are_unique() -> None:
    test_cases = get_test_cases()
    case_ids = [case["id"] for case in test_cases]

    duplicates = {
        case_id
        for case_id in case_ids
        if case_ids.count(case_id) > 1
    }

    assert not duplicates, (
        "Evaluation case IDs must be unique. "
        f"Duplicates: {sorted(duplicates)}"
    )


def test_all_cases_have_required_fields() -> None:
    for case in get_test_cases():
        missing_fields = REQUIRED_CASE_FIELDS - case.keys()

        assert not missing_fields, (
            f"{case.get('id', 'UNKNOWN_CASE')} is missing fields: "
            f"{sorted(missing_fields)}"
        )


def test_priorities_use_supported_values() -> None:
    for case in get_test_cases():
        assert case["priority"] in ALLOWED_PRIORITIES, (
            f"{case['id']} has unsupported priority: "
            f"{case['priority']}"
        )


def test_p0_cases_have_strict_pass_criteria() -> None:
    p0_cases = [
        case
        for case in get_test_cases()
        if case["priority"] == "P0"
    ]

    assert p0_cases, "Dataset must contain at least one P0 case"

    for case in p0_cases:
        assert case["required_facts"], (
            f"{case['id']} must define required facts"
        )

        assert case["forbidden_claims"], (
            f"{case['id']} must define forbidden claims"
        )

        assert len(case["pass_criteria"]) >= 3, (
            f"{case['id']} must contain at least "
            "three pass criteria"
        )


def test_access_control_does_not_expect_restricted_sources() -> None:
    access_control_cases = [
        case
        for case in get_test_cases()
        if case["category"] == "access_control"
    ]

    assert access_control_cases, (
        "Dataset must contain an access control scenario"
    )

    for case in access_control_cases:
        assert case["expected_source_ids"] == [], (
            f"{case['id']} must not expect restricted sources "
            "for an employee role"
        )


def test_correct_refusal_cases_do_not_expect_sources() -> None:
    refusal_cases = [
        case
        for case in get_test_cases()
        if case["category"] == "correct_refusal"
    ]

    assert refusal_cases, (
        "Dataset must contain a correct refusal scenario"
    )

    for case in refusal_cases:
        assert case["expected_source_ids"] == [], (
            f"{case['id']} should not expect sources when "
            "the knowledge base has no answer"
        )


def test_prompt_injection_cases_forbid_disclosures() -> None:
    injection_categories = {
        "prompt_injection",
        "indirect_prompt_injection",
    }

    injection_cases = [
        case
        for case in get_test_cases()
        if case["category"] in injection_categories
    ]

    assert injection_cases, (
        "Dataset must contain prompt injection scenarios"
    )

    for case in injection_cases:
        forbidden_text = " ".join(
            case["forbidden_claims"]
        ).lower()

        assert (
            "prompt" in forbidden_text
            or "инструк" in forbidden_text
            or "секрет" in forbidden_text
        ), (
            f"{case['id']} must explicitly forbid disclosure "
            "of system instructions or secrets"
        )


def test_document_versioning_uses_active_version() -> None:
    versioning_cases = [
        case
        for case in get_test_cases()
        if case["category"] == "document_versioning"
    ]

    assert versioning_cases, (
        "Dataset must contain a document versioning scenario"
    )

    for case in versioning_cases:
        active_versions = {
            source["version"]
            for source in case["available_sources"]
            if source.get("status") == "active"
        }

        expected_versions = set(
            case.get("expected_source_versions", [])
        )

        assert expected_versions == active_versions, (
            f"{case['id']} must expect only "
            "active document versions"
        )


def test_every_case_has_clear_description() -> None:
    for case in get_test_cases():
        assert len(case["title"].strip()) >= 10, (
            f"{case['id']} title is too short"
        )

        assert len(case["expected_behavior"].strip()) >= 20, (
            f"{case['id']} expected_behavior is too short"
        )
