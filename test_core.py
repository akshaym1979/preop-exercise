#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic>=2.8.0",
#   "pytest>=8.0.0",
# ]
# ///

"""Unit + regression tests for core.py.

Run with: `uv run test_core.py` (uses inline script metadata) or `pytest test_core.py`.

Test surface (per design §10):
  - Helpers & predicates (to_utc_date, is_hp_type, lab/vital selection)
  - Normalizer: tri-state active, canonical H&P selection, consent classification
  - Rule engine: each rule + missing-required-field + decision derivation
  - LLM-fallback paths (the four classify_* wrappers) — exercised under mocks
    even though they don't fire on the 50-record sample
  - Regression: the 5 records the baseline got wrong (00007, 00025, 00034, 00040, 00048)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import core
from core import (
    BloodPressureVital,
    ConsentSignedResponse,
    ConsentStatus,
    Document,
    DocTypeResponse,
    DrugClassificationResponse,
    LabResult,
    Medication,
    NormalizedState,
    PatientSubmission,
    PlanAdequacyResponse,
    ProcedureInfo,
    Sourced,
    TemperatureVital,
    classify_consent,
    classify_medications,
    derive_decision,
    evaluate_plan_adequacy,
    evaluate_rules,
    is_consent_type,
    is_hp_type,
    is_plan_type,
    missing_required_fields,
    normalize,
    normalize_lab_code,
    rule_1_documentation,
    rule_2_testing,
    rule_3_anticoagulation,
    rule_4_acute_safety,
    select_canonical_hp,
    select_most_recent_lab,
    select_most_recent_vital,
    sort_issues,
    to_utc_date,
    triage_submission,
)


# ---------------------------------------------------------------------------
# Fixtures: stub the LLM and cache so tests don't hit the network
# ---------------------------------------------------------------------------


class _StubLLM:
    """Configurable stub for `_openai_call`. Set responses by schema before each test."""

    def __init__(self) -> None:
        self.responses: dict[type, Any] = {}
        self.calls: list[tuple[type, str]] = []  # (schema, user_input)

    def set(self, schema: type, response: Any) -> None:
        self.responses[schema] = response

    def __call__(self, *, schema: type, prompt: str, user_input: str, model: str) -> Any:
        self.calls.append((schema, user_input))
        if schema not in self.responses:
            raise AssertionError(f"No stubbed response for {schema.__name__}")
        return self.responses[schema]


@pytest.fixture
def llm() -> _StubLLM:
    return _StubLLM()


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a fresh in-memory cache; nothing touches disk."""
    cache: dict[str, Any] = {}
    monkeypatch.setattr(core, "_cache_state", None)
    monkeypatch.setattr(core, "_load_cache", lambda: cache)
    monkeypatch.setattr(core, "_persist_cache", lambda c: None)


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch, llm: _StubLLM) -> _StubLLM:
    monkeypatch.setattr(core, "_openai_call", llm)
    return llm


# ---------------------------------------------------------------------------
# to_utc_date
# ---------------------------------------------------------------------------


class TestToUtcDate:
    def test_bare_iso_date(self) -> None:
        assert to_utc_date("2026-03-01") == date(2026, 3, 1)

    def test_utc_z_suffix(self) -> None:
        assert to_utc_date("2026-02-21T08:10:00Z") == date(2026, 2, 21)

    def test_negative_offset_crosses_midnight(self) -> None:
        # 23:30 EST = 04:30 UTC next day -> date increments
        assert to_utc_date("2026-02-20T23:30:00-05:00") == date(2026, 2, 21)

    def test_naive_datetime_treated_as_utc(self) -> None:
        assert to_utc_date("2026-02-21T08:10:00") == date(2026, 2, 21)


# ---------------------------------------------------------------------------
# Type predicates and lab code normalization
# ---------------------------------------------------------------------------


class TestPredicates:
    @pytest.mark.parametrize("doc_type", [
        "History and Physical",
        "Pre-op H&P (signed)",
        "Scanned H+P (H&P) - signed",
        "Imported: Hx & Physical (H&P)",
        "History/Physical (H&P)",
        "Preop Hist & Phys (H&P) [PDF]",
        "PREOP - H and P - signed",
    ])
    def test_hp_type_matches(self, doc_type: str) -> None:
        assert is_hp_type(doc_type), f"{doc_type!r} should match H&P regex"

    @pytest.mark.parametrize("doc_type", [
        "Pre-op Nursing Intake",
        "Anesthesia Pre-Assessment",
        "Clinic Follow-up Note",
        "Medical Clearance [PDF]",
        "Surgical Consent",
        "History & Phsyical",  # typo: "Phsyical" — intentionally not matched
    ])
    def test_hp_type_rejects(self, doc_type: str) -> None:
        assert not is_hp_type(doc_type), f"{doc_type!r} should NOT match H&P regex"

    @pytest.mark.parametrize("doc_type", [
        "Perioperative Medication Plan",
        "Perioperative Medication Review",
        "Anticoag Plan (scanned)",
        "Cardiology Progress Note - Anticoag",
    ])
    def test_plan_type_matches(self, doc_type: str) -> None:
        assert is_plan_type(doc_type)

    @pytest.mark.parametrize("doc_type", [
        "Surgical Consent",
        "Consent for Surgery",
        "Consent Counseling Note",  # treated as valid surgical consent per oracle
        "Procedure Consent Form",
        "Pre-op Consent Discussion",
    ])
    def test_consent_type_matches(self, doc_type: str) -> None:
        assert is_consent_type(doc_type)

    def test_lab_code_strips_lab_prefix(self) -> None:
        assert normalize_lab_code("LAB-CBC") == "CBC"
        assert normalize_lab_code("CBC") == "CBC"
        assert normalize_lab_code("LAB-CMP") == "CMP"
        assert normalize_lab_code("HBA1C") == "HBA1C"


# ---------------------------------------------------------------------------
# select_canonical_hp
# ---------------------------------------------------------------------------


def _doc(idx_marker: str, type_: str, date_: str | None, text: str = "") -> Document:
    return Document(doc_id=idx_marker, type=type_, date=date_, author="t", text=text)


class TestCanonicalHpSelection:
    def test_single_hp_in_window(self) -> None:
        docs = [
            _doc("a", "Pre-op Nursing Intake", "2026-02-15"),
            _doc("b", "History and Physical", "2026-02-20"),
        ]
        picked = select_canonical_hp(docs, date(2026, 3, 1))
        assert picked is not None
        assert picked.source_path == "documents[1]"

    def test_multi_hp_picks_most_recent(self) -> None:
        docs = [
            _doc("a", "H&P Note", "2026-02-20", text="current"),
            _doc("b", "Imported: H&P", "2026-01-30", text="Prior pre-op H&P retained for longitudinal chart context."),
        ]
        picked = select_canonical_hp(docs, date(2026, 3, 1))
        assert picked is not None
        assert picked.source_path == "documents[0]"  # 2026-02-20 is newer

    def test_ties_broken_by_lower_index(self) -> None:
        docs = [
            _doc("a", "Pre-op Nursing Intake", "2026-02-01"),
            _doc("b", "H&P Note", "2026-02-20"),
            _doc("c", "Pre-op H/P (signed)", "2026-02-20"),
        ]
        picked = select_canonical_hp(docs, date(2026, 3, 1))
        assert picked is not None
        assert picked.source_path == "documents[1]"  # lower index wins tie

    def test_docs_after_procedure_date_excluded(self) -> None:
        docs = [
            _doc("a", "H&P Note", "2026-03-05"),  # after procedure
            _doc("b", "Pre-op H&P", "2026-02-20"),
        ]
        picked = select_canonical_hp(docs, date(2026, 3, 1))
        assert picked is not None
        assert picked.source_path == "documents[1]"

    def test_no_procedure_date_picks_most_recent(self) -> None:
        # When procedure_date is unknown, no date filter applied; pick globally most recent.
        docs = [
            _doc("a", "Pre-op H&P", "2026-01-10"),
            _doc("b", "H&P Note", "2026-04-01"),
        ]
        picked = select_canonical_hp(docs, None)
        assert picked is not None
        assert picked.source_path == "documents[1]"

    def test_retained_doc_used_when_only_hp(self) -> None:
        # Case_00002 regression: retained-marked doc is the ONLY H&P-typed doc;
        # design intentionally does NOT hard-filter retained text. The oracle treats
        # this doc as the canonical H&P and flags it out-of-window.
        docs = [
            _doc(
                "a",
                "Scanned History and Physical Examination",
                "2026-01-30",
                text="Prior pre-op H&P retained for longitudinal chart context.",
            ),
            _doc("b", "Pre-op Nursing Intake", "2026-02-15"),
        ]
        picked = select_canonical_hp(docs, date(2026, 3, 3))
        assert picked is not None
        assert picked.source_path == "documents[0]"

    def test_no_match_returns_none(self) -> None:
        docs = [_doc("a", "Pre-op Nursing Intake", "2026-02-20")]
        assert select_canonical_hp(docs, date(2026, 3, 1)) is None


# ---------------------------------------------------------------------------
# select_most_recent_lab / select_most_recent_vital
# ---------------------------------------------------------------------------


class TestLabSelection:
    def test_picks_most_recent_after_lab_prefix_normalization(self) -> None:
        labs = [
            LabResult(id="1", code="LAB-CBC", effective_at="2026-02-15T08:00:00Z"),
            LabResult(id="2", code="CBC", effective_at="2026-02-21T08:00:00Z"),
            LabResult(id="3", code="HBA1C", effective_at="2026-02-25T08:00:00Z"),
        ]
        picked = select_most_recent_lab(labs, "CBC")
        assert picked is not None
        assert picked.value.id == "2"
        assert picked.source_path == "labs[1]"

    def test_no_lab_of_canonical_code(self) -> None:
        labs = [LabResult(id="1", code="HBA1C", effective_at="2026-02-25T08:00:00Z")]
        assert select_most_recent_lab(labs, "CBC") is None
        assert select_most_recent_lab(labs, "CMP") is None


class TestVitalSelection:
    def test_filters_by_type_field_not_union_member(self) -> None:
        # Both members of the union have a `type` field; we filter on the field, not
        # which member Pydantic chose during parsing.
        vitals = [
            BloodPressureVital(type="blood_pressure", systolic=130, diastolic=85, date="2026-02-15T08:00:00Z"),
            TemperatureVital(type="temperature", value_f=98.6, date="2026-02-21T08:00:00Z"),
            BloodPressureVital(type="blood_pressure", systolic=140, diastolic=88, date="2026-02-22T08:00:00Z"),
        ]
        bp = select_most_recent_vital(vitals, "blood_pressure")
        temp = select_most_recent_vital(vitals, "temperature")
        assert bp is not None and bp.source_path == "vitals[2]"
        assert temp is not None and temp.source_path == "vitals[1]"


# ---------------------------------------------------------------------------
# classify_medications (LLM fallback path + tri-state active)
# ---------------------------------------------------------------------------


class TestClassifyMedications:
    def test_allowlist_fast_path_active(self, stub_openai: _StubLLM) -> None:
        # apixaban is in the allowlist; no LLM call needed.
        meds = [Medication(name="apixaban", active=True)]
        active, unknown = classify_medications(meds, model="gpt-4.1-mini")
        assert len(active) == 1 and active[0].source_path == "medications[0]"
        assert unknown == ()
        assert stub_openai.calls == []  # no LLM call

    def test_allowlist_unknown_active(self, stub_openai: _StubLLM) -> None:
        # warfarin in allowlist + active=null → unknown_status_medications.
        meds = [Medication(name="warfarin", active=None)]
        active, unknown = classify_medications(meds, model="gpt-4.1-mini")
        assert active == ()
        assert len(unknown) == 1
        assert stub_openai.calls == []  # allowlist matched; no LLM

    def test_llm_fallback_non_anticoag_high_confidence(self, stub_openai: _StubLLM) -> None:
        # lisinopril not in allowlist → LLM is invoked; high-confidence False → drop.
        stub_openai.set(DrugClassificationResponse, DrugClassificationResponse(is_anticoagulant=False, confidence="high"))
        meds = [Medication(name="lisinopril", active=True)]
        active, unknown = classify_medications(meds, model="gpt-4.1-mini")
        assert active == () and unknown == ()
        assert len(stub_openai.calls) == 1

    def test_llm_fallback_low_confidence_routes_to_safer_branch(self, stub_openai: _StubLLM) -> None:
        # Unknown drug + LLM returns low confidence → treat as anticoag (safer).
        stub_openai.set(
            DrugClassificationResponse,
            DrugClassificationResponse(is_anticoagulant=False, confidence="low"),
        )
        meds = [Medication(name="some-novel-drug", active=True)]
        active, unknown = classify_medications(meds, model="gpt-4.1-mini")
        assert len(active) == 1  # flagged for review despite is_anticoagulant=False
        assert unknown == ()

    def test_inactive_medication_dropped(self, stub_openai: _StubLLM) -> None:
        # active=False medications are not in scope for Rule 3 (no LLM call).
        meds = [Medication(name="apixaban", active=False)]
        active, unknown = classify_medications(meds, model="gpt-4.1-mini")
        assert active == () and unknown == ()


# ---------------------------------------------------------------------------
# classify_consent (keyword scan + LLM fallback for ambiguous text)
# ---------------------------------------------------------------------------


class TestClassifyConsent:
    def test_signed_keyword_match(self, stub_openai: _StubLLM) -> None:
        docs = [_doc("c", "Surgical Consent", "2026-02-25", "Electronic consent obtained and signed by patient.")]
        consent = classify_consent(docs, model="gpt-4.1-mini")
        assert consent.document is not None
        assert consent.signed is True
        assert stub_openai.calls == []  # no LLM

    def test_unsigned_keyword_match(self, stub_openai: _StubLLM) -> None:
        docs = [_doc("c", "Surgical Consent", "2026-02-25", "Consent documented but unsigned; awaiting patient signature.")]
        consent = classify_consent(docs, model="gpt-4.1-mini")
        assert consent.signed is False
        assert stub_openai.calls == []

    def test_llm_fallback_on_ambiguous_text(self, stub_openai: _StubLLM) -> None:
        # Text matches neither keyword set → LLM is invoked.
        stub_openai.set(
            ConsentSignedResponse,
            ConsentSignedResponse(signed=True, confidence="high"),
        )
        docs = [_doc("c", "Surgical Consent", "2026-02-25", "Consent obtained per protocol.")]
        consent = classify_consent(docs, model="gpt-4.1-mini")
        assert consent.signed is True
        assert len(stub_openai.calls) == 1

    def test_llm_low_confidence_routes_to_unsigned(self, stub_openai: _StubLLM) -> None:
        # Ambiguous text + low-confidence LLM → safer branch (unsigned).
        stub_openai.set(
            ConsentSignedResponse,
            ConsentSignedResponse(signed=True, confidence="low"),
        )
        docs = [_doc("c", "Surgical Consent", "2026-02-25", "Consent obtained per protocol.")]
        consent = classify_consent(docs, model="gpt-4.1-mini")
        assert consent.signed is False

    def test_no_consent_doc(self, stub_openai: _StubLLM) -> None:
        docs = [_doc("hp", "H&P Note", "2026-02-25", "...")]
        consent = classify_consent(docs, model="gpt-4.1-mini")
        assert consent.document is None
        assert consent.signed is False


# ---------------------------------------------------------------------------
# Plan-adequacy evaluation
# ---------------------------------------------------------------------------


class TestPlanAdequacy:
    def test_picks_most_recent_plan_doc(self, stub_openai: _StubLLM) -> None:
        # Multiple plan-typed docs; the most-recent one's text should reach the LLM.
        stub_openai.set(PlanAdequacyResponse, PlanAdequacyResponse(adequate=False, reason="stub"))
        plan_docs = (
            Sourced(_doc("a", "Perioperative Medication Plan", "2026-02-10", "old plan"), "documents[0]"),
            Sourced(_doc("b", "Perioperative Medication Plan", "2026-02-25", "new plan"), "documents[1]"),
        )
        evaluate_plan_adequacy(plan_docs, model="gpt-4.1-mini")
        assert stub_openai.calls == [(PlanAdequacyResponse, "new plan")]


# ---------------------------------------------------------------------------
# Rule 1 — documentation (date-window edge case at exactly 30 days)
# ---------------------------------------------------------------------------


def _state(**kwargs: Any) -> NormalizedState:
    """Build a NormalizedState with sensible defaults; override fields via kwargs."""
    defaults: dict[str, Any] = dict(
        procedure_date=Sourced(date(2026, 3, 1), "procedure.procedure_date"),
        procedure_risk=Sourced("LOW", "procedure.procedure_risk"),
        canonical_hp=None,
        consent=ConsentStatus(document=None, signed=False),
        most_recent_cbc=None,
        most_recent_cmp=None,
        active_anticoagulants=(),
        unknown_status_medications=(),
        plan_documents=(),
        plan_adequacy=None,
        most_recent_bp=Sourced(BloodPressureVital(type="blood_pressure", systolic=120, diastolic=70, date="2026-02-20T08:00:00Z"), "vitals[0]"),
        most_recent_temp=Sourced(TemperatureVital(type="temperature", value_f=98.6, date="2026-02-20T08:00:00Z"), "vitals[1]"),
        all_doc_types=("Pre-op Nursing Intake",),
        all_lab_codes=("HBA1C",),
    )
    defaults.update(kwargs)
    return NormalizedState(**defaults)


class TestRule1:
    def test_hp_exactly_30_days_passes(self) -> None:
        hp = _doc("a", "H&P Note", "2026-01-30")
        state = _state(canonical_hp=Sourced(hp, "documents[0]"),
                       consent=ConsentStatus(document=Sourced(_doc("c", "Surgical Consent", "2026-02-25", "signed"), "documents[1]"), signed=True))
        issues = rule_1_documentation(state)
        # 30 days exactly: 2026-03-01 - 2026-01-30 = 30 days; must be within 30 → passes.
        assert [i for i in issues if i.category == "REQUIRED_DOCUMENTATION"] == []

    def test_hp_31_days_fails(self) -> None:
        hp = _doc("a", "H&P Note", "2026-01-29")
        state = _state(canonical_hp=Sourced(hp, "documents[0]"),
                       consent=ConsentStatus(document=Sourced(_doc("c", "Surgical Consent", "2026-02-25", "signed"), "documents[1]"), signed=True))
        issues = rule_1_documentation(state)
        assert any(i.category == "REQUIRED_DOCUMENTATION" and "outside" in i.description.lower() for i in issues)

    def test_missing_procedure_date_blocks_hp_check(self) -> None:
        # No cascade: rule 1 short-circuits when procedure_date is None.
        state = _state(procedure_date=None, canonical_hp=None)
        issues = rule_1_documentation(state)
        # Consent check still runs; H&P check does not.
        assert not any("History and Physical" in i.description for i in issues)


# ---------------------------------------------------------------------------
# Rule 2 — testing (date-window edge case at exactly 14 days for HIGH)
# ---------------------------------------------------------------------------


class TestRule2:
    def test_cbc_exactly_14_days_for_high_passes(self) -> None:
        cbc = LabResult(id="1", code="CBC", effective_at="2026-02-15T00:00:00Z")
        state = _state(
            procedure_risk=Sourced("HIGH", "procedure.procedure_risk"),
            most_recent_cbc=Sourced(cbc, "labs[0]"),
            most_recent_cmp=Sourced(LabResult(id="2", code="CMP", effective_at="2026-02-15T00:00:00Z"), "labs[1]"),
        )
        # 2026-03-01 - 2026-02-15 = 14 days; must be within 14 → passes.
        issues = rule_2_testing(state)
        assert [i for i in issues if i.category == "REQUIRED_TESTING"] == []

    def test_cbc_15_days_for_high_fails(self) -> None:
        cbc = LabResult(id="1", code="CBC", effective_at="2026-02-14T00:00:00Z")
        state = _state(
            procedure_risk=Sourced("HIGH", "procedure.procedure_risk"),
            most_recent_cbc=Sourced(cbc, "labs[0]"),
            most_recent_cmp=Sourced(LabResult(id="2", code="CMP", effective_at="2026-02-15T00:00:00Z"), "labs[1]"),
        )
        issues = rule_2_testing(state)
        assert any(i.category == "REQUIRED_TESTING" and "CBC" in i.description for i in issues)

    def test_cmp_required_only_for_high(self) -> None:
        state = _state(
            procedure_risk=Sourced("MODERATE", "procedure.procedure_risk"),
            most_recent_cbc=Sourced(LabResult(id="1", code="CBC", effective_at="2026-02-21T00:00:00Z"), "labs[0]"),
            most_recent_cmp=None,
        )
        issues = rule_2_testing(state)
        assert not any("CMP" in i.description for i in issues)

    def test_short_circuits_on_missing_procedure_date(self) -> None:
        state = _state(procedure_date=None, most_recent_cbc=None)
        assert rule_2_testing(state) == []

    def test_short_circuits_on_missing_procedure_risk(self) -> None:
        state = _state(procedure_risk=None, most_recent_cbc=None)
        assert rule_2_testing(state) == []


# ---------------------------------------------------------------------------
# Rule 3 — anticoagulation
# ---------------------------------------------------------------------------


class TestRule3:
    def test_no_anticoag_no_issue(self) -> None:
        assert rule_3_anticoagulation(_state()) == []

    def test_active_anticoag_no_plan(self) -> None:
        med = Medication(name="apixaban", active=True)
        state = _state(active_anticoagulants=(Sourced(med, "medications[1]"),))
        issues = rule_3_anticoagulation(state)
        assert len(issues) == 1
        assert issues[0].category == "ANTICOAGULATION_MANAGEMENT"
        assert "apixaban" in issues[0].evidence.details  # grounding anchor

    def test_active_anticoag_with_adequate_plan_no_issue(self) -> None:
        med = Medication(name="apixaban", active=True)
        plan = _doc("p", "Perioperative Medication Plan", "2026-02-20", "Hold apixaban 48h pre-op; resume 24h post-op.")
        state = _state(
            active_anticoagulants=(Sourced(med, "medications[1]"),),
            plan_documents=(Sourced(plan, "documents[2]"),),
            plan_adequacy=PlanAdequacyResponse(adequate=True, reason="clear hold/resume"),
        )
        assert rule_3_anticoagulation(state) == []

    def test_active_anticoag_with_inadequate_plan_emits(self) -> None:
        med = Medication(name="apixaban", active=True)
        plan = _doc("p", "Perioperative Medication Plan", "2026-02-20", "follow up with cardiology")
        state = _state(
            active_anticoagulants=(Sourced(med, "medications[1]"),),
            plan_documents=(Sourced(plan, "documents[2]"),),
            plan_adequacy=PlanAdequacyResponse(adequate=False, reason="defers"),
        )
        assert len(rule_3_anticoagulation(state)) == 1


# ---------------------------------------------------------------------------
# Rule 4 — acute safety
# ---------------------------------------------------------------------------


class TestRule4:
    def test_systolic_180_exact_fires(self) -> None:
        bp = BloodPressureVital(type="blood_pressure", systolic=180, diastolic=100, date="2026-02-20T08:00:00Z", source="clinic")
        state = _state(most_recent_bp=Sourced(bp, "vitals[1]"))
        assert any(i.category == "ACUTE_SAFETY_EXCLUSION" for i in rule_4_acute_safety(state))

    def test_systolic_179_passes(self) -> None:
        bp = BloodPressureVital(type="blood_pressure", systolic=179, diastolic=109, date="2026-02-20T08:00:00Z", source="clinic")
        state = _state(most_recent_bp=Sourced(bp, "vitals[1]"))
        assert rule_4_acute_safety(state) == []

    def test_diastolic_110_exact_fires(self) -> None:
        bp = BloodPressureVital(type="blood_pressure", systolic=170, diastolic=110, date="2026-02-20T08:00:00Z", source="clinic")
        state = _state(most_recent_bp=Sourced(bp, "vitals[1]"))
        assert any(i.category == "ACUTE_SAFETY_EXCLUSION" for i in rule_4_acute_safety(state))

    def test_temp_100_4_exact_passes(self) -> None:
        # threshold is `> 100.4` (strict greater-than)
        temp = TemperatureVital(type="temperature", value_f=100.4, date="2026-02-20T08:00:00Z", source="clinic")
        state = _state(most_recent_temp=Sourced(temp, "vitals[1]"))
        assert rule_4_acute_safety(state) == []

    def test_temp_100_5_fires(self) -> None:
        temp = TemperatureVital(type="temperature", value_f=100.5, date="2026-02-20T08:00:00Z", source="clinic")
        state = _state(most_recent_temp=Sourced(temp, "vitals[1]"))
        assert any(i.category == "ACUTE_SAFETY_EXCLUSION" for i in rule_4_acute_safety(state))


# ---------------------------------------------------------------------------
# Missing-required-field cross-cutting emitter
# ---------------------------------------------------------------------------


class TestMissingRequiredFields:
    def test_null_procedure_date(self) -> None:
        issues = missing_required_fields(_state(procedure_date=None))
        assert any(i.evidence.source == "procedure.procedure_date" for i in issues)

    def test_null_procedure_risk(self) -> None:
        issues = missing_required_fields(_state(procedure_risk=None))
        assert any(i.evidence.source == "procedure.procedure_risk" for i in issues)

    def test_missing_bp_vital(self) -> None:
        issues = missing_required_fields(_state(most_recent_bp=None))
        assert any("blood_pressure" in i.evidence.details for i in issues)

    def test_missing_temp_vital(self) -> None:
        issues = missing_required_fields(_state(most_recent_temp=None))
        assert any("temperature" in i.evidence.details for i in issues)

    def test_unknown_status_anticoag(self) -> None:
        med = Medication(name="warfarin", active=None)
        state = _state(unknown_status_medications=(Sourced(med, "medications[1]"),))
        issues = missing_required_fields(state)
        anticoag_missing = [i for i in issues if i.evidence.source == "medications[1]"]
        assert len(anticoag_missing) == 1
        assert "warfarin" in anticoag_missing[0].evidence.details
        assert "active=null" in anticoag_missing[0].evidence.details


# ---------------------------------------------------------------------------
# Decision derivation + ordering
# ---------------------------------------------------------------------------


class TestDecision:
    def test_empty_issues_ready(self) -> None:
        assert derive_decision([]) == "READY"

    def test_acute_safety_dominates(self) -> None:
        from core import TriageIssue, TriageIssueEvidence
        issues = [
            TriageIssue(category="REQUIRED_DOCUMENTATION", description="x", evidence=TriageIssueEvidence(source="documents", details="y")),
            TriageIssue(category="ACUTE_SAFETY_EXCLUSION", description="x", evidence=TriageIssueEvidence(source="vitals[1]", details="y")),
        ]
        assert derive_decision(issues) == "NOT_CLEARED"

    def test_any_other_issue_needs_followup(self) -> None:
        from core import TriageIssue, TriageIssueEvidence
        issues = [TriageIssue(category="REQUIRED_TESTING", description="x", evidence=TriageIssueEvidence(source="labs", details="y"))]
        assert derive_decision(issues) == "NEEDS_FOLLOW_UP"

    def test_sort_by_rule_number(self) -> None:
        from core import TriageIssue, TriageIssueEvidence
        issues = [
            TriageIssue(category="MISSING_REQUIRED_DATA", description="m", evidence=TriageIssueEvidence(source="procedure.procedure_date", details="d")),
            TriageIssue(category="REQUIRED_DOCUMENTATION", description="r", evidence=TriageIssueEvidence(source="documents", details="d")),
        ]
        sorted_ = sort_issues(issues)
        assert sorted_[0].category == "REQUIRED_DOCUMENTATION"
        assert sorted_[1].category == "MISSING_REQUIRED_DATA"


# ---------------------------------------------------------------------------
# End-to-end regression: the 5 records the baseline got wrong
# ---------------------------------------------------------------------------


_DATASET = Path(__file__).resolve().parent / "data" / "patients_sample_50.jsonl"


def _load_case(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for line in _DATASET.read_text().splitlines():
        row = json.loads(line)
        if row["case_id"] == case_id:
            return row["submission"], row["expected_output"]
    raise KeyError(case_id)


@pytest.mark.parametrize("case_id,expected_decision,expected_categories", [
    ("case_00007", "NEEDS_FOLLOW_UP", {"REQUIRED_DOCUMENTATION"}),  # H&P 45 days out
    ("case_00025", "NEEDS_FOLLOW_UP", {"REQUIRED_DOCUMENTATION"}),  # No consent doc
    ("case_00034", "NEEDS_FOLLOW_UP", {"MISSING_REQUIRED_DATA"}),   # procedure_risk null
    ("case_00040", "NEEDS_FOLLOW_UP", {"REQUIRED_DOCUMENTATION"}),  # H&P 45 days out
    ("case_00048", "NEEDS_FOLLOW_UP", {"REQUIRED_DOCUMENTATION"}),  # H&P 45 days out
])
def test_baseline_misses_now_pass(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    expected_decision: str,
    expected_categories: set[str],
) -> None:
    """The 5 records the baseline (LLM-only) got wrong should all match the oracle now."""
    sub, oracle = _load_case(case_id)

    # Stub the LLM for the few records that fire it (e.g., 00007/00040/00048 don't need it,
    # but other records can call classify_drug for lisinopril). Provide a generic stub.
    def fake_openai(*, schema, prompt, user_input, model):
        if schema is DrugClassificationResponse:
            return DrugClassificationResponse(is_anticoagulant=False, confidence="high")
        if schema is PlanAdequacyResponse:
            return PlanAdequacyResponse(adequate=False, reason="stub")
        if schema is ConsentSignedResponse:
            return ConsentSignedResponse(signed=False, confidence="high")
        if schema is DocTypeResponse:
            return DocTypeResponse(role="other", confidence="high")
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr(core, "_openai_call", fake_openai)

    out = triage_submission(sub, model="gpt-4.1-mini")
    assert out.decision == expected_decision, f"{case_id}: oracle={expected_decision} got={out.decision}"
    actual_categories = {i.category for i in out.issues}
    assert actual_categories == expected_categories, f"{case_id}: oracle={expected_categories} got={actual_categories}"
    assert out.decision == oracle["decision"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
