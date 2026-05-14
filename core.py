"""Pre-op triage system.

Hybrid architecture: a deterministic rule engine handles mechanical operations
(date math, threshold checks, schema enforcement); narrow LLM extractors handle
the work that requires reading free text (plan adequacy in the load-bearing
path; drug/doc-type/consent classification as gated fallbacks). LLM responses
are cached on disk by content hash for byte-stable replay.

See `notes/design.md` for the full design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

_logger = logging.getLogger(__name__)

# -------------------------
# Type aliases used by the harness and our rule engine
# -------------------------

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

class TriageIssueEvidence(BaseModel):

    source: str
    details: str

class TriageIssue(BaseModel):

    category: IssueCategory
    description: str
    evidence: TriageIssueEvidence

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str


class PreparedPatientCase(BaseModel):
    """Serialized eval case with submission payload and expected oracle output."""

    case_id: str
    submission: PatientSubmission
    expected_output: TriageOutput


def triage_output_json_schema() -> dict[str, object]:
    """Return the JSON schema used for structured model outputs."""

    schema = TriageOutput.model_json_schema()
    return schema


# -------------------------
# Date helper
# -------------------------


def to_utc_date(s: str) -> date:
    """Parse an ISO 8601 date or datetime string to a UTC `date`.

    Accepts:
        - Bare dates:        "2026-03-01"
        - Datetimes w/ Z:    "2026-02-21T08:10:00Z"
        - Datetimes w/ tz:   "2026-02-21T03:10:00-05:00"
        - Naive datetimes:   "2026-02-21T08:10:00" (assumed UTC)

    Used uniformly across rules so date math is timezone-correct on the day
    boundary; cf. design doc §2 + §6.
    """
    parsed: str = s.replace("Z", "+00:00") if s.endswith("Z") else s
    dt = datetime.fromisoformat(parsed)
    if dt.tzinfo is None:
        return dt.date()
    return dt.astimezone(timezone.utc).date()


# -------------------------
# LLM response schemas (sent to OpenAI with strict: true)
# -------------------------


class PlanAdequacyResponse(BaseModel):
    """Result of evaluating a perioperative anticoagulation plan."""

    model_config = ConfigDict(extra="forbid")

    adequate: bool
    reason: str  # short rationale; not emitted in the final output


class DrugClassificationResponse(BaseModel):
    """Whether a medication is an anticoagulant. Used as fallback when name doesn't match the allowlist."""

    model_config = ConfigDict(extra="forbid")

    is_anticoagulant: bool
    confidence: Literal["high", "medium", "low"]


class DocTypeResponse(BaseModel):
    """Classification of a clinical document type into a policy role.

    Used as fallback when the deterministic regex matchers don't recognize the type string.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["history_and_physical", "surgical_consent", "anticoag_plan", "other"]
    confidence: Literal["high", "medium", "low"]


class ConsentSignedResponse(BaseModel):
    """Whether a consent document's text indicates a signed consent."""

    model_config = ConfigDict(extra="forbid")

    signed: bool
    confidence: Literal["high", "medium", "low"]


# -------------------------
# Internal types (used by the rule engine; not part of the wire contract)
# -------------------------

T = TypeVar("T")


@dataclass(frozen=True)
class Sourced(Generic[T]):
    """A derived value paired with the citation `source_path` the rule engine will emit."""

    value: T
    source_path: str  # e.g. "documents[0]", "vitals[2]", "procedure.procedure_date"


@dataclass(frozen=True)
class ConsentStatus:
    """Result of consent classification: document presence plus signed/unsigned state."""

    document: Sourced[Document] | None
    signed: bool  # True if signed-keyword or LLM-classified-signed; False otherwise.


@dataclass(frozen=True)
class NormalizedState:
    """Derived facts the rule engine evaluates. Computed once by `normalize()`."""

    procedure_date: Sourced[date] | None
    procedure_risk: Sourced[ProcedureRisk] | None

    canonical_hp: Sourced[Document] | None
    consent: ConsentStatus

    most_recent_cbc: Sourced[LabResult] | None
    most_recent_cmp: Sourced[LabResult] | None

    active_anticoagulants: tuple[Sourced[Medication], ...]
    unknown_status_medications: tuple[Sourced[Medication], ...]  # anticoag-classified AND active=null
    plan_documents: tuple[Sourced[Document], ...]
    plan_adequacy: PlanAdequacyResponse | None

    most_recent_bp: Sourced[BloodPressureVital] | None
    most_recent_temp: Sourced[TemperatureVital] | None

    # Inventories carried for grounding-anchor strings (§6 grounding invariant).
    # Both are populated from the raw submission so rules don't need a second pass.
    all_doc_types: tuple[str, ...] = ()
    all_lab_codes: tuple[str, ...] = ()


# -------------------------
# Lookup tables (used by the normalizer)
# -------------------------

ANTICOAG_ALLOWLIST: frozenset[str] = frozenset({
    "apixaban", "warfarin", "rivaroxaban", "dabigatran",
    "edoxaban", "heparin", "enoxaparin",
})

# H&P type matcher. Each alternate captures a common synonym observed in the sample's 100+
# unique doc type strings ("History and Physical" appears verbatim only once).
_HP_TYPE_RE = re.compile(
    r"\b(?:"
    r"h\s*&\s*p"
    r"|h\s+and\s+p"
    r"|h\s*/\s*p"
    r"|h\s*\+\s*p"
    r"|history\s+(?:and|&|/)\s+physical"
    r"|hist\s+(?:&\s+)?phys"
    r"|hx\s+(?:&\s+)?physical"
    r"|history/physical"
    r")\b",
    re.IGNORECASE,
)

_PLAN_DOC_RE = re.compile(
    r"perioperative\s+medication\s+(?:plan|review)"
    r"|anticoag(?:ulation)?\s+plan"
    r"|cardiology\s+progress\s+note\s*-\s*anticoag",
    re.IGNORECASE,
)

UNSIGNED_CONSENT_KEYWORDS: tuple[str, ...] = (
    "unsigned",
    "awaiting signature",
    "signature not yet",
    "signature pending",
)

SIGNED_CONSENT_KEYWORDS: tuple[str, ...] = (
    "signed",
    "signature on file",
    "signed by",
    "electronic consent obtained",
)

LAB_CODE_PREFIX = "LAB-"
RECOGNIZED_LAB_CODES: frozenset[str] = frozenset({"CBC", "CMP"})


def normalize_lab_code(code: str) -> str:
    """Strip the source-system `LAB-` prefix and uppercase the result."""
    upper = code.upper()
    return upper[len(LAB_CODE_PREFIX):] if upper.startswith(LAB_CODE_PREFIX) else upper


# -------------------------
# Type predicates
# -------------------------


def is_hp_type(doc_type: str) -> bool:
    return _HP_TYPE_RE.search(doc_type) is not None


def is_plan_type(doc_type: str) -> bool:
    return _PLAN_DOC_RE.search(doc_type) is not None


def is_consent_type(doc_type: str) -> bool:
    return "consent" in doc_type.lower()


# -------------------------
# LLM cache and structured-output helper
# -------------------------

_CACHE_PATH = Path(__file__).resolve().parent / "data" / "llm_cache.json"
_cache_state: dict[str, Any] | None = None

TResponse = TypeVar("TResponse", bound=BaseModel)


def canonical_json(obj: object) -> str:
    """Stable JSON serialization for content hashing: sorted keys, no whitespace, ASCII-escaped."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_cache() -> dict[str, Any]:
    """Load the on-disk LLM cache into module state on first access.

    Returns a mutable dict. Mutations are persisted via `_persist_cache` on each miss.
    Corrupted cache files are reset to empty with a warning (per design §9).
    """
    global _cache_state
    if _cache_state is None:
        if _CACHE_PATH.exists():
            try:
                _cache_state = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                _logger.warning("LLM cache at %s unreadable (%s); starting empty.", _CACHE_PATH, exc)
                _cache_state = {}
        else:
            _cache_state = {}
    return _cache_state


def _persist_cache(cache: dict[str, Any]) -> None:
    """Atomically write cache to disk: temp file + rename."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".tmp")
    tmp.write_text(
        json.dumps(cache, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(_CACHE_PATH)


def _hash_key(schema_name: str, model: str, prompt: str, user_input: object) -> str:
    """SHA-256 of (schema_name, model, prompt, canonical-JSON input), joined by null bytes.

    Including the schema name guards against the (unlikely but real) case of two
    different schemas sharing a prompt/input - each schema gets its own cache slot.
    """
    parts = b"\0".join([
        schema_name.encode("utf-8"),
        model.encode("utf-8"),
        prompt.encode("utf-8"),
        canonical_json(user_input).encode("utf-8"),
    ])
    return hashlib.sha256(parts).hexdigest()


def _strict_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the OpenAI Responses `text.format` block for strict JSON schema mode."""
    return {
        "format": {
            "type": "json_schema",
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        }
    }


def _openai_call(
    *,
    schema: type[TResponse],
    prompt: str,
    user_input: str,
    model: str,
) -> TResponse:
    """One structured-output call to the OpenAI Responses API.

    Separated from `cached_call` so tests can monkeypatch this function with a stub.
    Retries once with a 1-second backoff per design §9; subsequent failure propagates.
    """
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "instructions": prompt,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            }
        ],
        "text": _strict_response_format(schema),
        "temperature": 0,
    }
    try:
        response = client.responses.create(**request)
    except Exception as exc:
        _logger.warning("OpenAI call failed (%s); retrying once after 1s.", exc)
        time.sleep(1.0)
        response = client.responses.create(**request)
    return schema.model_validate_json(response.output_text)


def cached_call(
    schema: type[TResponse],
    prompt: str,
    user_input: str,
    *,
    model: str,
) -> TResponse:
    """Look up the (model, prompt, input) tuple in the on-disk cache; call the LLM on miss.

    The cache file lives at `data/llm_cache.json` and is checked into the repo
    so cross-clone replay is byte-stable (design §8).
    """
    cache = _load_cache()
    key = _hash_key(schema.__name__, model, prompt, user_input)
    if key in cache:
        return schema.model_validate(cache[key]["response"])

    response = _openai_call(schema=schema, prompt=prompt, user_input=user_input, model=model)
    cache[key] = {
        "schema_name": schema.__name__,
        "model": model,
        "response": response.model_dump(),
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _persist_cache(cache)
    return response


# -------------------------
# LLM prompts (constants; included in cache hash)
# -------------------------

PLAN_ADEQUACY_PROMPT = """\
You are evaluating a perioperative anticoagulation plan for a surgical patient.

A plan is ADEQUATE only if its text clearly states both:
  (a) when to HOLD the anticoagulant pre-procedure, AND
  (b) when to RESUME it post-procedure,
with specific hold/resume guidance (dates, durations, or clinical milestones).

A plan is INADEQUATE if it is missing, defers to another clinician without specifics
(e.g., "follow up with cardiology"), is pending finalization, or lacks clear hold/resume
guidance.

Examples of INADEQUATE plan text:
  - "Follow up with cardiology for peri-op recommendations."
  - "Anticoagulant noted; perioperative management details not yet documented."
  - "Plan pending specialist input."

Example of ADEQUATE plan text:
  - "Hold apixaban 48 hours pre-op. Resume 24 hours post-op pending hemostasis."

Default to adequate=false when uncertain. The `reason` field should be a short
sentence (<=20 words) explaining your judgment.
"""

DRUG_CLASS_PROMPT = """\
You are classifying a medication name. Return is_anticoagulant=true if the medication
is an anticoagulant - a drug that prevents blood clot formation, used clinically to
manage thrombosis risk.

Examples of anticoagulants:
  warfarin, apixaban, rivaroxaban, dabigatran, edoxaban, heparin, enoxaparin, fondaparinux.

Examples of non-anticoagulants:
  - lisinopril (ACE inhibitor for blood pressure)
  - metformin (oral antidiabetic)
  - aspirin (antiplatelet; NOT an anticoagulant per the Cadence policy)
  - atorvastatin (statin for cholesterol)

If the name is unfamiliar or could plausibly be either, return confidence=low.
The caller treats confidence=low as a decline and routes to the safer (flag) branch.
"""

DOC_TYPE_PROMPT = """\
You are classifying a clinical document type string into one of four roles:
  - history_and_physical: a pre-operative History and Physical (H&P) document
  - surgical_consent: a signed surgical consent for the planned procedure
  - anticoag_plan: a perioperative anticoagulation management plan
  - other: anything else (nursing intake, anesthesia notes, follow-up notes, etc.)

Examples:
  - "History and Physical Examination" -> history_and_physical
  - "Pre-op H&P (signed)" -> history_and_physical
  - "Surgical Consent" -> surgical_consent
  - "Consent Counseling Note" -> surgical_consent
  - "Perioperative Medication Plan" -> anticoag_plan
  - "Pre-op Nursing Intake" -> other

Return confidence=low if the type string is ambiguous.
"""

CONSENT_SIGNED_PROMPT = """\
You are determining whether a consent document's text indicates the consent was signed
by the patient.

Examples of SIGNED text:
  - "Electronic consent obtained and signed by patient for procedure."
  - "Patient reviewed risks/benefits and signed surgical consent."
  - "Signed consent scanned and verified before scheduling."

Examples of UNSIGNED text:
  - "Consent documented but unsigned; awaiting patient signature."
  - "Unsigned consent on chart; provider requested signature before scheduling."
  - "Verbal consent only; signature pending."

If the text is genuinely ambiguous, return confidence=low. The caller treats
confidence=low as unsigned (safer to flag for follow-up).
"""


# -------------------------
# LLM classify_* wrappers
# -------------------------


def classify_plan_adequacy(plan_text: str, *, model: str) -> PlanAdequacyResponse:
    return cached_call(PlanAdequacyResponse, PLAN_ADEQUACY_PROMPT, plan_text, model=model)


def classify_drug(medication_name: str, *, model: str) -> DrugClassificationResponse:
    return cached_call(DrugClassificationResponse, DRUG_CLASS_PROMPT, medication_name, model=model)


def classify_doc_type(doc_type_string: str, *, model: str) -> DocTypeResponse:
    return cached_call(DocTypeResponse, DOC_TYPE_PROMPT, doc_type_string, model=model)


def classify_consent_signed(consent_text: str, *, model: str) -> ConsentSignedResponse:
    return cached_call(ConsentSignedResponse, CONSENT_SIGNED_PROMPT, consent_text, model=model)


# -------------------------
# Normalizer
# -------------------------


def _safe_to_utc_date(value: str | None) -> date | None:
    """Parse an ISO date/datetime string; return None on missing or malformed input."""
    if not value:
        return None
    try:
        return to_utc_date(value)
    except (ValueError, TypeError):
        return None


def _is_anticoag_medication(name: str | None, *, model: str) -> bool:
    """Allowlist fast path; LLM fallback for unmatched names.

    LLM `low` confidence is treated as a decline and routes to True (safer: flag).
    """
    name_lower = (name or "").strip().lower()
    if not name_lower:
        return False  # empty name can't be classified
    if name_lower in ANTICOAG_ALLOWLIST:
        return True
    response = classify_drug(name_lower, model=model)
    if response.confidence == "low":
        return True  # safer
    return response.is_anticoagulant


def classify_medications(
    medications: list[Medication], *, model: str,
) -> tuple[tuple[Sourced[Medication], ...], tuple[Sourced[Medication], ...]]:
    """Partition medications into (active anticoagulants, unknown-status anticoagulants).

    Non-anticoagulant medications and explicitly inactive (`active=False`) ones are dropped.
    """
    active: list[Sourced[Medication]] = []
    unknown: list[Sourced[Medication]] = []
    for idx, med in enumerate(medications):
        if not _is_anticoag_medication(med.name, model=model):
            continue
        srcd = Sourced(med, f"medications[{idx}]")
        if med.active is True:
            active.append(srcd)
        elif med.active is None:
            unknown.append(srcd)
        # active is False -> drop silently
    return tuple(active), tuple(unknown)


def _pick_most_recent(
    items: list[tuple[int, date, Any]],
) -> tuple[int, date, Any] | None:
    """Sort by date descending, ties broken by lower index. Empty input -> None."""
    if not items:
        return None
    # toordinal gives stable integer comparison; negate for descending.
    items.sort(key=lambda x: (-x[1].toordinal(), x[0]))
    return items[0]


def select_canonical_hp(
    documents: list[Document], procedure_date: date | None,
) -> Sourced[Document] | None:
    """Pick the canonical pre-op H&P.

    Filters by H&P type (regex match), then by date <= `procedure_date` (when
    known), then picks the most recent. Ties broken by lower document index.
    If `procedure_date` is None, no date filter is applied.

    Note: we do not filter by document text content (e.g. a "retained
    historical H&P" boilerplate marker). Doing so would leave no canonical H&P
    in records where the only available doc happens to carry such text - in
    which case the downstream rule engine would emit a "missing H&P" finding
    even though a document is present. Trusting the type field and relying on
    date-based selection keeps the rule engine's judgment honest: if the doc
    is too old, Rule 1 flags it out-of-window; if not, it stands.
    """
    candidates: list[tuple[int, date, Document]] = []
    for idx, doc in enumerate(documents):
        if not doc.type or not is_hp_type(doc.type):
            continue
        d = _safe_to_utc_date(doc.date)
        if d is None:
            continue
        if procedure_date is not None and d > procedure_date:
            continue
        candidates.append((idx, d, doc))
    picked = _pick_most_recent(candidates)
    if picked is None:
        return None
    idx, _, doc = picked
    return Sourced(doc, f"documents[{idx}]")


def _consent_signed_state(text: str, *, model: str) -> bool:
    """Returns True if the consent text indicates a signed consent.

    Keyword scan first (unsigned wins ties over signed since the unsigned set
    includes the literal word `signed` in some signed-keyword phrasings - but the
    unsigned set is checked first only when it actually matches concrete unsigned
    phrasings like "unsigned" or "awaiting signature"). LLM fallback only when
    neither keyword set fires.
    """
    if not text:
        return False  # no text -> safer to flag as unsigned
    text_lower = text.lower()
    has_unsigned = any(kw in text_lower for kw in UNSIGNED_CONSENT_KEYWORDS)
    has_signed = any(kw in text_lower for kw in SIGNED_CONSENT_KEYWORDS)
    if has_unsigned and not has_signed:
        return False
    if has_signed and not has_unsigned:
        return True
    if not has_unsigned and not has_signed:
        # Truly ambiguous; defer to LLM
        response = classify_consent_signed(text, model=model)
        if response.confidence == "low":
            return False  # safer
        return response.signed
    # Both matched - rare; prefer unsigned (safer)
    return False


def classify_consent(
    documents: list[Document], *, model: str,
) -> ConsentStatus:
    """Find the most relevant consent-typed doc and classify its signed/unsigned state.

    Multiplicity rule: if multiple consent-typed docs exist, prefer the most recent
    by date; ties by lower index. Docs without a parseable date sort to the end.
    None of the 50 sample records have multiple consent-typed docs.
    """
    candidates: list[tuple[tuple[int, int, int], int, Document]] = []
    for idx, doc in enumerate(documents):
        if not doc.type or not is_consent_type(doc.type):
            continue
        d = _safe_to_utc_date(doc.date)
        if d is None:
            sort_key = (1, 0, idx)  # bucket "no date", index ascending
        else:
            sort_key = (0, -d.toordinal(), idx)  # bucket "has date", date desc, index asc
        candidates.append((sort_key, idx, doc))
    if not candidates:
        return ConsentStatus(document=None, signed=False)
    candidates.sort(key=lambda x: x[0])
    _, idx, doc = candidates[0]
    signed = _consent_signed_state(doc.text or "", model=model)
    return ConsentStatus(document=Sourced(doc, f"documents[{idx}]"), signed=signed)


def select_most_recent_lab(
    labs: list[LabResult], canonical_code: str,
) -> Sourced[LabResult] | None:
    """Pick the most recent lab whose normalized code equals `canonical_code` (e.g. "CBC", "CMP")."""
    candidates: list[tuple[int, date, LabResult]] = []
    for idx, lab in enumerate(labs):
        if not lab.code:
            continue
        if normalize_lab_code(lab.code) != canonical_code:
            continue
        d = _safe_to_utc_date(lab.effective_at)
        if d is None:
            continue
        candidates.append((idx, d, lab))
    picked = _pick_most_recent(candidates)
    if picked is None:
        return None
    idx, _, lab = picked
    return Sourced(lab, f"labs[{idx}]")


def select_most_recent_vital(
    vitals: list[Any], vital_type: str,
) -> Sourced[Any] | None:
    """Most recent vital whose `type` field equals `vital_type` ("blood_pressure" or "temperature").

    Filters by the `type` field rather than Pydantic union membership, since `Vital` is
    a non-discriminated union and parsing can resolve unexpectedly.
    """
    candidates: list[tuple[int, date, Any]] = []
    for idx, vital in enumerate(vitals):
        if getattr(vital, "type", None) != vital_type:
            continue
        d = _safe_to_utc_date(getattr(vital, "date", None))
        if d is None:
            continue
        candidates.append((idx, d, vital))
    picked = _pick_most_recent(candidates)
    if picked is None:
        return None
    idx, _, vital = picked
    return Sourced(vital, f"vitals[{idx}]")


def evaluate_plan_adequacy(
    plan_documents: tuple[Sourced[Document], ...], *, model: str,
) -> PlanAdequacyResponse | None:
    """LLM-classify the most recent plan-typed document. None if no plan docs."""
    if not plan_documents:
        return None
    # Sort by date descending, ties by lower index; docs without parseable dates go last.
    def sort_key(s: Sourced[Document]) -> tuple[int, int, str]:
        d = _safe_to_utc_date(s.value.date)
        if d is None:
            return (1, 0, s.source_path)
        return (0, -d.toordinal(), s.source_path)
    sorted_plans = sorted(plan_documents, key=sort_key)
    plan_text = (sorted_plans[0].value.text or "").strip()
    if not plan_text:
        return PlanAdequacyResponse(adequate=False, reason="Plan document text is empty.")
    return classify_plan_adequacy(plan_text, model=model)


def normalize(submission: PatientSubmission, *, model: str) -> NormalizedState:
    """Compose all of the normalizer steps into a single immutable `NormalizedState`.

    Side effects: may invoke the LLM (drug classification, plan adequacy) which
    in turn writes to the cache file on misses. Pure with respect to its inputs
    once the cache is warm.
    """
    procedure = submission.procedure

    procedure_date: Sourced[date] | None = None
    if procedure and procedure.procedure_date:
        d = _safe_to_utc_date(procedure.procedure_date)
        if d is not None:
            procedure_date = Sourced(d, "procedure.procedure_date")

    procedure_risk: Sourced[ProcedureRisk] | None = None
    if procedure and procedure.procedure_risk:
        procedure_risk = Sourced(procedure.procedure_risk, "procedure.procedure_risk")

    documents = list(submission.documents or [])
    labs = list(submission.labs or [])
    vitals = list(submission.vitals or [])
    medications = list(submission.medications or [])

    canonical_hp = select_canonical_hp(
        documents,
        procedure_date.value if procedure_date else None,
    )
    consent = classify_consent(documents, model=model)
    most_recent_cbc = select_most_recent_lab(labs, "CBC")
    most_recent_cmp = select_most_recent_lab(labs, "CMP")
    most_recent_bp = select_most_recent_vital(vitals, "blood_pressure")
    most_recent_temp = select_most_recent_vital(vitals, "temperature")

    active_anticoagulants, unknown_status_medications = classify_medications(
        medications, model=model,
    )

    plan_documents = tuple(
        Sourced(doc, f"documents[{idx}]")
        for idx, doc in enumerate(documents)
        if doc.type and is_plan_type(doc.type)
    )

    plan_adequacy = None
    if active_anticoagulants and plan_documents:
        plan_adequacy = evaluate_plan_adequacy(plan_documents, model=model)

    all_doc_types = tuple(d.type for d in documents if d.type)
    all_lab_codes = tuple(lab.code for lab in labs if lab.code)

    return NormalizedState(
        procedure_date=procedure_date,
        procedure_risk=procedure_risk,
        canonical_hp=canonical_hp,
        consent=consent,
        most_recent_cbc=most_recent_cbc,
        most_recent_cmp=most_recent_cmp,
        active_anticoagulants=active_anticoagulants,
        unknown_status_medications=unknown_status_medications,
        plan_documents=plan_documents,
        plan_adequacy=plan_adequacy,
        most_recent_bp=most_recent_bp,
        most_recent_temp=most_recent_temp,
        all_doc_types=all_doc_types,
        all_lab_codes=all_lab_codes,
    )


# -------------------------
# Rule engine
# -------------------------


def _issue(
    category: IssueCategory,
    description: str,
    source: str,
    details: str,
) -> TriageIssue:
    return TriageIssue(
        category=category,
        description=description,
        evidence=TriageIssueEvidence(source=source, details=details),
    )


def _doc_type_list(state: NormalizedState) -> str:
    """Comma-separated list of unique document types, preserved in submitter order.

    Used as a grounding anchor in missing-doc issues.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for t in state.all_doc_types:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ", ".join(ordered)


def _lab_code_list(state: NormalizedState) -> str:
    """Comma-separated unique lab codes (original, not normalized).

    Used as a grounding anchor in missing-lab issues. The sample always includes
    HBA1C (5 chars), which suffices for fuzzy grounding via strategy (b).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for c in state.all_lab_codes:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ", ".join(ordered)


def missing_required_fields(state: NormalizedState) -> list[TriageIssue]:
    """Emit MISSING_REQUIRED_DATA issues for fields needed by downstream rules.

    Cross-cutting: applies regardless of which rule would have used the field.
    No cascade - a missing field blocks the dependent rule but does NOT emit
    downstream rule issues for the same root cause.
    """
    issues: list[TriageIssue] = []

    if state.procedure_date is None:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA",
            "Missing procedure date",
            "procedure.procedure_date",
            "procedure.procedure_date is null",
        ))

    if state.procedure_risk is None:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA",
            "Missing procedure risk",
            "procedure.procedure_risk",
            "procedure.procedure_risk is null",
        ))

    if state.most_recent_bp is None:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA",
            "Missing blood pressure vital",
            "vitals",
            "No blood_pressure vital with valid date found",
        ))

    if state.most_recent_temp is None:
        issues.append(_issue(
            "MISSING_REQUIRED_DATA",
            "Missing temperature vital",
            "vitals",
            "No temperature vital with valid date found",
        ))

    for srcd in state.unknown_status_medications:
        med = srcd.value
        issues.append(_issue(
            "MISSING_REQUIRED_DATA",
            "Unknown anticoagulant active status",
            srcd.source_path,
            f"Medication {med.name} has active=null; cannot determine if currently taking",
        ))

    return issues


_HP_WINDOW_DAYS = 30


def rule_1_documentation(state: NormalizedState) -> list[TriageIssue]:
    """Rule 1 - Required documentation.

    Short-circuits if `procedure_date` is missing (no cascade). H&P window
    of 30 days; consent must be signed.
    """
    issues: list[TriageIssue] = []

    if state.procedure_date is not None:
        # H&P checks
        if state.canonical_hp is None:
            types_str = _doc_type_list(state)
            issues.append(_issue(
                "REQUIRED_DOCUMENTATION",
                "History and Physical document missing",
                "documents",
                f"No History and Physical document with valid date found; document types present: {types_str}",
            ))
        else:
            hp_doc = state.canonical_hp.value
            hp_date = _safe_to_utc_date(hp_doc.date)
            if hp_date is not None:
                days_prior = (state.procedure_date.value - hp_date).days
                if days_prior > _HP_WINDOW_DAYS:
                    issues.append(_issue(
                        "REQUIRED_DOCUMENTATION",
                        "H&P outside 30-day window",
                        state.canonical_hp.source_path,
                        f"H&P date {hp_doc.date} vs procedure_date {state.procedure_date.value.isoformat()} "
                        f"({days_prior} days prior; must be within {_HP_WINDOW_DAYS})",
                    ))

    # Consent checks (independent of procedure_date)
    if state.consent.document is None:
        types_str = _doc_type_list(state)
        issues.append(_issue(
            "REQUIRED_DOCUMENTATION",
            "Signed surgical consent missing",
            "documents",
            f"No Surgical Consent document found; document types present: {types_str}",
        ))
    elif not state.consent.signed:
        consent_doc = state.consent.document.value
        excerpt = (consent_doc.text or "").strip()[:200]
        issues.append(_issue(
            "REQUIRED_DOCUMENTATION",
            "Surgical consent not clearly signed",
            state.consent.document.source_path,
            f"Consent document text does not clearly indicate signed consent: {excerpt}",
        ))

    return issues


_LOW_MODERATE_LAB_WINDOW_DAYS = 30
_HIGH_LAB_WINDOW_DAYS = 14


def _lab_window_days(risk: ProcedureRisk) -> int:
    return _HIGH_LAB_WINDOW_DAYS if risk == "HIGH" else _LOW_MODERATE_LAB_WINDOW_DAYS


def _check_lab_window(
    state: NormalizedState,
    lab_sourced: Sourced[LabResult],
    test_code: str,
    window_days: int,
) -> TriageIssue | None:
    """Emit a stale-lab issue if the lab's effective_at is outside the policy window."""
    assert state.procedure_date is not None  # caller checked
    lab = lab_sourced.value
    lab_date = _safe_to_utc_date(lab.effective_at)
    if lab_date is None:
        return None
    days_prior = (state.procedure_date.value - lab_date).days
    if days_prior > window_days:
        return _issue(
            "REQUIRED_TESTING",
            f"{test_code} outside window",
            lab_sourced.source_path,
            f"{test_code} effective_at {lab.effective_at} vs procedure_date "
            f"{state.procedure_date.value.isoformat()} ({days_prior} days prior; must be within {window_days})",
        )
    return None


def rule_2_testing(state: NormalizedState) -> list[TriageIssue]:
    """Rule 2 - Required testing by procedure risk.

    Short-circuits if procedure_date or procedure_risk is missing.
    """
    if state.procedure_date is None or state.procedure_risk is None:
        return []

    risk = state.procedure_risk.value
    window = _lab_window_days(risk)
    issues: list[TriageIssue] = []
    codes_str = _lab_code_list(state)

    # CBC check (applies to all risk levels)
    if state.most_recent_cbc is None:
        issues.append(_issue(
            "REQUIRED_TESTING",
            "CBC missing",
            "labs",
            f"No CBC result with valid effective_at found for procedure_risk {risk}; lab codes present: {codes_str}",
        ))
    else:
        stale = _check_lab_window(state, state.most_recent_cbc, "CBC", window)
        if stale:
            issues.append(stale)

    # CMP check (HIGH risk only)
    if risk == "HIGH":
        if state.most_recent_cmp is None:
            issues.append(_issue(
                "REQUIRED_TESTING",
                "CMP missing",
                "labs",
                f"No CMP result with valid effective_at found for procedure_risk HIGH; lab codes present: {codes_str}",
            ))
        else:
            stale = _check_lab_window(state, state.most_recent_cmp, "CMP", window)
            if stale:
                issues.append(stale)

    return issues


def rule_3_anticoagulation(state: NormalizedState) -> list[TriageIssue]:
    """Rule 3 - Anticoagulation management.

    Emits ANTICOAGULATION_MANAGEMENT when an active anticoagulant exists AND
    either no plan-typed doc is present, or LLM-judged plan adequacy is False.
    """
    if not state.active_anticoagulants:
        return []

    plan_inadequate_or_missing = (
        not state.plan_documents
        or (state.plan_adequacy is not None and not state.plan_adequacy.adequate)
    )
    if not plan_inadequate_or_missing:
        return []

    # Cite the first active anticoagulant (lowest medications[N] index)
    first = state.active_anticoagulants[0]
    med = first.value
    drug_name = (med.name or "anticoagulant").strip()
    return [_issue(
        "ANTICOAGULATION_MANAGEMENT",
        "Missing perioperative anticoagulation plan",
        first.source_path,
        f"Active anticoagulant {drug_name} ({first.source_path}) but no clear perioperative plan document found",
    )]


_SBP_THRESHOLD = 180
_DBP_THRESHOLD = 110
_TEMP_F_THRESHOLD = 100.4


def rule_4_acute_safety(state: NormalizedState) -> list[TriageIssue]:
    """Rule 4 - Acute safety exclusions, using only the most recent reading of each type."""
    issues: list[TriageIssue] = []

    bp = state.most_recent_bp
    if bp is not None:
        v = bp.value
        sbp = getattr(v, "systolic", None)
        dbp = getattr(v, "diastolic", None)
        if (isinstance(sbp, (int, float)) and sbp >= _SBP_THRESHOLD) or \
           (isinstance(dbp, (int, float)) and dbp >= _DBP_THRESHOLD):
            v_date = getattr(v, "date", "") or ""
            v_source = getattr(v, "source", "") or ""
            issues.append(_issue(
                "ACUTE_SAFETY_EXCLUSION",
                "Blood pressure meets exclusion threshold",
                bp.source_path,
                f"latest BP on {v_date} from {v_source}: systolic={sbp}, diastolic={dbp}; "
                f"threshold systolic>={_SBP_THRESHOLD} or diastolic>={_DBP_THRESHOLD}",
            ))

    temp = state.most_recent_temp
    if temp is not None:
        v = temp.value
        value_f = getattr(v, "value_f", None)
        if isinstance(value_f, (int, float)) and value_f > _TEMP_F_THRESHOLD:
            v_date = getattr(v, "date", "") or ""
            v_source = getattr(v, "source", "") or ""
            issues.append(_issue(
                "ACUTE_SAFETY_EXCLUSION",
                "Temperature exceeds exclusion threshold",
                temp.source_path,
                f"latest temperature on {v_date} from {v_source}: value_f={value_f}; "
                f"threshold is > {_TEMP_F_THRESHOLD}",
            ))

    return issues


def evaluate_rules(state: NormalizedState) -> list[TriageIssue]:
    """Compose missing-required-field emitter with the four rule functions."""
    return (
        missing_required_fields(state)
        + rule_1_documentation(state)
        + rule_2_testing(state)
        + rule_3_anticoagulation(state)
        + rule_4_acute_safety(state)
    )


# -------------------------
# Output construction
# -------------------------

_RULE_NUMBER: dict[str, int] = {
    "REQUIRED_DOCUMENTATION": 1,
    "REQUIRED_TESTING": 2,
    "ANTICOAGULATION_MANAGEMENT": 3,
    "ACUTE_SAFETY_EXCLUSION": 4,
    "MISSING_REQUIRED_DATA": 5,
}


def derive_decision(issues: list[TriageIssue]) -> Decision:
    """Acute safety wins; otherwise any issue → NEEDS_FOLLOW_UP; otherwise READY.

    The NOT_CLEARED precedence applies to the decision label only; all applicable
    issues are still reported in the issues list (see case_00002).
    """
    for issue in issues:
        if issue.category == "ACUTE_SAFETY_EXCLUSION":
            return "NOT_CLEARED"
    return "NEEDS_FOLLOW_UP" if issues else "READY"


def sort_issues(issues: list[TriageIssue]) -> list[TriageIssue]:
    """Deterministic ordering: rule number ascending, then source_path lexicographic."""
    return sorted(
        issues,
        key=lambda i: (_RULE_NUMBER.get(i.category, 99), i.evidence.source),
    )


def build_explanation(issues: list[TriageIssue]) -> str:
    """Join sorted issues as `"<CATEGORY>: <description>"` with " | "."""
    return " | ".join(f"{i.category}: {i.description}" for i in issues)


# -------------------------
# Entrypoint
# -------------------------


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Hybrid triage: deterministic rule engine, with narrow LLM extractors.

    Pipeline (design §3):
        PatientSubmission -> normalize -> NormalizedState -> rule engine
            -> issues -> decision derivation -> output builder -> TriageOutput
    """
    if isinstance(submission, PatientSubmission):
        validated = submission
    else:
        validated = PatientSubmission.model_validate(submission)

    state = normalize(validated, model=model)
    issues = evaluate_rules(state)
    issues = sort_issues(issues)
    decision = derive_decision(issues)
    explanation = build_explanation(issues)

    return TriageOutput(
        decision=decision,
        issues=issues,
        explanation=explanation,
    )
