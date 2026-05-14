# Tech Design — Pre-Op Triage System

## §1. Purpose & scope

Implementation design for the pre-op triage system in [`Cadence___Engineering_Take-Home.pdf`](../../Cadence___Engineering_Take-Home.pdf). Reader is assumed to have read the spec and Appendix A (the policy).

**In scope:** the `triage_submission` replacement in `core.py`, intermediate data shapes, rule engine, LLM extractors, output construction, determinism, error handling, testing.
**Out of scope:** harness scripts (`run_evals.py`, `run_baseline.py`, `view_report.py`); deployment.

**Major decisions:**

- **Hybrid architecture.** Deterministic rules for mechanical operations (date math, threshold checks, schema enforcement). LLM for free-text judgment.
- **LLM in the load-bearing path for one task: plan adequacy (Rule 3).** The policy explicitly requires judgment ("clear," "incomplete or ambiguous"). A rule of "always flag if anticoag is active" passes the sample by coincidence — every sample anticoag patient has an inadequate plan — but doesn't actually evaluate adequacy.
- **LLM as gated fallback** for drug identification, doc-type classification, and consent signed/unsigned detection. Rules cover every case the sample exercises; fallbacks fire only when the deterministic detector declines to match.
- **Rules only** for date arithmetic, lab code normalization, vital thresholds, output construction.
- **Determinism via content-addressed disk cache** of LLM responses, checked into the repo. First call non-deterministic; subsequent calls byte-exact. Rules are pure functions.
- **Output built in Python, not parsed from an LLM string.** Final `TriageOutput` constructed directly; LLM only returns small typed classifications consumed by the rule engine.

---

## §2. Requirements

The policy (spec Appendix A) defines four rules and a missing-data default; not duplicated here. The non-obvious requirements forced by the harness or the dataset:

| Requirement | Evidence |
|---|---|
| Output must validate as `core.TriageOutput`. `IssueCategory` is a closed `Literal` of 5 values; anything else parse-fails all 4 metrics. | `core.py:57-63` |
| `issue_categories_match_oracle` compares deduped **sets**. Multiple stale labs collapse to one `REQUIRED_TESTING`. | `run_evals.py:368-374` |
| `evidence.details` should embed literal submission values to pass the grounding fuzzy fallback. | `run_evals.py:245-323` |
| **No cascade** on missing `procedure_date`. Emit only `MISSING_REQUIRED_DATA` plus date-independent rule violations. | All 4 null-`procedure_date` records (00000, 00024, 00031, 00046) |
| `NOT_CLEARED` precedence is on the decision label, not the issue list. Acute safety + doc failures co-existing: decision `NOT_CLEARED`, issues include both. | Case 00002 |
| **Tri-state `medications.active`.** `active=null` → `MISSING_REQUIRED_DATA`, not "not taking the drug." | Cases 00001, 00023, 00037 (`warfarin` with `active=null`; oracle emits `MISSING_REQUIRED_DATA`) |
| Lab code normalization: strip `LAB-` prefix before matching. Inferred from `LAB-CBC`/`CBC` and `LAB-CMP`/`CMP` equivalence; no other `LAB-X` codes in sample to verify. | Sample lab corpus |
| HBA1C is dataset noise (never cited, not in policy). Ignore. | All 50 records |
| Doc-type matching needs substring/regex over synonyms; "History and Physical" appears verbatim once across 100+ unique type strings. | Doc-type inventory |
| Canonical H&P: most recent H&P-typed doc on or before `procedure_date`; ties by lower document index. Submitter ordering (`doc[0]` canonical in all 38) is consistent in sample but not a stable contract. The boilerplate `"Prior pre-op H&P retained..."` text is a synthetic-data artifact, optional fast-filter. | All 36 applicable multi-H&P records |
| Consent signed/unsigned by text keyword scan (lists in §6); ambiguous text (neither list matches) defers to LLM fallback in §7. | Cases 00008, 00017, 00044 (unsigned); case 00000 (signed sample) |
| Date math: integer day delta on UTC-normalized dates. | All out-of-window citations use integer "N days prior" |
| Citation source conventions: bare collection (`documents`, `labs`) for "missing entirely"; indexed (`labs[2]`) for "specific item out of policy"; dotted path (`procedure.procedure_date`) for missing field. | Oracle citation patterns |

**Operational:** outputs must be byte-stable across repeated calls (audit trail). Failure-mode asymmetry: false negatives are clinically expensive — when in doubt, flag. Every output is a valid `TriageOutput` or an explicit exception; never a silent partial response.

---

## §3. High-level architecture

A single-input transform: `PatientSubmission → TriageOutput`. No service boundaries.

```
PatientSubmission → Normalizer → NormalizedState → Rule engine → Issues
                                                                    ↓
                                              TriageOutput ← Output builder ← Decision
```

Each stage is a pure function. The split between **normalizer** (derive facts with provenance) and **rule engine** (apply policy to derived facts) keeps rule code small and obviously correct; it also separates issue collection from decision derivation, which the policy itself separates (NOT_CLEARED precedence applies to the decision label while all applicable issues are still reported — see case 00002).

**LLM enters in two ways:**

- *Load-bearing*: plan adequacy on Rule 3. When the patient has an active anticoagulant and a plan-typed doc exists, the text goes to an LLM returning `{adequate: bool, reason: str}`. On the sample this fires 7 times and always returns `inadequate` — but the system is actually evaluating, not pattern-matching apixaban presence.
- *Gated fallbacks*: drug identification, doc-type classification, consent signed/unsigned detection. Each fires only when the deterministic detector declines to match. On the sample, none fire.

Every LLM invocation is cached by `sha256(model || prompt || input)`. First call non-deterministic; subsequent calls hit the cache and are byte-exact. The cache file is checked into the repo.

**Not in the architecture:** no LLM for mechanical operations (date math, thresholds, output construction); no multi-step prompt chains, critic loops, or self-consistency voting; no JSON parsing of LLM strings into the final output.

**Three canonical flows:** (1) clean state → empty issues → `READY` (3 sample records); (2) single rule violation → one issue → `NEEDS_FOLLOW_UP` (37 records); (3) acute safety co-occurring with other failures → multiple issues, decision `NOT_CLEARED` (case 00002).

---

## §4. Tech stack & library choices

Forced by the environment: Python ≥ 3.11, Pydantic v2, OpenAI Python SDK, `uv`. Implementation lives in `core.py`; new files (tests, cache) added alongside without modifying harness scripts.

**Load-bearing decisions:**

- **LLM response cache: single JSON file on disk.** Reviewable, zero deps, ≤7 entries after a sample run. Disk persistence (vs in-memory) is required for cross-clone audit replay, not for the in-process determinism harness. Rejected: `diskcache`/SQLite (unnecessary infrastructure).
- **Structured outputs for LLM calls: OpenAI `strict: true` with hand-crafted JSON schemas.** Eliminates the parse-failure class (case 00034 emitted `"NEEDS_FOLLOW_UP"` as an `IssueCategory` and lost all 4 metrics). Schemas are 2–3 properties each; cheaper than introducing `instructor`.
- **Date handling: stdlib `datetime` with explicit UTC conversion.** One helper (`to_utc_date`, §6) reused across rules.

Provider stays on OpenAI: path of least resistance, key provided; LLM-calling layer is thin and would be localized to swap.

**Defaults:** stdlib `logging`; `pytest`; rely on Pydantic for runtime validation. No vector DB, no async, no web framework, no prompt-engineering libraries.

---

## §5. Data model

Four shapes: `PatientSubmission` (input, Pydantic — exists in `core.py`); `TriageOutput` (wire output, Pydantic — exists in `core.py`); `NormalizedState` (internal, frozen dataclass); LLM extractor request/response pairs (Pydantic, for schema export). Pydantic earns its keep at boundaries; internal types use frozen dataclasses (no validation overhead on already-validated derived data, and immutability enforced).

### NormalizedState

```python
@dataclass(frozen=True)
class Sourced(Generic[T]):
    value: T
    source_path: str   # "documents[0]", "vitals[2]", "procedure.procedure_date"

@dataclass(frozen=True)
class NormalizedState:
    procedure_date:    Sourced[date] | None
    procedure_risk:    Sourced[ProcedureRisk] | None
    canonical_hp:      Sourced[Document] | None
    consent:           ConsentStatus              # doc + tri-state signed flag
    most_recent_cbc:   Sourced[LabResult] | None
    most_recent_cmp:   Sourced[LabResult] | None
    active_anticoagulants:       tuple[Sourced[Medication], ...]
    unknown_status_medications:  tuple[Sourced[Medication], ...]   # anticoag-classified AND active=null
    plan_documents:              tuple[Sourced[Document], ...]
    plan_adequacy:     PlanAdequacyResponse | None
    most_recent_bp:    Sourced[BloodPressureVital] | None
    most_recent_temp:  Sourced[TemperatureVital] | None
```

Provenance captured once; `Sourced[T] | None` makes missing data first-class. Vitals are filtered by `type` field after Pydantic parses (the `Vital` union has no discriminator, so union membership alone is unreliable).

**Tri-state `medications.active`** (third state routes to `MISSING_REQUIRED_DATA`, never silently treated as "absent"):

| Value | Meaning |
|---|---|
| `True` | On drug |
| `False` | Not on drug |
| `None` | Unknown — flag if anticoag-classified |

`ConsentStatus.signed` is `bool`: `True` if signed-keyword matches OR LLM returns signed with non-low confidence; `False` otherwise (unsigned-keyword match, LLM classifies as unsigned, or LLM low-confidence decline). Absence of a consent doc is represented by `ConsentStatus.document is None`, not by `signed`.

### LLM schemas

Small Pydantic models, exported as JSON schemas with `strict: true`:

```python
class PlanAdequacyResponse(BaseModel):
    adequate: bool
    reason: str

class DrugClassificationResponse(BaseModel):
    is_anticoagulant: bool
    confidence: Literal["high", "medium", "low"]

class DocTypeResponse(BaseModel):
    role: Literal["history_and_physical", "surgical_consent", "anticoag_plan", "other"]
    confidence: Literal["high", "medium", "low"]

class ConsentSignedResponse(BaseModel):
    signed: bool
    confidence: Literal["high", "medium", "low"]
```

`low` confidence treated as a decline → safer (flag) branch. Encodes "default to flag on uncertainty" at the data-model level.

### Cache record

`data/llm_cache.json`, keyed by `sha256(model || canonical_prompt || canonical_input)`:

```json
{
  "<hash>": {
    "schema_name": "PlanAdequacyResponse",
    "model": "gpt-4.1-mini",
    "response": { "adequate": false, "reason": "Plan defers to cardiology..." },
    "cached_at": "<ISO timestamp, metadata only — never participates in hash>"
  }
}
```

Checked into the repo. After a sample run: ≤7 entries (plan-adequacy calls only; fallbacks don't fire on sample).

---

## §6. Rule-by-rule design

### Grounding invariant (load-bearing)

Every non-`MISSING_REQUIRED_DATA` issue's `details` MUST contain at least one **≥4-character literal value** drawn from the cited submission item or another item in the category-relevant section. This satisfies the harness's `issues_value_grounding` check (`run_evals.py:245-323`). Numeric values <4 chars are silently rejected (`run_evals.py:215`) — why the baseline's BP citations failed. `MISSING_REQUIRED_DATA` issues are exempt (`run_evals.py:187-189`). The "Grounding anchor" column in each rule table names the value(s) that satisfy the invariant.

### Lookup tables

| Registry | Value |
|---|---|
| Anticoagulant allowlist | `apixaban, warfarin, rivaroxaban, dabigatran, edoxaban, heparin, enoxaparin` (unmatched → LLM drug-class fallback) |
| H&P type regex (case-insensitive) | `\b(h\s*&\s*p \| h\s+and\s+p \| h\s*/\s*p \| h\s*\+\s*p \| history\s+(?:and\|&\|/)\s+physical \| hist\s+&?\s+phys \| hx\s+&?\s+physical \| history/physical)\b` (whitespace around `\|` is for readability; implementation collapses) |
| Plan-doc regex (case-insensitive on `type`) | `perioperative\s+medication\s+(plan\|review) \| anticoag(ulation)?\s+plan \| cardiology\s+progress\s+note\s*-\s*anticoag` |
| Consent type matcher | Case-insensitive substring `consent` in `type`. All 50 sample records treat any `consent`-typed doc as a valid surgical consent. Production non-surgical consents (HIPAA, research, etc.) deferred to §11. |
| Unsigned-consent keywords | `unsigned`, `awaiting signature`, `signature not yet`, `signature pending` |
| Signed-consent keywords | `signed`, `signature on file`, `signed by`, `electronic consent obtained` |
| Historical-H&P signal (not used) | The sample contains a byte-identical boilerplate `"Prior pre-op H&P retained for longitudinal chart context."` on retained historical H&P docs. Considered as a fast-filter but ultimately not implemented: date-based selection (above) picks correctly without it, and a hard filter would mis-handle records where the only available H&P happens to have this marker (e.g., case_00002). Documented here as a sample-specific signal that exists but is not relied on. |
| Lab code normalization | Strip `LAB-` prefix; for Rule 2 matching, only consider canonical codes `{CBC, CMP}`. |
| Date normalization | `to_utc_date(s)` accepts ISO 8601 dates or datetimes (with `Z` or `+/-HH:MM` offsets); naive datetimes assumed UTC. Date math: `(to_utc_date(d1) - to_utc_date(d2)).days`. |

### Missing-required-field issues (cross-cutting)

Emitted by the normalizer when a required field is absent. **No cascade**: a missing field blocks the dependent rule but does NOT emit downstream-rule issues.

| Trigger | Source | Details |
|---|---|---|
| `procedure.procedure_date is None` | `procedure.procedure_date` | `procedure.procedure_date is null` |
| `procedure.procedure_risk is None` | `procedure.procedure_risk` | `procedure.procedure_risk is null` |
| No blood_pressure vital | `vitals` | `No blood_pressure vital with valid date found` |
| No temperature vital | `vitals` | `No temperature vital with valid date found` |
| Anticoag-classified medication with `active=null` | `medications[N]` | `Medication <name> has active=null; cannot determine if currently taking` |

(Non-anticoag medications with `active=null` are out of scope.)

### Rules

Each rule is a pure function over `NormalizedState`. Missing required fields are handled above; rules short-circuit on absent inputs.

**Rule 1 — Documentation**

| Trigger | Source | Details | Grounding anchor |
|---|---|---|---|
| `canonical_hp is None` | `documents` | `No History and Physical document with valid date found; document types present: <type_list>` | doc `type` strings via fuzzy strategy (b) |
| H&P > 30 days old | `documents[N]` | `H&P date <d1> vs procedure_date <d2> (D days prior; must be within 30)` | `<d1>` from `documents[N].date` |
| `consent.document is None` | `documents` | `No Surgical Consent document found; document types present: <type_list>` | doc `type` strings |
| `consent.signed is False` | `documents[N]` | `Consent document text does not clearly indicate signed consent: <text excerpt>` | `<excerpt>` is a ≥8-char substring of `documents[N].text` |

**Rule 2 — Testing** (window: 30 days for LOW/MODERATE, 14 days for HIGH)

| Trigger | Source | Details | Grounding anchor |
|---|---|---|---|
| CBC missing | `labs` | `No CBC result with valid effective_at found for procedure_risk <R>; lab codes present: <code_list>` | present lab `code` values (sample always includes `HBA1C` ≥4 chars) ground via fuzzy strategy (b) |
| CBC stale | `labs[N]` | `CBC effective_at <ts> vs procedure_date <d> (D days prior; must be within <W>)` | `<ts>` is full `labs[N].effective_at` |
| CMP missing (HIGH only) | `labs` | analogous | analogous |
| CMP stale (HIGH only) | `labs[N]` | analogous | `<ts>` from `labs[N].effective_at` |

**Rule 3 — Anticoagulation**

| Trigger | Source | Details | Grounding anchor |
|---|---|---|---|
| Active anticoag AND (no plan doc OR `plan_adequacy.adequate=False`) | `medications[N]` | `Active anticoagulant <drug_name> (medications[N]) but no clear perioperative plan document found` | `<drug_name>` from `medications[N].name` |

If multiple plan-typed docs, pick the most recent by `date`. The `unknown_status_medications` case routes through the cross-cutting `MISSING_REQUIRED_DATA` table.

**Rule 4 — Acute safety** (most recent reading only; missing-vital cases are cross-cutting)

| Trigger | Source | Details | Grounding anchor |
|---|---|---|---|
| SBP ≥ 180 or DBP ≥ 110 | `vitals[N]` | `latest BP on <date> from <source>: systolic=<s>, diastolic=<d>; threshold systolic>=180 or diastolic>=110` | `<date>` (10-char) or `<source>` from `vitals[N]` |
| Temp > 100.4 °F | `vitals[N]` | `latest temperature on <date> from <source>: value_f=<v>; threshold is > 100.4` | same |

### Decision derivation

```
if any issue.category == ACUTE_SAFETY_EXCLUSION → NOT_CLEARED
elif issues non-empty                            → NEEDS_FOLLOW_UP
else                                              → READY
```

Issues sorted by `(rule_number, source_path)` where `1=REQUIRED_DOCUMENTATION, 2=REQUIRED_TESTING, 3=ANTICOAGULATION_MANAGEMENT, 4=ACUTE_SAFETY_EXCLUSION, 5=MISSING_REQUIRED_DATA`. Deterministic across runs; ordering doesn't affect the score (categories are scored as deduped sets) and the oracle isn't internally consistent on ordering, so we don't match it. `explanation` joins issues as `"<CATEGORY>: <description>"` with `" | "`.

---

## §7. LLM usage strategy

Four schema-constrained calls, all routed through a single cached helper:

| Call | Schema | Fires when |
|---|---|---|
| Plan adequacy (load-bearing) | `PlanAdequacyResponse` | Active anticoagulant + ≥1 plan-typed doc |
| Drug class (fallback) | `DrugClassificationResponse` | Medication name doesn't match allowlist |
| Doc type (fallback) | `DocTypeResponse` | Doc type doesn't match any role regex. NOT invoked for positive consent-substring matches. |
| Consent signed (fallback) | `ConsentSignedResponse` | Consent doc present AND neither keyword list matches |

**Cached helper:**

```python
def cached_call(schema, prompt, user_input, *, model):
    key = sha256_hex(
        schema.__name__.encode("utf-8") + b"\0" +
        model.encode("utf-8") + b"\0" +
        prompt.encode("utf-8") + b"\0" +
        canonical_json(user_input).encode("utf-8")
    )
    if key in cache: return schema.model_validate(cache[key]["response"])
    response = openai_call(model, prompt, user_input, schema, strict=True, temperature=0)
    cache[key] = {"schema_name": schema.__name__, "model": model,
                  "response": response.model_dump(), "cached_at": now()}
    persist_cache()
    return response

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
```

`\0` separators between hash components prevent collision attacks. `canonical_json` provides stable serialization across processes. `cached_at` is metadata only (excluded from hash, never read into output). Cache loaded at module import; persisted on each miss.

**Prompts:** ≤200 tokens each, one example, return schema only. Stored as constants in `core.py` so they participate in the content hash.

**Confidence handling:** `low` → treat as decline → safer (flag) branch. No retry logic needed.

---

## §8. Determinism strategy

- **Rule engine: pure functions** → byte-identical output by construction.
- **LLM responses: disk cache** keyed by content hash, checked into the repo.
- **No temperature/seed reliance.** OpenAI doesn't guarantee determinism with `temperature=0`; the cache is the contract.
- **Stable ordering**: issues sorted by `(rule_number, source_path)`.
- **Stable JSON**: Pydantic `model_dump(mode="json")`.

**In-process determinism** (`make determinism` loops 10× in one process): iteration 1 populates the cache; iterations 2–10 hit it → `exact_output_match_pct = 100`, up from baseline 10%.

**Cross-clone reproducibility**: on a fresh clone, run `make baseline` once to populate `data/llm_cache.json`, then commit it. Subsequent clones hit the committed cache. Document this in the submission note (`SUBMISSION.md` at repo root; the take-home's own `README.md` is left unchanged).

---

## §9. Error handling

- **Input validation**: Pydantic on `PatientSubmission`. Reject malformed inputs at the boundary.
- **Missing required fields**: normalizer produces `None`; rules emit `MISSING_REQUIRED_DATA`. No raised exceptions.
- **Malformed dates**: Pydantic catches at validation; anything that slips through raises.
- **LLM call failures**: retry once (1s/2s backoff); on second failure, raise. Schema validation failures (rare under `strict: true`) raise.
- **Cache file corruption**: rebuild from scratch, log warning.
- **Enum drift**: prevented by Pydantic `Literal` types; tests cover exhaustively.

---

## §10. Testing strategy

- **Unit tests** (`test_core.py`) on the normalizer and each rule. Coverage targets: tri-state `active` flag, historical-H&P filter, lab code normalization, date-window edge cases (exactly on threshold), most-recent vital selection, missing-required-field paths.
- **LLM-fallback path tests**: the three fallbacks fire 0× on the sample. Mock the LLM and exercise each path so a refactor can't silently break the wiring.
- **Integration via harness**: `make baseline` + `make evals`. Per-metric targets:
  - `decision_match_oracle`: 100% (rule engine deterministic)
  - `issue_categories_match_oracle`: 100% (categories specified by construction per §6)
  - `json_schema_valid`: 100% (output built directly)
  - `issues_value_grounding`: ≥95% (grounding-anchor column embeds a value for every non-missing issue; production-edge cases like 3-char-only lab codes may leave specific records vulnerable)
  - Aggregate (`run_evals.py:40`): `(1.0 + 1.0 + 1.0 + 0.5 × 0.95) / 3.5 ≈ 99%`

  The 5 baseline misses (00007, 00025, 00034, 00040, 00048) become regression cases.
- **Determinism via harness**: `make determinism` 10× on case_00000 → `exact_output_match_pct = 100`.
- **Manual TUI review** (`make report`) for right-answer-for-wrong-reason failures.

---

## §11. Open questions & documented assumptions

- **Anesthesia/Consult H&P precision.** `Pre-anesthesia H&P` and `Consult H&P` match the H&P regex. Policy is silent on whether these satisfy Rule 1; design accepts them.
- **Typo'd doc types.** `"History & Phsyical"` (case 00002) rejected by regex; relies on LLM doc-type fallback in production.
- **Free-text dates not extracted.** All date math uses structured fields (`procedure.procedure_date`, `documents[i].date`, `labs[i].effective_at`, `vitals[i].date`). Case_00000 confirms the convention: null `procedure_date` is `MISSING_REQUIRED_DATA` even when H&P text mentions a target date. Revisit if production data systematically uses free-text dates.
- **`LAB-X` generalization** unverified for codes beyond CBC/CMP.
- **Plan-adequacy "adequate" branch untested.** Sample has zero clearly-adequate anticoag plans; hidden test would be the first real exercise.
- **ACM citation source.** Oracle splits between `documents[N]` (6 cases) and `documents` (1 case); design uses `medications[N]` instead — semantically clearer, grounding-equivalent via fuzzy strategy (b), and source isn't directly graded.
- **Cache growth unbounded.** Fine at current scale; TTL/size cap needed if it grew to thousands.
- **Provider lock-in** to OpenAI `strict: true` schema export; localized to one helper.
- **Non-surgical consent types** (HIPAA, research, photography, transfusion): substring `consent` matcher would over-accept. None in sample. Mitigation if observed: extend doc-type LLM fallback with a "consent kind" branch.
