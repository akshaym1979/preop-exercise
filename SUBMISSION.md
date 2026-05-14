# Pre-Op Triage — Submission Notes

**Submitted by:** Akshay More

This file accompanies the take-home submission. The original `README.md` (build/run instructions from the exercise) is unchanged; everything below is additive.

## Submission contents

- **`notes/design.md`** — full technical design (architecture, data model, rule-by-rule design, LLM strategy, determinism, error handling, testing, open questions).
- **`core.py`** — the implementation. The original baseline `triage_submission` has been replaced with a hybrid (rule engine + narrow LLM extractors). All other harness scripts (`run_evals.py`, `run_baseline.py`, `view_report.py`) are unchanged.
- **`test_core.py`** — unit and regression tests (86 tests; run with `uv run test_core.py` or `pytest test_core.py`).
- **`data/llm_cache.json`** — checked-in LLM response cache (7 entries from a warmed baseline run). See "Determinism" below for why it's committed.

## What changed in `core.py`

- Removed the baseline `BASELINE_SYSTEM_PROMPT`, `build_user_prompt`, and the LLM-only `triage_submission`.
- Added: type predicates, lookup tables (anticoagulant allowlist, regexes, consent-keyword sets), a content-hashed disk cache, three narrow LLM extractors (plan adequacy, drug class, consent signed), the normalizer, the rule engine, and a new `triage_submission` that wires them together.
- The wire types (`PatientSubmission`, `TriageOutput`, `TriageIssue`, etc.) and `triage_output_json_schema()` are kept as the harness imports them.

## Run instructions (unchanged from `README.md`)

```bash
make baseline       # runs new triage_submission on the 50-record sample
make evals          # scores outputs against oracle
make determinism    # 10x in-process determinism check on case_00000
make report         # TUI for inspecting per-case results
```

The new implementation is a drop-in replacement; no additional steps required.

## Results on the 50-record sample

| Metric | New implementation | Baseline (for reference) |
|---|---|---|
| `decision_match_oracle` | 100% | 90% |
| `issue_categories_match_oracle` | 100% | 74% |
| `json_schema_valid` | 100% | 98% |
| `issues_value_grounding` | 100% | 66% |
| **Aggregate** | **100.0%** | **84.29%** |
| `exact_output_match_pct` (10× determinism) | **100%** | 10% |

All 5 records the baseline got wrong (00007, 00025, 00034, 00040, 00048) are now correctly handled and covered as regression tests in `test_core.py`.

## Determinism and the checked-in cache

The system caches LLM responses on disk by content hash (`sha256(schema_name || model || prompt || canonical_input)`). On the sample, the cache warms to **7 entries**: 5 plan-adequacy responses (the 7 apixaban patients share 5 unique plan-doc texts) and 2 drug-class fallbacks (lisinopril, metformin).

**The cache file is intentionally checked into the repo.** Without it, a fresh clone running `make baseline` would invoke the LLM on its first iteration and produce LLM-drift-dependent output. With the committed cache, every clone produces byte-identical output on every metric. This trades a small repo footprint (~3 KB) for cross-clone reproducibility, which the design treats as part of the audit-trail story (see `notes/design.md` §8).

If you'd rather see the cache regenerate from scratch:

```bash
rm data/llm_cache.json && echo '{}' > data/llm_cache.json
make baseline
```

## Where LLM is used (and where it isn't)

- **Load-bearing**: plan adequacy assessment (Rule 3). The policy explicitly requires a judgment call ("clear," "incomplete or ambiguous"), and there's no mechanical way to determine adequacy from a free-text plan document. Fires on the 7 apixaban patients in the sample.
- **Gated fallbacks** (fire only when the deterministic detector declines): drug classification, consent signed/unsigned classification. On the sample, only drug classification fires (for lisinopril and metformin — both correctly classified as non-anticoagulants).
- **Rules only** for: date arithmetic, lab code normalization, vital threshold checks, output construction, schema enforcement.

Every LLM call uses OpenAI Responses API with `strict: true` JSON-schema output (eliminating the parse-failure class that hit the baseline on case_00034). Each response is validated against a small Pydantic schema before being consumed by the rule engine.

## Documented assumptions

See `notes/design.md` §11 for the full list. The two worth surfacing here:

- **Plan adequacy "adequate" branch is structurally present but untested on the sample.** All 7 sample plans are inadequate, so the LLM has only been exercised in one direction. A hidden test with a clearly-adequate plan would be the first real test of Rule 3's positive branch.
- **The consent matcher uses substring `"consent"`.** Verified against all 50 sample records as treating every consent-typed doc as a valid surgical consent (including the otherwise-ambiguous `Consent Counseling Note`). In production, non-surgical consent variants (HIPAA, research, photography, transfusion) would need a separate doc-kind classifier — not done here because zero such cases appear in the sample.

## Testing

`test_core.py` covers:
- to_utc_date edge cases (Z suffix, offsets, naive)
- Type predicates (positive + negative cases)
- Canonical H&P selection (single, multi, tie-by-index, retained-text-as-only-HP, no procedure_date)
- Lab selection with prefix normalization
- Vital selection by `type` field (not Pydantic union membership)
- Tri-state medication active flag (True/False/null)
- LLM fallbacks under both accept and low-confidence-decline branches
- Date-window edge cases (exactly 30 days, exactly 14 days; boundary at 100.4 °F)
- Each rule + missing-required-field cross-cutting emitter
- Decision derivation + issue ordering
- Regression tests for the 5 baseline misses
