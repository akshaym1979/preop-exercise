# Implementation Tasks — Pre-Op Triage System

Sequenced task list for implementing the design in [`design.md`](./design.md). Each task is sized for ~30 min – 2 hr of focused work. The order is the natural build order; checking them off in sequence gives a runnable end-to-end system at T22, with tests and verification following.

Parenthesized references point to the design doc section that specifies the task.

## Setup

- [ ] T1. Create supporting files. Initialize `data/llm_cache.json` as an empty JSON object, and create a stub `test_core.py` alongside `core.py`.
- [ ] T2. Implement `to_utc_date(s) -> date`. Handles ISO date strings, ISO datetimes with `Z` or `+/-HH:MM` offsets; naive datetimes assumed UTC. (§6 lookup tables)

## Types & lookup tables

- [ ] T3. Define internal types: `Sourced[T]` as a frozen Generic dataclass, `NormalizedState` as a frozen dataclass with 12 fields, and `ConsentStatus` with `document` (Sourced Document or None) and `signed: bool`. (§5)
- [ ] T4. Define LLM response models: `PlanAdequacyResponse`, `DrugClassificationResponse`, `DocTypeResponse`, `ConsentSignedResponse` — Pydantic, `strict: true`-compatible schemas. (§5)
- [ ] T5. Define lookup tables as module constants: anticoagulant allowlist, H&P type regex (compiled), plan-doc regex (compiled), consent substring matcher, signed/unsigned keyword sets, retained-H&P text marker. (§6 lookup tables)
- [ ] T6. Implement type predicates: `is_hp_type`, `is_plan_type`, `is_consent_type`.

## Cache & LLM layer

- [ ] T7. Implement cache helpers: `canonical_json` (sorted keys, separators, ASCII), `load_cache`, atomic `persist_cache`. (§7)
- [ ] T8. Implement `cached_call`. Hash key via SHA-256 of the model id, prompt, and canonical-JSON input with null-byte separators; check cache, call OpenAI with `strict: true` and `temperature=0`, write-through on miss. Full signature in §7. (§7)
- [ ] T9. Define LLM prompts as module constants. Four constants, ≤200 tokens each, one example each, return-schema-only instruction. (§7)
- [ ] T10. Implement four LLM call wrappers: `classify_plan_adequacy`, `classify_drug`, `classify_doc_type`, `classify_consent_signed`. Each wraps `cached_call`. Confidence `low` is treated as a decline; route to the safer (flag) branch. (§7)

## Normalizer

- [ ] T11. Implement canonical H&P selection. Filter H&P-typed docs to those dated on or before `procedure_date`, optionally pre-filter retained-text docs, sort descending by date with ties broken by lower index, return a Sourced Document or None. (§6 lookup tables and Rule 1)
- [ ] T12. Implement consent classification. Find consent-typed doc; signed-keyword scan, then unsigned-keyword scan; LLM fallback only when neither matches; build `ConsentStatus`. (§6 Rule 1 and §7)
- [ ] T13. Implement lab selection. Strip `LAB-` prefix; group by canonical code in CBC or CMP; pick most recent by `effective_at`. Return Sourced LabResult or None for CBC and CMP separately. (§6 Rule 2)
- [ ] T14. Implement vital selection. Filter vitals by `type` field (`blood_pressure` or `temperature`) — do not rely on Pydantic union resolution; pick most recent of each type. (§5, §6 Rule 4)
- [ ] T15. Implement medication classification. Separate medications into active anticoagulants (allowlist plus LLM-classified for unknown names) and unknown-status anticoagulants (`active=null` AND anticoag-classified). (§6 Rule 3 and cross-cutting)
- [ ] T16. Implement plan-adequacy evaluation. Only invoked when an active anticoagulant exists AND at least one plan-typed doc exists. Pick the most recent plan doc; call `classify_plan_adequacy`. (§6 Rule 3 and §7)
- [ ] T17. Wire the `normalize` function that produces a `NormalizedState`. Composes T11 through T16. (§3, §5)

## Rule engine

- [ ] T18. Implement the missing-required-field emitter. Five triggers per the cross-cutting table. (§6 cross-cutting)
- [ ] T19. Implement Rules 1 through 4 as four pure functions over `NormalizedState`. Each returns a list of issues. Templates per §6 rule tables, with grounding-anchor values embedded in `details`. (§6 Rules 1–4)
- [ ] T20. Wire the `evaluate_rules` function. Concatenate missing-required-field emitter plus four rule outputs. (§3)

## Output construction

- [ ] T21. Implement decision derivation, issue ordering, and explanation builder. Decision: `ACUTE_SAFETY_EXCLUSION` implies `NOT_CLEARED`; else issues-empty implies `READY`; else `NEEDS_FOLLOW_UP`. Issues sorted by rule number then source path. Explanation joins category and description pairs with the pipe delimiter. (§6 decision derivation)
- [ ] T22. Wire `triage_submission` end-to-end. Replace the baseline implementation in `core.py`. Compose: normalize, then evaluate_rules, then derive_decision, then sort, then build_explanation, then construct a `TriageOutput`. Full signature in §3. (§3)

## Testing

- [ ] T23. Unit tests for the normalizer: type predicates; historical-H&P filtering; canonical H&P selection (single, multi, tie-by-index, retained-text); tri-state `active` flag; lab code normalization; most-recent vital by type; plan-adequacy invocation gating.
- [ ] T24. Unit tests for each rule, including missing-required-field paths; date-window edge cases (exactly 30 and 14 days); cross-cutting cases (vitals empty, `procedure_risk` null). (§10)
- [ ] T25. LLM-fallback mock tests. Exercise each of the four call types under both decline and accept branches; ensure the wiring is covered even though the sample does not fire fallbacks. (§10)
- [ ] T26. Per-record regression tests for the 5 baseline misses: cases 00007, 00025, 00034, 00040, 00048. (§10)

## Verification against harness

- [ ] T27. Run `make baseline` with the new implementation. Investigate any per-record errors.
- [ ] T28. Run `make evals` and inspect aggregate. Targets per §10: decision, categories, and schema each at 100 percent; grounding at 95 percent or higher; aggregate around 99 percent.
- [ ] T29. Run `make report` (TUI) and drill into any failures. Look for right-answer-for-wrong-reason patterns.
- [ ] T30. Run `make determinism` on case_00000. Confirm `exact_output_match_pct` is 100. (§8)
- [ ] T31. Commit `data/llm_cache.json` with the sample's plan-adequacy entries (at most 7). (§8)

## Submission packaging

Prerequisites already done: fork created at github.com/your-handle/preop-exercise; local `origin` repointed to the fork; `upstream` (optional) points to cadencerpm/preop-exercise.

- [ ] T32. Set up `.gitignore` for internal files. Add `notes/analysis-journal.md` (and optionally `notes/tasks.md` if you'd rather not share the implementation checklist) so they don't accidentally end up in the fork.
- [ ] T33. Create a submission branch off main: run `git checkout -b submission`. Keeps the fork's `main` clean and lets evaluators view a self-contained branch.
- [ ] T34. Create `SUBMISSION.md` at repo root (alongside the existing `README.md`, which is left untouched). Include: a pointer to `notes/design.md`, brief build/run notes (only deviations from the existing README's workflow), and the one-time cache-seeding step from §8. Do not link to `notes/analysis-journal.md`.
- [ ] T35. Stage and commit all submission files in one coherent commit (or a few logical ones): `core.py`, `SUBMISSION.md`, `notes/design.md`, `data/llm_cache.json`, `test_core.py`, `.gitignore`.
- [ ] T36. Push the submission branch to your fork: `git push -u origin submission`.
- [ ] T37. Share the fork URL or branch URL with evaluators per the take-home's submission instructions (zip, PR-against-fork, or direct link — whichever applies).

---

Parallelism notes: the order shown is the natural critical path. Independent pairs that could run concurrently if multiple developers were involved:

- T5 (lookup tables) with T7 through T10 (cache and LLM layer)
- T23 through T26 (tests) with T27 through T28 (harness verification)

For a single implementer, sequential order is fine. The first end-to-end runnable point is T22; the first scoreable point is T27.
