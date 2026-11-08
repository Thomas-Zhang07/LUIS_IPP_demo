# Cleanup progress tracker

Working rule: one logical change per commit; each commit must pass
`python -m compileall -q .` and `pytest -q`. This file is ticked in the same
commit that closes an item and is deleted when the cleanup is complete.

> Baseline exception: `python -m compileall` is red at commit 1 because of the
> pre-existing `onoff.py` encoding failure (D1). It turned green at commit 2.
## Commit sequence

- [x] 1. `chore: bootstrap test tooling, changelog and tracker`
- [x] 2. `fix(encoding): convert GBK sources to UTF-8 and repair onoff.py`
- [x] 3. `feat(protocol): shared frame builders and serial transport`
- [x] 4. `refactor(serial-utils): unify port selection`
- [x] 5. `refactor(all-on): use shared protocol layer`
- [x] 6. `feat(acoustics): shared physics and quantization`
- [x] 7. `fix(focus): shared physics; dry-run never prompts for a port`
- [x] 8. `feat(trajectory): deadline-based streaming and generators`
- [x] 9. `refactor(line-circle): adopt shared trajectory runner`
- [x] 10. `fix(onoff-patterns): open serial once, add dry-run, warn on wire time`
- [x] 11. `refactor(onoff-demo): adopt shared toggle runner; add dry-run`
- [x] 12. `refactor(onoff-steer): shared toggle runner; carrier flags`
- [x] 13. `feat(array-factor): extract analysis module; remove dead beta code`
- [x] 14. `refactor(phase-steer): shared acoustics; carrier flags`
- [x] 15. `refactor(phase-steer-2d): thin CLI over shared modules`
- [x] 16. `refactor(mode-control): main guard, --port/--dry-run`
- [x] 17. `docs: rewrite README from actual behavior`
- [ ] 18. `chore: final verification, changelog and tracker cleanup`
- [ ] 19. AGY reviewer (`consult_agy`) + independent code-review agent pass; address findings

## Defects

- [x] D14 — no `requirements.txt`, no tests (commit 1)
- [x] D1 — `onoff.py` GBK without coding declaration → SyntaxError (commit 2)
- [x] D2 — GBK sources with declarations (`single_point_hover.py`, `single_point_vortex.py`) (commit 2)
- [x] D3 — README contains 8 U+FFFD characters (commit 17)
- [x] D4 — `--dry-run` prompts for / requires a port (hover, vortex) (commit 7)
- [x] D5 — port reopened per half-cycle/bit (onoff, bit_pattern_onoff) (commit 10)
- [x] D6 — wavelength inconsistency (commit 6)
- [x] D7 — circle default 2.0 s vs docs 5 s (commit 9)
- [x] D8 — README wrong about radius and Ctrl+C all-off behavior (commit 17)
- [x] D9 — dead beta-optimization code (168 lines) (commit 13)
- [x] D10 — duplicated port selection / serial config / frame builders / focus math (commits 3–16)
- [x] D11 — `assert` used for validation (commit 7)
- [x] D12 — drifting `sleep(interval)` in line trajectory (commit 9)
- [x] D13 — `mode_control.py` runs on import, hardcoded COM9 (commit 16)
- [x] D15 — README omits scripts and has stale samples (commit 17)

## Verification checkpoints

- [x] Phase 1 (commits 1–6): compile + tests green; .py encoding audit clean
      (README U+FFFD remains until commit 17)
- [x] Phase 2 (commits 7–12): hardware-free dry-runs for all streaming scripts
      (11/11 exit 0)
- [ ] Phase 3 (commits 13–18): full test suite + dry-run matrix
- [ ] Post-cleanup: independent code-review agent pass, findings triaged/fixed

## Review follow-ups (AGY + code-review)

- [x] R1 — quantizer snapped a 2π boundary to bin 15 instead of bin 0
- [x] R2 — ON/OFF scripts reject UART-achievable rates below 20 Hz
- [x] R3 — headless runs crash on `EOFError` instead of a clean message
- [ ] R4 — rotation deadlines anchored per-frame rather than to the epoch
- [ ] R5 — document focus/steering phase-sign and vortex-center caveats
- [ ] R6 — triage the code-review skill findings
