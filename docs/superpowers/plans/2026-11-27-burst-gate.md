# Burst / gate host tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host CLI that emits isolated, low-output acoustic bursts on both boards by gating the latched phase banks, plus a channel-frame lint.

**Architecture:** A pure `build_burst_plan` produces ordered `(payload_a, payload_b, delay)` steps for two gate modes: `swap` (FE+FD bank toggle, ~0.5 ms edges) and `frame` (full ON/OFF frames, ~67 ms minimum). A thin `run_burst` writes each step through `dual_board.BoardLink.send` and sleeps. ON frames use a low amplitude code by default.

**Tech Stack:** Python 3.10+, NumPy, pySerial, pytest.

**Spec:** `docs/superpowers/specs/2026-11-27-burst-gate-design.md`

## Global Constraints

- ON frame default amplitude code `2` (width 1, ≈−14 dB); code >2 requires `--allow-high-amplitude`.
- OFF frame is `0xFE + 256 × 0x10 + 0xFD`; ON frame is `0xFE + 256 × (code<<4) + 0xFD`.
- A swap needs a rising edge: every toggle payload is exactly `bytes([0xFE, 0xFD])`.
- Frame select bytes: A = `0xC0`, B = `0xC1`.
- No RTL change; no shared-bus support in this tool (separate ports only).
- Every full channel frame must pass `protocol.validate_channel_frame`.

## Review Focus

- **Accidental full-power output:** a missing `--allow-high-amplitude` check must not let code 15 through. Pinned in Task 3's CLI test.
- **Consecutive FD (no rising edge):** a toggle without a preceding FE silently fails to swap. Pinned in Task 2's `FE`-before-`FD` test.
- **Frame not fully sent before a toggle:** writing a toggle before the frame drains can corrupt it. `BoardLink.send` flushes; pinned in Task 3's runner test ordering.
- **Wrong board select:** an ON/OFF frame missing or swapping select bytes addresses the wrong board. Pinned in Task 2's select-byte test.
- **Command byte as channel data:** a hand-built uniform frame with `0xFD`/`0x18` is unsafe. Pinned in Task 1's lint tests.

---

### Task 1: Channel-frame lint and `build_uniform_frame` hardening

**Files:**
- Modify: `demo/protocol.py`
- Test: `demo/tests/test_protocol.py`

**Interfaces:**
- Consumes: `DECODER_COMMAND_BYTES`, `OFF_ALIAS_BYTE`, `HEADER`, `FOOTER`, `FRAME_SIZE`, `N_EMITTERS`
- Produces: `protocol.validate_channel_frame(frame: bytes) -> bytes`

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_channel_frame_rejects_decoder_commands():
    for byte in sorted(protocol.DECODER_COMMAND_BYTES) + [protocol.OFF_ALIAS_BYTE]:
        frame = bytes([protocol.HEADER]) + bytes([byte]) * N_EMITTERS + bytes([protocol.FOOTER])
        with pytest.raises(ValueError):
            protocol.validate_channel_frame(frame)


def test_validate_channel_frame_accepts_on_off_and_amplitude():
    for frame in (protocol.build_on_frame(), protocol.build_off_frame(),
                  protocol.build_amplitude_frame(_values(), np.full(N_EMITTERS, 7, int))):
        assert protocol.validate_channel_frame(frame) == frame


def test_validate_channel_frame_rejects_bad_layout():
    with pytest.raises(ValueError):
        protocol.validate_channel_frame(b"\xfe" + bytes(N_EMITTERS))
    with pytest.raises(ValueError):
        protocol.validate_channel_frame(b"\x00" + bytes(N_EMITTERS) + b"\xfd")


def test_build_uniform_frame_rejects_decoder_commands():
    with pytest.raises(ValueError):
        protocol.build_uniform_frame(0xFD)
    assert protocol.build_uniform_frame(protocol.OFF_VALUE)[1:-1] == bytes([0x10]) * N_EMITTERS
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_protocol.py -k "validate_channel_frame or uniform_frame_rejects" -v`
Expected: FAIL (no `validate_channel_frame`; `build_uniform_frame(0xFD)` does not raise).

- [ ] **Step 3: Implement**

`validate_channel_frame(frame)` checks length `FRAME_SIZE`, header, footer, then raises on any interior byte in `DECODER_COMMAND_BYTES` or equal to `OFF_ALIAS_BYTE`; returns the frame. `build_uniform_frame` calls it on the built frame before returning.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add protocol.py tests/test_protocol.py
git commit -m "feat: validate channel frames before sending"
```

---

### Task 2: Burst plan builder

**Files:**
- Create: `demo/burst_sender.py`
- Test: `demo/tests/test_burst_sender.py`

**Interfaces:**
- Produces:
  - `burst_sender.GATE_SWAP = "swap"`, `burst_sender.GATE_FRAME = "frame"`
  - `burst_sender.TOGGLE = bytes([0xFE, 0xFD])`
  - `@dataclass(frozen=True) class BurstStep: payload_a: bytes; payload_b: bytes; delay_s: float`
  - `build_burst_plan(gate: str, amplitude_code: int, repeats: int, dwell_s: float, gap_s: float, allow_high_amplitude: bool = False) -> list[BurstStep]`

- [ ] **Step 1: Write the failing tests**

```python
import burst_sender as bs

def _plan(gate):
    return bs.build_burst_plan(gate, amplitude_code=2, repeats=2, dwell_s=0.004, gap_s=0.1)

def test_frame_plan_primes_banks_then_alternates():
    steps = _plan(bs.GATE_FRAME)
    assert [s.payload_a[0] for s in steps] == [0xC0] * 6
    assert steps[0].payload_a == steps[1].payload_a          # OFF, OFF
    assert steps[2].payload_a[1:] != steps[0].payload_a[1:]  # ON differs from OFF
    assert steps[3].payload_a[1:] == steps[0].payload_a[1:]  # OFF after ON
    assert steps[2].delay_s == 0.004 and steps[3].delay_s == 0.1

def test_swap_plan_pairs_fe_before_fd():
    steps = _plan(bs.GATE_SWAP)
    toggles = [s for s in steps if s.payload_a == bs.TOGGLE]
    assert len(toggles) == 3                                 # off, on, off for 2 repeats
    assert all(s.payload_a == s.payload_b == bytes([0xFE, 0xFD]) for s in toggles)

def test_on_frame_uses_low_amplitude_and_is_safe():
    steps = _plan(bs.GATE_FRAME)
    on = steps[2].payload_a[1:]
    protocol.validate_channel_frame(on)
    assert set(on[1:-1]) == {0x20}                           # code 2, phase 0

def test_plan_rejects_high_code_without_ack():
    with pytest.raises(ValueError):
        bs.build_burst_plan(bs.GATE_SWAP, amplitude_code=15, repeats=1, dwell_s=0.004, gap_s=0.1)
    plan = bs.build_burst_plan(
        bs.GATE_SWAP, amplitude_code=15, repeats=1, dwell_s=0.004, gap_s=0.1,
        allow_high_amplitude=True,
    )
    assert plan
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_burst_sender.py -v`
Expected: FAIL (`burst_sender` does not exist).

- [ ] **Step 3: Implement**

`build_on_frame(code)` uses `protocol.build_amplitude_frame(np.zeros(256, int), np.full(256, code, int))`. Reject codes >2 and `OFF_VALUE`. `build_burst_plan` primes with two OFF frames (A/B select), then:
- `frame`: per repeat, ON then OFF step, delays `dwell_s` and `gap_s`.
- `swap`: one ON frame step (delay `dwell_s`), then per repeat an OFF toggle (delay `gap_s`) and, except after the last, an ON toggle (delay `dwell_s`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_burst_sender.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add burst_sender.py tests/test_burst_sender.py
git commit -m "feat: add burst gate plan builder"
```

---

### Task 3: Runner and CLI

**Files:**
- Modify: `demo/burst_sender.py`
- Test: `demo/tests/test_burst_sender.py`

**Interfaces:**
- Produces: `run_burst(link, steps, sleep=time.sleep) -> None`, `build_parser()`, `main(argv=None) -> int`
- Consumes: `dual_board.BoardLink`, `protocol.wire_time_s`

- [ ] **Step 1: Write the failing tests**

```python
class FakeLink:
    def __init__(self): self.calls = []
    def send(self, a, b): self.calls.append((bytes(a), bytes(b)))

def test_run_burst_writes_steps_and_sleeps():
    link, sleeps = FakeLink(), []
    steps = bs.build_burst_plan(bs.GATE_SWAP, 2, 2, 0.004, 0.1)
    bs.run_burst(link, steps, sleep=sleeps.append)
    assert [(a[0], b[0]) for a, b in link.calls] == [(0xC0, 0xC1)] * 3 + [(0xFE, 0xFE)] * 3
    assert sleeps == [0.0, 0.0, 0.004, 0.1, 0.004, 0.1]

def test_cli_rejects_high_amplitude_without_ack():
    assert bs.main(["--dry-run", "--amplitude-code", "15"]) == 2
    assert bs.main(["--dry-run", "--amplitude-code", "15", "--allow-high-amplitude"]) == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_burst_sender.py -k "run_burst or cli" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`run_burst` iterates steps, calls `link.send(payload_a, payload_b)`, then `sleep(delay_s)`. CLI flags: `--gate`, `--amplitude-code`, `--allow-high-amplitude`, `--dwell-ms`, `--gap-ms`, `--repeats`, `--port-a`, `--port-b`, `--trim`, `--baud`, `--dry-run`, `--list-ports`. Dry run prints the gate, frame counts, low-output code and a timing caveat without opening a port. Live run requires both ports, optionally sends trim to B via `BoardLink.send_trim`, then `run_burst`.

- [ ] **Step 4: Run tests and dry run**

Run: `python -m pytest tests/test_burst_sender.py -v && python burst_sender.py --dry-run`
Expected: PASS; dry run prints the plan and exits 0.

- [ ] **Step 5: Commit**

```bash
git add burst_sender.py tests/test_burst_sender.py
git commit -m "feat: add gated burst sender CLI"
```

---

### Task 4: Documentation

**Files:**
- Modify: `demo/README.md`, `Devin_bot/.../pcb-final-checks/host-cli.md` (untracked workspace; edit only)

- [ ] **Step 1: Add the tool to `README.md`** with the two gate modes, low-output default and timing caveat.
- [ ] **Step 2: Add a burst note to the bench CLI card** describing the sequence, the FE-before-FD requirement and the start-reference limitation.
- [ ] **Step 3: Commit the demo README change**

```bash
git add README.md
git commit -m "docs: document the burst sender"
```
