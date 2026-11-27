"""Hardware-free tests for the gated burst plan builder and runner."""

import pytest

import burst_sender as bs
import protocol


def _plan(gate):
    return bs.build_burst_plan(
        gate, amplitude_code=2, repeats=2, dwell_s=0.004, gap_s=0.1
    )


def test_frame_plan_primes_banks_then_alternates():
    steps = _plan(bs.GATE_FRAME)
    assert [s.payload_a[0] for s in steps] == [0xC0] * 6
    assert steps[0].payload_a == steps[1].payload_a  # OFF, OFF
    assert steps[2].payload_a[1:] != steps[0].payload_a[1:]  # ON differs
    assert steps[3].payload_a[1:] == steps[0].payload_a[1:]  # OFF after ON
    assert steps[2].delay_s == 0.004 and steps[3].delay_s == 0.1


def test_swap_plan_pairs_fe_before_fd():
    steps = _plan(bs.GATE_SWAP)
    toggles = [s for s in steps if s.payload_a == bs.TOGGLE]
    assert len(toggles) == 3  # off, on, off for two repeats
    assert all(s.payload_a == s.payload_b == bytes([0xFE, 0xFD]) for s in toggles)


def test_on_frame_uses_low_amplitude_and_is_safe():
    steps = _plan(bs.GATE_FRAME)
    on = steps[2].payload_a[1:]
    protocol.validate_channel_frame(on)
    assert set(on[1:-1]) == {0x20}  # code 2, phase 0


def test_plan_rejects_high_code_without_ack():
    with pytest.raises(ValueError):
        bs.build_burst_plan(
            bs.GATE_SWAP, amplitude_code=15, repeats=1, dwell_s=0.004, gap_s=0.1
        )
    plan = bs.build_burst_plan(
        bs.GATE_SWAP,
        amplitude_code=15,
        repeats=1,
        dwell_s=0.004,
        gap_s=0.1,
        allow_high_amplitude=True,
    )
    assert plan


class FakeLink:
    def __init__(self):
        self.calls = []

    def send(self, payload_a, payload_b):
        self.calls.append((bytes(payload_a), bytes(payload_b)))


def test_run_burst_writes_steps_and_sleeps():
    link, sleeps = FakeLink(), []
    steps = bs.build_burst_plan(bs.GATE_SWAP, 2, 2, 0.004, 0.1)
    bs.run_burst(link, steps, sleep=sleeps.append)
    assert [(a[0], b[0]) for a, b in link.calls] == (
        [(0xC0, 0xC1)] * 3 + [(0xFE, 0xFE)] * 3
    )
    assert sleeps == [0.0, 0.0, 0.004, 0.1, 0.004, 0.1]


def test_cli_rejects_high_amplitude_without_ack():
    assert bs.main(["--dry-run", "--amplitude-code", "15"]) == 2
    assert (
        bs.main(
            ["--dry-run", "--amplitude-code", "15", "--allow-high-amplitude"]
        )
        == 0
    )
