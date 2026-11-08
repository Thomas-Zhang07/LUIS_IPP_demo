"""Tests for the shared serial port selection module."""

from types import SimpleNamespace

import pytest

import serial_utils


def _ports():
    return [
        SimpleNamespace(device="COM3", description="USB Serial"),
        SimpleNamespace(device="/dev/rfcomm0", description=None),
    ]


def test_select_by_index(monkeypatch):
    monkeypatch.setattr(serial_utils, "list_available_ports", _ports)
    assert serial_utils.select_serial_port(input_fn=lambda _: "1", output=lambda _: None) == "/dev/rfcomm0"


def test_select_by_device_name_case_insensitive(monkeypatch):
    monkeypatch.setattr(serial_utils, "list_available_ports", _ports)
    assert serial_utils.select_serial_port(input_fn=lambda _: "com3", output=lambda _: None) == "COM3"


def test_invalid_input_reprompts(monkeypatch):
    monkeypatch.setattr(serial_utils, "list_available_ports", _ports)
    replies = iter(["", "99", "nope", "0"])
    messages = []
    selected = serial_utils.select_serial_port(
        input_fn=lambda _: next(replies), output=messages.append
    )
    assert selected == "COM3"
    assert any("cannot be empty" in message for message in messages)
    assert any("out of range" in message for message in messages)
    assert any("was not found" in message for message in messages)


def test_no_ports_raises(monkeypatch):
    monkeypatch.setattr(serial_utils, "list_available_ports", lambda: [])
    with pytest.raises(RuntimeError):
        serial_utils.select_serial_port(input_fn=lambda _: "0", output=lambda _: None)


def test_print_available_ports_reports_presence(monkeypatch):
    monkeypatch.setattr(serial_utils, "list_available_ports", _ports)
    messages = []
    assert serial_utils.print_available_ports(output=messages.append) is True
    assert any("COM3" in message for message in messages)

    monkeypatch.setattr(serial_utils, "list_available_ports", lambda: [])
    assert serial_utils.print_available_ports(output=messages.append) is False


def test_resolve_dry_run_never_prompts(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not prompt during a dry run")

    monkeypatch.setattr(serial_utils, "select_serial_port", explode)
    assert serial_utils.resolve_serial_port(None, dry_run=True) is None


def test_resolve_explicit_port_never_prompts(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not prompt when a port is supplied")

    monkeypatch.setattr(serial_utils, "select_serial_port", explode)
    assert serial_utils.resolve_serial_port("COM7", dry_run=False) == "COM7"


def test_resolve_interactive_calls_selector(monkeypatch):
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return "COM5"

    monkeypatch.setattr(serial_utils, "select_serial_port", fake_select)
    assert serial_utils.resolve_serial_port(None, dry_run=False) == "COM5"
    assert calls and calls[0]["baudrate"] == serial_utils.BAUD_RATE
