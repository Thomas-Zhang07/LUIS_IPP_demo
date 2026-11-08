"""Shared pytest fixtures and fakes.

No hardware is required: serial is faked and time is simulated so scheduling
behavior can be asserted deterministically.
"""

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))


class FakeSerial:
    """Minimal pyserial stand-in that records everything sent to it."""

    def __init__(self, *args, **kwargs):
        self.port = args[0] if args else kwargs.get("port")
        self.settings = kwargs
        self.writes = []
        self.dtr = None
        self.rts = None
        self.is_open = True
        self.closed = False

    def write(self, data):
        if self.closed:
            raise RuntimeError("write on closed port")
        self.writes.append(bytes(data))
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True
        self.is_open = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeSerialFactory:
    """Callable factory that creates and records FakeSerial instances."""

    def __init__(self):
        self.instances = []

    def __call__(self, *args, **kwargs):
        instance = FakeSerial(*args, **kwargs)
        self.instances.append(instance)
        return instance

    @property
    def last(self):
        return self.instances[-1]


class FakeClock:
    """Deterministic monotonic clock whose sleep advances time."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        if seconds > 0:
            self.now += seconds

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_serial_factory():
    return FakeSerialFactory()


@pytest.fixture
def fake_clock():
    return FakeClock()
