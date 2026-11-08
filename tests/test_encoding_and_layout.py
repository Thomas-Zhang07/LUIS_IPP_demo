"""Encoding and hardware-layout regression tests.

These guard D1/D2: every source must be UTF-8, legacy declarations must not
return, and the hardware transmit order must remain untouched.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_SOURCES = sorted(ROOT.glob("*.py"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_all_python_sources_are_utf8_without_bom():
    for path in PY_SOURCES:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} has a BOM"
        raw.decode("utf-8")


def test_no_legacy_coding_declaration():
    for path in PY_SOURCES:
        lowered = _read(path).lower()
        assert "coding: gbk" not in lowered, path.name
        assert "coding: gb2312" not in lowered, path.name


def test_no_replacement_characters_in_python_sources():
    for path in PY_SOURCES:
        assert "�" not in _read(path), f"{path.name} contains U+FFFD"


def test_onoff_source_compiles():
    # D1 regression: this file used to raise SyntaxError on import.
    compile(_read(ROOT / "onoff.py"), "onoff.py", "exec")


def test_chinese_strings_preserved():
    assert "无法打开串口" in _read(ROOT / "transducer_all_on.py")

    vortex = _read(ROOT / "single_point_vortex.py")
    assert "逆时针" in vortex
    assert "顺时针" in vortex


def test_emitters_first_group_is_unchanged():
    sys.path.insert(0, str(ROOT))
    import emitters

    assert emitters.EMITTERS_ORDER[:8] == [0, 32, 48, 17, 1, 49, 33, 16]
    assert len(emitters.EMITTERS_ORDER) == 256
    assert sorted(emitters.EMITTERS_ORDER) == list(range(256))
