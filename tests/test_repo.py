"""Repository-level checks: tooling exists and is wired correctly."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_lists_core_dependencies():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("numpy", "pyserial", "pytest"):
        assert package in text, f"{package} missing from requirements.txt"


def test_pytest_config_points_at_tests():
    config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "testpaths = tests" in config
    assert "pythonpath = ." in config


def test_emitters_module_imports():
    import emitters

    assert emitters.N_EMITTERS == 256
    assert len(emitters.EMITTERS_ORDER) == 256
