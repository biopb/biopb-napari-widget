import os

import pytest

from biopb_napari_widget import _settings

# Headless by default; set QT_QPA_PLATFORM to watch the widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Every test reads the defaults and writes nowhere real."""
    monkeypatch.setattr(_settings, "settings_path", lambda: tmp_path / "settings.json")
    _settings.SETTINGS.reload()
    yield
    _settings.SETTINGS.reload()
