"""The widgets' settings: defaults under a partial, possibly bad, file."""

import json

from biopb_napari_widget import _settings
from biopb_napari_widget._settings import SETTINGS, get_grid_params


def _write(obj):
    _settings.settings_path().write_text(json.dumps(obj))
    SETTINGS.reload()


def test_no_file_reads_the_defaults():
    assert SETTINGS.get("widget.server_url") == "localhost:50051"
    assert SETTINGS.get("grid.size_3d") == [64, 512, 512]


def test_a_stored_key_overrides_and_the_rest_default():
    _write({"detection": {"min_score": 0.7}})
    assert SETTINGS.get("detection.min_score") == 0.7
    assert SETTINGS.get("detection.nms") == "Off"


def test_a_value_of_the_wrong_type_reads_as_its_default():
    _write({"widget": {"is_3d": "yes"}, "grid": {"size_2d": [1, 2, 3]}})
    assert SETTINGS.get("widget.is_3d") is False
    assert SETTINGS.get("grid.size_2d") == [4096, 4096]


def test_an_unreadable_file_reads_as_the_defaults():
    _settings.settings_path().write_text("{not json")
    SETTINGS.reload()
    assert SETTINGS.get("memory.warn_threshold_mb") == 500


def test_set_writes_through_and_survives_a_reload():
    SETTINGS.set("widget.server_url", "host:1")
    SETTINGS.reload()
    assert SETTINGS.get("widget.server_url") == "host:1"
    stored = json.loads(_settings.settings_path().read_text())
    assert stored["widget"]["server_url"] == "host:1"


def test_a_batched_set_is_not_written_until_save():
    SETTINGS.set("widget.is_3d", True, persist=False)
    assert not _settings.settings_path().exists()
    SETTINGS.save()
    assert _settings.settings_path().exists()


def test_grid_params():
    size, stride = get_grid_params(True)
    assert size.tolist() == [64, 512, 512]
    assert stride.dtype.kind == "i"
