"""The widgets' settings: defaults, overridden by a JSON file the widgets write.

The file is ``napari-widget.json`` in biopb's config directory,
``$BIOPB_CONFIG_HOME/biopb`` (default ``~/.config/biopb`` on every platform),
beside biopb's own config files. Only the keys a user changed need to be in it;
anything missing, or of the wrong type, reads as its default.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, dict[str, Any]] = {
    "widget": {
        # The ProcessImage / ObjectDetection server the widgets target.
        "server_url": "localhost:50051",
        "is_3d": False,
    },
    "detection": {
        "min_score": 0.4,
        "size_hint": 32.0,
        "nms": "Off",
        "z_aspect_ratio": 1.0,
    },
    # Tiling for large-image detection, per axis; stride < size overlaps tiles.
    "grid": {
        "size_2d": [4096, 4096],
        "stride_2d": [4000, 4000],
        "size_3d": [64, 512, 512],
        "stride_3d": [48, 480, 480],
    },
    # Per-call gRPC timeouts, seconds.
    "timeout": {
        "health_check": 5.0,
        "get_op_names": 10.0,
        "detection_2d": 15,
        "detection_3d": 300,
        "process_image": 300,
    },
    "grpc": {
        "max_message_size_mb": 512,
        "max_concurrent_calls": 4,
    },
    # A single chunk above these sizes warns, then raises MemoryError.
    "memory": {
        "warn_threshold_mb": 500,
        "error_threshold_mb": 2000,
    },
}


def settings_path() -> Path:
    # biopb's rule: an absolute $BIOPB_CONFIG_HOME, else ~/.config; XDG_* is
    # not read. biopb refuses a relative value; a widget falls back instead.
    raw = os.environ.get("BIOPB_CONFIG_HOME")
    if raw and not os.path.isabs(raw):
        logger.warning("Ignoring relative BIOPB_CONFIG_HOME=%r", raw)
        raw = None
    base = Path(raw) if raw else Path.home() / ".config"
    return base / "biopb" / "napari-widget.json"


def _type_ok(value, default) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return type(value) is type(default)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float))
    if isinstance(default, list):
        return isinstance(value, list) and len(value) == len(default)
    return isinstance(value, type(default))


class Settings:
    """Lazy read of the file over :data:`DEFAULTS`; writes go back to it."""

    def __init__(self) -> None:
        self._data: dict | None = None
        self._lock = threading.RLock()

    def _load(self) -> dict:
        if self._data is None:
            data = copy.deepcopy(DEFAULTS)
            try:
                stored = json.loads(settings_path().read_text())
            except FileNotFoundError:
                stored = {}
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable %s: %s", settings_path(), exc)
                stored = {}
            for section, keys in data.items():
                given = stored.get(section) if isinstance(stored, dict) else None
                if not isinstance(given, dict):
                    continue
                for key, default in keys.items():
                    if key in given and _type_ok(given[key], default):
                        keys[key] = given[key]
            self._data = data
        return self._data

    def get(self, path: str):
        """The value at ``"section.key"``."""
        section, key = path.split(".")
        with self._lock:
            return self._load()[section][key]

    def set(self, path: str, value, *, persist: bool = True) -> None:
        section, key = path.split(".")
        with self._lock:
            self._load()[section][key] = value
            if persist:
                self.save()

    def save(self) -> None:
        with self._lock:
            path = settings_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump(self._load(), f, indent=2)
                os.replace(tmp, path)
            except OSError as exc:
                logger.warning("Could not save %s: %s", path, exc)

    def reload(self) -> None:
        with self._lock:
            self._data = None


SETTINGS = Settings()


def get_grid_params(is_3d: bool):
    """``(grid_size, stride)`` as int arrays for 2-D or 3-D detection."""
    import numpy as np

    suffix = "3d" if is_3d else "2d"
    return (
        np.array(SETTINGS.get(f"grid.size_{suffix}"), dtype=int),
        np.array(SETTINGS.get(f"grid.stride_{suffix}"), dtype=int),
    )
