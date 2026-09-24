"""napari widgets for biopb.

The root imports no widget, so napari's manifest (``napari.yaml``) is what loads
them, and the GUI-free writers import without Qt.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
