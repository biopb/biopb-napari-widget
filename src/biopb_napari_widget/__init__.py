"""napari widgets for biopb.

What another package may import: the Tensor Browser, ``add_tensor_layer`` (the
layer pipeline the browser uses) and ``wrap_levels``. They load on first use, so
importing the package pulls in neither Qt nor napari.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

_EXPORTS = {
    "TensorBrowserWidget": "tensor_browser",
    "add_tensor_layer": "_tensor_utils",
    "wrap_levels": "_viewer_compute",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
