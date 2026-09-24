"""What biopb-mcp imports from this package."""

import subprocess
import sys

import biopb_napari_widget


def test_the_exports_resolve():
    from biopb_napari_widget._tensor_utils import add_tensor_layer
    from biopb_napari_widget._viewer_compute import wrap_levels
    from biopb_napari_widget.tensor_browser import TensorBrowserWidget

    assert biopb_napari_widget.TensorBrowserWidget is TensorBrowserWidget
    assert biopb_napari_widget.add_tensor_layer is add_tensor_layer
    assert biopb_napari_widget.wrap_levels is wrap_levels


def test_importing_the_package_loads_no_qt():
    code = "import sys, biopb_napari_widget; print('qtpy' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
