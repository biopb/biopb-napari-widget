import os
import sys

import pytest

from biopb_napari_widget.image_processing import ImageProcessingWidget


# Skip on macOS CI due to OpenGL/vispy headless issues
# On macOS without a display, vispy canvas initialization fails
@pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="OpenGL context unavailable on macOS CI headless environment",
)
def test_widget_instantiation(make_napari_viewer, request, monkeypatch):
    """Test widget instantiation."""
    # Construction would start a worker fetching ops from a server.
    monkeypatch.setattr(ImageProcessingWidget, "_fetch_ops", lambda self: None)
    viewer = make_napari_viewer(show=False)
    request.addfinalizer(viewer.close)
    my_widget = ImageProcessingWidget(viewer)

    assert my_widget


def test_widget_basic():
    """Basic test that doesn't require a viewer (runs on all platforms)."""
    # Test that the widget module can be imported
    from biopb_napari_widget.image_processing import ImageProcessingWidget

    assert ImageProcessingWidget is not None
