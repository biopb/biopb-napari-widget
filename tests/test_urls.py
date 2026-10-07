"""``is_local_url``: the drop gate's test for a same-machine server."""

import pytest

from biopb_napari_widget._urls import is_local_url


@pytest.mark.parametrize(
    "url",
    [
        "grpc://localhost:8815",
        "grpcs://LOCALHOST:8815",
        "grpc://127.0.0.1:8815",
        "grpc://[::1]:8815",
        "http://localhost/x",
        "/just/a/path",  # no host
    ],
)
def test_local(url):
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "grpc://lab:8815",
        "grpcs://mantis-060:8815",
        "grpc://10.0.0.5:8815",
        "grpc://127.0.0.2:8815",  # loopback, but not one of the recognised hosts
        "grpc://[::1",  # unparseable
    ],
)
def test_not_local(url):
    assert not is_local_url(url)
