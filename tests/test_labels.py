"""``split_label_array_id``: the widget's own reading of a label set's path."""

import pytest

from biopb_napari_widget._labels import LabelAddress, split_label_array_id


@pytest.mark.parametrize(
    "array_id, expected",
    [
        ("src0/@labels/nuclei", LabelAddress("src0", "nuclei", None)),
        ("src0/Image:0/@labels/nuclei", LabelAddress("src0/Image:0", "nuclei", None)),
        ("src0/@labels/nuclei/1", LabelAddress("src0", "nuclei", "1")),
        ("src0/@labels/nuclei/a/b", LabelAddress("src0", "nuclei", "a/b")),
        # the last @labels segment with a name after it wins
        ("s/@labels/x/@labels/y", LabelAddress("s/@labels/x", "y", None)),
    ],
)
def test_a_set_is_taken_apart(array_id, expected):
    assert split_label_array_id(array_id) == expected


@pytest.mark.parametrize(
    "array_id",
    [
        "src0",
        "src0/Image:0",
        "src0/labels/nuclei",  # an OME-Zarr group, not the wire marker
        "src0/@labels",  # no name after the marker
        "src0/@labels/",
        "@labels/nuclei",  # a source called "@labels" is a source
        "",
    ],
)
def test_anything_else_names_no_set(array_id):
    assert split_label_array_id(array_id) is None
