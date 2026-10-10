"""ROI annotations -> napari Points / Shapes layer specs."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from biopb.image import ROI, Ellipse, Point, Polygon, Polyline, Rectangle, RoiAnnotation

from biopb_napari_widget._rois import (
    MAX_REPEATED_VERTICES,
    add_roi_layers,
    roi_layer_specs,
    sets_in,
)


def desc(shape, labels):
    return SimpleNamespace(array_id="img", shape=tuple(shape), dim_labels=list(labels))


def ann(roi, *, set_name="s", label="", roi_id="", plane=None):
    return RoiAnnotation(
        roi_id=roi_id,
        array_id="img",
        set_name=set_name,
        label=label,
        roi=roi,
        plane=plane or {},
    )


def pt(x, y, **kw):
    return ann(ROI(point=Point(x=x, y=y)), **kw)


def rect(x0, y0, x1, y1, **kw):
    return ann(
        ROI(
            rectangle=Rectangle(
                top_left=Point(x=x0, y=y0), bottom_right=Point(x=x1, y=y1)
            )
        ),
        **kw,
    )


YX = desc((100, 200), ["y", "x"])
ZYX = desc((10, 100, 200), ["z", "y", "x"])
CZYX = desc((3, 10, 100, 200), ["c", "z", "y", "x"])


def one(specs):
    assert len(specs) == 1
    return specs[0]


class TestGeometry:
    def test_point_is_y_x(self):
        spec = one(roi_layer_specs([pt(5, 7)], YX))
        assert spec.kind == "points"
        assert np.allclose(spec.data, [[7, 5]])

    def test_rectangle_corners_in_y_x(self):
        spec = one(roi_layer_specs([rect(10, 20, 30, 40)], YX))
        assert spec.kind == "shapes"
        assert spec.kwargs["shape_type"] == ["rectangle"]
        assert np.allclose(spec.data[0], [[20, 10], [20, 30], [40, 30], [40, 10]])

    def test_polygon_and_polyline(self):
        pts = [Point(x=0, y=0), Point(x=4, y=0), Point(x=4, y=3)]
        specs = roi_layer_specs(
            [
                ann(ROI(polygon=Polygon(points=pts)), set_name="a"),
                ann(ROI(polyline=Polyline(points=pts, width=2.5)), set_name="b"),
            ],
            YX,
        )
        a, b = specs
        assert a.kwargs["shape_type"] == ["polygon"]
        assert b.kwargs["shape_type"] == ["path"]
        assert b.kwargs["edge_width"] == [2.5]
        assert np.allclose(a.data[0], [[0, 0], [0, 4], [3, 4]])

    def test_ellipse_rotates_about_its_centre(self):
        e = ann(
            ROI(
                ellipse=Ellipse(
                    center=Point(x=50, y=40),
                    radius=Point(x=10, y=4),
                    rotation=math.pi / 2,
                )
            )
        )
        corners = np.asarray(one(roi_layer_specs([e], YX)).data[0])
        # A quarter turn swaps the extents: 8 along x, 20 along y.
        assert np.ptp(corners[:, 1]) == pytest.approx(8)
        assert np.ptp(corners[:, 0]) == pytest.approx(20)
        assert corners.mean(axis=0) == pytest.approx([40, 50])

    def test_unrotated_ellipse_is_its_bounding_box(self):
        e = ann(ROI(ellipse=Ellipse(center=Point(x=50, y=40), radius=Point(x=10, y=4))))
        c = np.asarray(one(roi_layer_specs([e], YX)).data[0])
        assert c[:, 1].min() == 40 and c[:, 1].max() == 60
        assert c[:, 0].min() == 36 and c[:, 0].max() == 44


class TestLayers:
    def test_a_set_with_points_and_shapes_makes_two_layers(self):
        specs = roi_layer_specs([pt(1, 1), rect(0, 0, 5, 5)], YX)
        assert [(s.kind, s.name) for s in specs] == [
            ("points", "s (points)"),
            ("shapes", "s (shapes)"),
        ]

    def test_one_family_keeps_the_set_name(self):
        assert one(roi_layer_specs([pt(1, 1)], YX)).name == "s"

    def test_one_layer_per_set(self):
        specs = roi_layer_specs([pt(1, 1, set_name="a"), pt(2, 2, set_name="b")], YX)
        assert [s.name for s in specs] == ["a", "b"]

    def test_empty_set_name_is_default(self):
        assert one(roi_layer_specs([pt(1, 1, set_name="")], YX)).name == "default"

    def test_only_sets_restricts(self):
        rois = [pt(1, 1, set_name="a"), pt(2, 2, set_name="b")]
        assert [s.name for s in roi_layer_specs(rois, YX, only_sets=["b"])] == ["b"]

    def test_label_and_roi_id_are_kept_as_features(self):
        spec = one(roi_layer_specs([pt(1, 1, label="focus", roi_id="r1")], YX))
        assert list(spec.kwargs["features"]["label"]) == ["focus"]
        assert list(spec.kwargs["features"]["roi_id"]) == ["r1"]
        assert spec.kwargs["metadata"]["roi_ids"] == ["r1"]
        assert spec.kwargs["metadata"]["array_id"] == "img"

    def test_masks_and_meshes_are_skipped(self):
        empty = ann(ROI())
        assert roi_layer_specs([empty], YX) == []

    def test_sets_in_counts_in_first_seen_order(self):
        rois = [pt(1, 1, set_name="b"), pt(1, 1, set_name="a"), pt(2, 2, set_name="b")]
        assert sets_in(rois) == {"b": 2, "a": 1}


class TestPlanes:
    def test_unpinned_on_a_stack_is_a_2d_layer_napari_broadcasts(self):
        spec = one(roi_layer_specs([pt(5, 7)], ZYX))
        assert np.asarray(spec.data).shape == (1, 2)
        assert list(spec.axes) == [1, 2]

    def test_pinned_axis_becomes_a_leading_coordinate(self):
        spec = one(roi_layer_specs([pt(5, 7, plane={0: 4})], ZYX))
        assert np.allclose(spec.data, [[4, 7, 5]])
        assert list(spec.axes) == [0, 1, 2]

    def test_layer_spans_from_the_earliest_pinned_axis(self):
        spec = one(roi_layer_specs([pt(5, 7, plane={0: 2, 1: 4})], CZYX))
        assert np.allclose(spec.data, [[2, 4, 7, 5]])
        spec = one(roi_layer_specs([pt(5, 7, plane={1: 4})], CZYX))
        assert np.allclose(spec.data, [[4, 7, 5]])  # c is left to broadcast

    def test_unpinned_axis_inside_the_range_is_repeated(self):
        # c pinned, z not: the ROI must appear on all 10 z planes of c=1.
        spec = one(roi_layer_specs([pt(5, 7, plane={0: 1})], CZYX))
        data = np.asarray(spec.data)
        assert data.shape == (10, 4)
        assert set(data[:, 0]) == {1} and list(data[:, 1]) == list(range(10))
        assert len(spec.kwargs["features"]) == 10

    def test_mixed_pins_in_one_layer(self):
        rois = [pt(1, 1, plane={0: 3}), pt(2, 2)]  # one pinned, one on every plane
        data = np.asarray(one(roi_layer_specs(rois, ZYX)).data)
        assert data.shape == (1 + 10, 3)

    def test_the_repetition_is_capped(self):
        big = desc((2, MAX_REPEATED_VERTICES + 50, 100, 200), ["t", "c", "y", "x"])
        spec = one(roi_layer_specs([pt(1, 1, plane={0: 0})], big))
        assert len(spec.data) == MAX_REPEATED_VERTICES

    def test_interleaved_samples_axis_is_not_a_layer_axis(self):
        rgb = desc((10, 100, 200, 3), ["z", "y", "x", "s"])
        spec = one(roi_layer_specs([pt(5, 7, plane={0: 1})], rgb))
        assert np.allclose(spec.data, [[1, 7, 5]])
        assert list(spec.axes) == [0, 1, 2]

    def test_a_pin_on_y_x_or_out_of_range_is_ignored(self):
        spec = one(roi_layer_specs([pt(5, 7, plane={1: 5, 2: 5, 9: 1})], ZYX))
        assert np.allclose(np.asarray(spec.data), [[7, 5]])


class TestScale:
    def test_each_layer_takes_the_trailing_scale(self):
        scale = [2.0, 0.5, 0.25, 0.25]
        two = one(roi_layer_specs([pt(1, 1)], CZYX, scale=scale))
        assert two.kwargs["scale"] == [0.25, 0.25]
        three = one(roi_layer_specs([pt(1, 1, plane={1: 2})], CZYX, scale=scale))
        assert three.kwargs["scale"] == [0.5, 0.25, 0.25]

    def test_no_scale_is_none(self):
        assert "scale" not in one(roi_layer_specs([pt(1, 1)], YX)).kwargs


class TestAdd:
    def test_adds_points_and_shapes_with_axis_labels(self):
        viewer = MagicMock()
        viewer.dims.axis_labels = ("-3", "-2", "-1")
        specs = roi_layer_specs(
            [pt(1, 1, plane={0: 0}), rect(0, 0, 5, 5, plane={0: 0})], ZYX
        )
        add_roi_layers(viewer, specs)
        viewer.add_points.assert_called_once()
        viewer.add_shapes.assert_called_once()
        assert viewer.add_points.call_args.kwargs["name"] == "s (points)"

    def test_real_napari_layers_accept_the_specs(self):
        from napari.layers import Points, Shapes

        specs = roi_layer_specs(
            [
                pt(1, 2, plane={0: 3}, label="a", roi_id="r"),
                rect(0, 0, 5, 5, plane={0: 3}),
            ],
            ZYX,
            scale=[1.0, 0.5, 0.5],
        )
        points, shapes = specs
        kw = {k: v for k, v in points.kwargs.items() if k != "metadata"}
        layer = Points(np.asarray(points.data, dtype=float), **kw)
        assert layer.ndim == 3 and list(layer.features["roi_id"]) == ["r"]
        kw = {k: v for k, v in shapes.kwargs.items() if k != "metadata"}
        layer = Shapes([np.asarray(d, dtype=float) for d in shapes.data], **kw)
        assert layer.ndim == 3 and layer.nshapes == 1
