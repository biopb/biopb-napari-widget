"""A tensor's ROI annotations as napari ``Points`` and ``Shapes`` layers.

The server stores annotations per tensor, grouped by ``set_name`` (see
``biopb.image.RoiAnnotation``). Geometry is level-0 pixel coordinates in the
tensor's own Y/X axes, and a sparse ``plane`` pin fixes an annotation to one
index of the *other* axes -- an axis absent from the pin means "every index", so
one ROI follows a z-stack.

napari layers have one geometry family each, and an N-D layer shows a vertex only
on the plane its leading coordinates name. So per set:

- points and shapes go to separate layers (``Points`` / ``Shapes``), named after
  the set and suffixed only when a set holds both;
- the layer's axes run from the earliest axis any of its ROIs pins to the end
  (excluding an interleaved samples axis). Axes before that are left out and
  napari broadcasts the layer over them, so an unpinned stack costs nothing;
- an ROI that is unpinned on an axis *inside* that range is repeated along it,
  because napari has no "every index".

Mask and mesh arms are not stored as annotations and are skipped.
"""

from __future__ import annotations

import itertools
import logging
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ._tensor_utils import (
    _apply_axis_labels,
    _resolve_axes,
    build_layer_scale,
    canonical_dim_labels,
)

logger = logging.getLogger(__name__)

#: Most copies of one layer's vertices made to fill an unpinned axis. Past it the
#: layer keeps what fits and says so.
MAX_REPEATED_VERTICES = 200_000

_DEFAULT_SET = "default"

#: Edge colours cycled over the sets, so two sets on one image are told apart.
_EDGE_COLORS = ("yellow", "cyan", "magenta", "lime", "orange", "white")


@dataclass
class RoiLayerSpec:
    """One napari layer to add: everything but the viewer."""

    kind: str  # "points" | "shapes"
    name: str
    data: Any
    kwargs: Dict[str, Any] = field(default_factory=dict)
    #: The layer's axis names (trailing slice of the image's), or ``None``.
    dim_labels: List[str] | None = None


def _arm(roi) -> str | None:
    try:
        return roi.roi.WhichOneof("shape")
    except (AttributeError, ValueError):
        return None


def _geometry(roi) -> tuple[str, List[tuple], float] | None:
    """``(napari shape_type, (y, x) vertices, edge width)`` for a 2-D shape arm,
    else ``None``."""
    arm = _arm(roi)
    g = roi.roi
    if arm == "rectangle":
        r = g.rectangle
        x0, y0 = r.top_left.x, r.top_left.y
        x1, y1 = r.bottom_right.x, r.bottom_right.y
        return "rectangle", [(y0, x0), (y0, x1), (y1, x1), (y1, x0)], 1.0
    if arm == "ellipse":
        e = g.ellipse
        # In-plane rotation about the centre, radians, from +x toward +y.
        c, s = math.cos(e.rotation), math.sin(e.rotation)
        rx, ry = e.radius.x, e.radius.y
        corners = [
            (e.center.y + dx * s + dy * c, e.center.x + dx * c - dy * s)
            for dx, dy in ((-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry))
        ]
        return "ellipse", corners, 1.0
    if arm == "polygon":
        return "polygon", [(p.y, p.x) for p in g.polygon.points], 1.0
    if arm == "polyline":
        return (
            "path",
            [(p.y, p.x) for p in g.polyline.points],
            float(g.polyline.width) or 1.0,
        )
    return None


def _pin(roi, n_lead: int) -> Dict[int, int]:
    """The ROI's pin on the leading (non-Y/X) axes; anything else is dropped."""
    return {int(a): int(i) for a, i in roi.plane.items() if 0 <= int(a) < n_lead}


def _prefixes(pin: Dict[int, int], lead: int, n_lead: int, shape):
    """Leading coordinates for an ROI: its pinned index on each axis from *lead*,
    every index where it is unpinned. Lazy, so a huge axis costs only what is
    taken."""
    return itertools.product(
        *([pin[a]] if a in pin else range(int(shape[a])) for a in range(lead, n_lead))
    )


def _edge_color(set_name: str) -> str:
    """Stable per set name, so the same set looks the same however it is loaded
    and two sets loaded one after the other are told apart."""
    return _EDGE_COLORS[zlib.crc32(set_name.encode()) % len(_EDGE_COLORS)]


def _build_layer(kind, name, group, tensor_desc, shape, n_axes, scale, labels, color):
    """One layer from *group*, a list of ``(roi, geometry)`` (geometry is ``None``
    for a point)."""
    import pandas as pd

    n_lead = n_axes - 2
    pins = [_pin(roi, n_lead) for roi, _ in group]
    # The layer starts at the earliest axis any of its ROIs pins; axes before it
    # are left out and napari broadcasts the layer over them.
    lead = min((min(p) for p in pins if p), default=n_lead)

    data, shape_types, widths, feats = [], [], [], []
    for (roi, geom), pin in zip(group, pins, strict=True):
        room = MAX_REPEATED_VERTICES - len(data)
        for prefix in itertools.islice(_prefixes(pin, lead, n_lead, shape), room):
            if kind == "points":
                data.append(prefix + (roi.roi.point.y, roi.roi.point.x))
            else:
                shape_type, verts, width = geom
                data.append([prefix + v for v in verts])
                shape_types.append(shape_type)
                widths.append(width)
            feats.append((roi.label, roi.roi_id))
    if len(data) >= MAX_REPEATED_VERTICES:
        logger.warning(
            "ROI set %s: unpinned annotations repeated along an axis reach the "
            "%d-vertex limit; the layer may be incomplete",
            name,
            MAX_REPEATED_VERTICES,
        )

    kwargs: Dict[str, Any] = {
        "features": pd.DataFrame(feats, columns=["label", "roi_id"]),
        "metadata": {
            "array_id": tensor_desc.array_id,
            "roi_set": name,
            "roi_ids": [roi_id for _, roi_id in feats],
        },
    }
    if scale is not None and len(scale) >= n_axes:
        kwargs["scale"] = list(scale[lead:n_axes])
    if kind == "points":
        # ~1% of the image, in data units; never a speck.
        kwargs["size"] = max(6.0, min(shape[n_lead], shape[n_lead + 1]) / 100.0)
        kwargs["face_color"] = color
    else:
        kwargs.update(
            shape_type=shape_types,
            edge_width=widths,
            edge_color=color,
            face_color="transparent",
        )
    return RoiLayerSpec(
        kind=kind,
        name=name,
        data=data,
        kwargs=kwargs,
        dim_labels=labels[lead:n_axes] if labels else None,
    )


def roi_layer_specs(
    rois,
    tensor_desc,
    *,
    scale: Sequence[float] | None = None,
) -> List[RoiLayerSpec]:
    """The layers *rois* become on the image *tensor_desc* describes: per set, a
    Points layer and a Shapes layer for whichever the set holds.

    *scale* is the image layer's scale vector (one entry per non-samples axis);
    each layer takes the trailing slice that matches its axes.
    """
    shape = list(tensor_desc.shape)
    _, _, _, s_idx = _resolve_axes(shape, tensor_desc.dim_labels)
    n_axes = len(shape) - (1 if s_idx is not None else 0)
    labels = canonical_dim_labels(tensor_desc)

    by_set: Dict[str, list] = {}
    for roi in rois:
        by_set.setdefault(roi.set_name or _DEFAULT_SET, []).append(roi)

    specs: List[RoiLayerSpec] = []
    for name, members in by_set.items():
        points = [(r, None) for r in members if _arm(r) == "point"]
        shapes = [(r, g) for r in members if (g := _geometry(r)) is not None]
        if skipped := len(members) - len(points) - len(shapes):
            logger.warning(
                "ROI set %s: %d annotation(s) are not points or 2-D shapes; skipped",
                name,
                skipped,
            )
        color = _edge_color(name)
        for kind, group in (("points", points), ("shapes", shapes)):
            if group:
                layer_name = f"{name} ({kind})" if points and shapes else name
                specs.append(
                    _build_layer(
                        kind,
                        layer_name,
                        group,
                        tensor_desc,
                        shape,
                        n_axes,
                        scale,
                        labels,
                        color,
                    )
                )
    return specs


def image_scale(client, source_id: str, tensor_desc) -> List[float] | None:
    """The image layer's scale vector for *tensor_desc* (one entry per
    non-samples axis), or ``None`` when the server advertises none."""
    _, _, _, s_idx = _resolve_axes(tensor_desc.shape, tensor_desc.dim_labels)
    scale, _ = build_layer_scale(
        client,
        source_id,
        len(tensor_desc.shape),
        tensor_id=tensor_desc.array_id,
        tensor_desc=tensor_desc,
        rgb=s_idx is not None,
    )
    return scale


def add_roi_layers(viewer, specs: Sequence[RoiLayerSpec]):
    """Add *specs* to *viewer*; returns the layers."""
    import numpy as np

    layers = []
    for spec in specs:
        if spec.kind == "points":
            layer = viewer.add_points(
                np.asarray(spec.data, dtype=float), name=spec.name, **spec.kwargs
            )
        else:
            layer = viewer.add_shapes(
                [np.asarray(d, dtype=float) for d in spec.data],
                name=spec.name,
                **spec.kwargs,
            )
        _apply_axis_labels(viewer, layer, spec.dim_labels)
        layers.append(layer)
    return layers
