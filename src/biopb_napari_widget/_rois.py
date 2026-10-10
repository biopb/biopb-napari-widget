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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ._tensor_utils import _apply_axis_labels, _resolve_axes, canonical_dim_labels

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
    #: Axes of the image the layer spans, as indices into its non-samples axes.
    axes: Sequence[int] = ()


def roi_set_name(roi) -> str:
    return roi.set_name or _DEFAULT_SET


def sets_in(rois) -> Dict[str, int]:
    """Annotation count per set, in first-seen order."""
    counts: Dict[str, int] = {}
    for roi in rois:
        name = roi_set_name(roi)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _arm(roi) -> str | None:
    try:
        return roi.roi.WhichOneof("shape")
    except (AttributeError, ValueError):
        return None


def _geometry(roi) -> tuple[str, List[tuple]] | None:
    """``(napari shape_type, (y, x) vertices)`` for a 2-D shape arm, else ``None``."""
    arm = _arm(roi)
    g = roi.roi
    if arm == "rectangle":
        r = g.rectangle
        x0, y0 = r.top_left.x, r.top_left.y
        x1, y1 = r.bottom_right.x, r.bottom_right.y
        return "rectangle", [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]
    if arm == "ellipse":
        e = g.ellipse
        cx, cy = e.center.x, e.center.y
        rx, ry = e.radius.x, e.radius.y
        # In-plane rotation about the centre, radians, from +x toward +y.
        c, s = math.cos(e.rotation), math.sin(e.rotation)
        out = []
        for dx, dy in ((-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)):
            out.append((cy + dx * s + dy * c, cx + dx * c - dy * s))
        return "ellipse", out
    if arm == "polygon":
        return "polygon", [(p.y, p.x) for p in g.polygon.points]
    if arm == "polyline":
        return "path", [(p.y, p.x) for p in g.polyline.points]
    return None


def _pin(roi, n_lead: int) -> Dict[int, int]:
    """The ROI's pin on the leading (non-Y/X) axes; anything else is dropped."""
    return {int(a): int(i) for a, i in dict(roi.plane).items() if 0 <= int(a) < n_lead}


def _prefixes(pin: Dict[int, int], lead: int, n_lead: int, shape) -> List[tuple]:
    """Leading coordinates for an ROI: its pinned index on each axis from *lead*,
    every index where it is unpinned."""
    choices = [
        [pin[a]] if a in pin else range(int(shape[a])) for a in range(lead, n_lead)
    ]
    return list(itertools.product(*choices))


def roi_layer_specs(
    rois,
    tensor_desc,
    *,
    scale: Sequence[float] | None = None,
    only_sets: Sequence[str] | None = None,
) -> List[RoiLayerSpec]:
    """The layers *rois* become on the image *tensor_desc* describes.

    *scale* is the image layer's scale vector (one entry per non-samples axis);
    each layer takes the trailing slice that matches its axes. *only_sets*
    restricts to those set names.
    """
    shape = list(tensor_desc.shape)
    _, _, _, s_idx = _resolve_axes(shape, tensor_desc.dim_labels)
    n_axes = len(shape) - (1 if s_idx is not None else 0)
    n_lead = n_axes - 2  # every axis ahead of Y, X
    wanted = set(only_sets) if only_sets is not None else None

    by_set: Dict[str, list] = {}
    for roi in rois:
        name = roi_set_name(roi)
        if wanted is None or name in wanted:
            by_set.setdefault(name, []).append(roi)

    labels = canonical_dim_labels(tensor_desc)
    specs: List[RoiLayerSpec] = []
    for set_index, (name, members) in enumerate(by_set.items()):
        points = [(r, None) for r in members if _arm(r) == "point"]
        shapes = [(r, _geometry(r)) for r in members]
        shapes = [(r, g) for r, g in shapes if g is not None]
        skipped = len(members) - len(points) - len(shapes)
        if skipped:
            logger.warning(
                "ROI set %s: %d annotation(s) are not points or 2-D shapes; skipped",
                name,
                skipped,
            )
        both = bool(points) and bool(shapes)
        color = _EDGE_COLORS[set_index % len(_EDGE_COLORS)]

        for kind, group in (("points", points), ("shapes", shapes)):
            if not group:
                continue
            pins = [_pin(r, n_lead) for r, _ in group]
            lead = min((min(p) for p in pins if p), default=n_lead)
            data, shape_types, widths, feats = [], [], [], []
            budget = MAX_REPEATED_VERTICES
            truncated = False
            for (roi, geom), pin in zip(group, pins, strict=True):
                for prefix in _prefixes(pin, lead, n_lead, shape):
                    if budget <= 0:
                        truncated = True
                        break
                    budget -= 1
                    if kind == "points":
                        data.append(prefix + (roi.roi.point.y, roi.roi.point.x))
                    else:
                        shape_type, verts = geom
                        data.append([prefix + v for v in verts])
                        shape_types.append(shape_type)
                        widths.append(
                            float(roi.roi.polyline.width)
                            if shape_type == "path" and roi.roi.polyline.width > 0
                            else 1.0
                        )
                    feats.append({"label": roi.label, "roi_id": roi.roi_id})
            if truncated:
                logger.warning(
                    "ROI set %s: unpinned annotations repeated along an axis hit the "
                    "%d-vertex limit; the layer is incomplete",
                    name,
                    MAX_REPEATED_VERTICES,
                )

            axes = tuple(range(lead, n_axes))
            kwargs: Dict[str, Any] = {
                "name": f"{name} ({kind})" if both else name,
                "features": _features(feats),
                "metadata": {
                    "array_id": tensor_desc.array_id,
                    "roi_set": name,
                    "roi_ids": [f["roi_id"] for f in feats],
                    "dim_labels": labels[lead:n_axes] if labels else None,
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
            specs.append(
                RoiLayerSpec(
                    kind=kind,
                    name=kwargs["name"],
                    data=data,
                    kwargs=kwargs,
                    axes=axes,
                )
            )
    return specs


def _features(rows: List[dict]):
    import pandas as pd

    return pd.DataFrame(rows, columns=["label", "roi_id"])


def add_roi_layers(viewer, specs: Sequence[RoiLayerSpec]):
    """Add *specs* to *viewer*; returns the layers."""
    import numpy as np

    layers = []
    for spec in specs:
        kwargs = dict(spec.kwargs)
        dim_labels = kwargs["metadata"].pop("dim_labels", None)
        if spec.kind == "points":
            layer = viewer.add_points(np.asarray(spec.data, dtype=float), **kwargs)
        else:
            layer = viewer.add_shapes(
                [np.asarray(d, dtype=float) for d in spec.data], **kwargs
            )
        _apply_axis_labels(viewer, layer, dim_labels)
        layers.append(layer)
    return layers
