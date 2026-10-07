"""Where a label set's ``array_id`` says it belongs.

A label set is an ordinary tensor of its image, addressed under a marked
segment: ``<image array_id>/@labels/<name>[/<level>]``. The catalog has no
``role`` column and a listing's per-tensor entry carries no metadata, so the
path is the only statement that a tensor is a set, and this module is the one
place in the widget that reads it.

Owned here rather than imported from ``biopb.tensor`` (which is dropping
``split_label_array_id``). The rule matches the server's and the web app's
(``label-id.ts``): the **last** ``@labels`` segment with a name after it is the
one, whatever the image's own field contains; a name is slash-free, so anything
past it is a native pyramid level.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

#: The segment that marks a label set under its image, in a wire id.
LABELS_SEGMENT = "@labels"


class LabelAddress(NamedTuple):
    """A label set's ``array_id``, taken apart."""

    image_array_id: str  #: the image this set annotates, as an ``array_id``
    name: str  #: the set's name, slash-free
    level: Optional[str]  #: a native pyramid level under the set, if named


def split_label_array_id(array_id: str) -> Optional[LabelAddress]:
    """Take a label set's ``array_id`` apart, or ``None`` if it names no set.

    ``"src0/@labels/nuclei"`` -> image ``"src0"``, name ``"nuclei"``. Pass a
    *stable* id: a content-pinned ``id@token`` is not one.
    """
    parts = array_id.split("/")
    # From 1: ``parts[0]`` is the source_id, which is slash-free and so cannot
    # be the ``@labels`` segment of a field. A bare source called "@labels" is
    # a source.
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] != LABELS_SEGMENT or not parts[i + 1]:
            continue
        return LabelAddress(
            image_array_id="/".join(parts[:i]),
            name=parts[i + 1],
            level="/".join(parts[i + 2 :]) or None,
        )
    return None
