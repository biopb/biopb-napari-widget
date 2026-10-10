"""Tensor browser widget for napari.

Provides a tree-based UI to browse biopb.tensor datastore catalog and add
selected tensors as dask arrays to the napari viewer. Supports authentication
tokens and search filtering.

Uses pure Qt for complex UI (tree widget, custom layouts).
"""

import html
import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Sequence, Set
from urllib.parse import urlparse

from biopb.tensor import Connection, ResolveCancelled
from qtpy.QtCore import QSize, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._catalog import CatalogSource
from .._labels import split_label_array_id
from .._rois import add_roi_layers, image_scale, roi_layer_specs
from .._tensor_utils import add_tensor_layer
from .._urls import is_local_url
from ._sources import SourceList

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


# ==============================================================================
# Tree Building Utilities (adapted from JS SourceTree.tsx)
# ==============================================================================


class _TreeNode:
    """Internal tree node for building the source tree."""

    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: str,  # "folder" or "source"
        depth: int,
        source: CatalogSource | None = None,
    ):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.depth = depth
        self.source = source
        self.children: List[_TreeNode] = []
        # Set on the top-level node of a drag-dropped (``dnd://``) branch: the
        # tree shows a remove [x] on it, and ``remove_root`` is the ``dnd://``
        # prefix passed to ``deregister_local_path`` to deregister the whole branch.
        self.dropped: bool = False
        self.remove_root: str | None = None


# Origin scheme stamped by the tensor server on a drag-dropped source's catalog
# ``source_url`` (server-side ``DND_URL_PREFIX`` in ``source_manager.py``). It is a
# display-only marker of drop provenance; the tree strips it so a dropped source
# renders under a clean root, identical to a scheme-less re-root. Keep in sync
# with the server constant and the web viewer's ``getPathParts``.
_DND_URL_PREFIX = "dnd://"


def _get_path_parts(url: str) -> List[str]:
    """Extract tree path parts from source_url.

    Splits on both POSIX (``/``) and Windows (``\\``) separators so a catalog
    indexed on Windows — whose ``source_url`` is a backslash path like
    ``C:\\Users\\me\\img.tif`` — builds the same folder tree as a POSIX one
    instead of collapsing into a single flat leaf (the whole path as one name).
    A leading drive-letter token (``C:``) is dropped: ``urlparse`` reads it as a
    URL scheme so it is usually already gone, but when it survives it is just
    noise in the folder hierarchy.

    For an authority URL (a non-empty netloc — remote ``tensor-server`` mirrors
    ``grpc://host:port/remote/path``, ``s3://bucket/key``, …) the endpoint
    ``<scheme>://<netloc>`` is emitted as the FIRST part, so mirrored sources nest
    by their remote filepath under an endpoint root rather than collapsing into a
    flat ``grpc:`` node (biopb/biopb#297). A local ``file://`` url has an empty
    netloc, so it is unchanged (still just its path).

    A ``dnd://`` drop-origin url is stripped of its scheme and split as a plain
    path, so a dropped source renders under a clean top-level root just like a
    scheme-less re-root. Stripping it as a string (rather than via ``urlparse``)
    also avoids netloc/port misparsing of a basename like ``exp:2.zarr``.

    Mirror of the web viewer's ``getPathParts`` in
    ``web/packages/app/src/components/SourceTree.tsx`` — keep the two
    behaviorally in lockstep.
    """
    if not url:
        return []
    if url.startswith(_DND_URL_PREFIX):
        raw = url[len(_DND_URL_PREFIX) :]
        return [p for p in re.split(r"[\\/]+", raw) if p]
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None
    if parsed is not None and parsed.scheme and parsed.netloc:
        path_parts = [p for p in re.split(r"[\\/]+", parsed.path) if p]
        return [f"{parsed.scheme}://{parsed.netloc}"] + path_parts
    raw = (parsed.path if parsed is not None else None) or url
    parts = [p for p in re.split(r"[\\/]+", raw) if p]
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    return parts


# Marks a label-set row. A filled ring rather than a word: the row already
# carries the set's name and shape, and the tree elides long labels.
_LABEL_GLYPH = "\u25c9"

#: Marks a source row whose file is a cloud placeholder (``needs_recall``).
_CLOUD_GLYPH = "\u2601"
_CLOUD_TOOLTIP = "Cloud file, not downloaded. Resolving it downloads the whole file."
#: Marks a source the server is still indexing in the background.
_PENDING_GLYPH = "\u22ef"
_PENDING_TOOLTIP = "Being indexed in the background."
_GLYPH_W, _GLYPH_H = 26, 16


def _format_shape(shape: List[int]) -> str:
    """Format shape as compact string."""
    return "×".join(str(s) for s in shape)


def _row_shape(shape: List[int]) -> str:
    """Shape for a tree row: leading singleton axes squeezed out (a (1, 1, Z, Y,
    X) source reads Z×Y×X); an all-singleton shape keeps its last axis."""
    dims = list(shape)
    while len(dims) > 1 and dims[0] == 1:
        dims.pop(0)
    return _format_shape(dims)


#: Item data role holding a row's suffix (shape / tensor count), which
#: ``_SuffixDelegate`` draws small and grey after the name.
_SUFFIX_ROLE = Qt.ItemDataRole.UserRole + 3


class _SuffixDelegate(QStyledItemDelegate):
    """Draws a row's name as usual and its suffix after it, smaller and grey.

    The name elides first, so the suffix stays readable on a narrow panel.
    """

    _GAP = 8

    def paint(self, painter, option, index):
        suffix = index.data(_SUFFIX_ROLE)
        if not suffix:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        name = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget
        )
        rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget
        )

        small = QFont(opt.font)
        if small.pointSizeF() > 0:
            small.setPointSizeF(small.pointSizeF() * 0.85)
        else:
            small.setPixelSize(max(1, round(small.pixelSize() * 0.85)))
        suffix_w = QFontMetrics(small).horizontalAdvance(suffix)
        name_w = max(rect.width() - suffix_w - self._GAP, rect.width() // 2)
        elided = QFontMetrics(opt.font).elidedText(
            name, Qt.TextElideMode.ElideRight, name_w
        )

        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        group = QPalette.ColorGroup.Active
        fg = opt.palette.color(
            group,
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text,
        )
        grey = QColor(fg)
        grey.setAlphaF(0.6)

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(fg)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided
        )
        x = rect.left() + QFontMetrics(opt.font).horizontalAdvance(elided) + self._GAP
        if x + suffix_w <= rect.right() + 1:
            painter.setFont(small)
            painter.setPen(grey)
            painter.drawText(
                rect.adjusted(x - rect.left(), 0, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                suffix,
            )
        painter.restore()


def _tensor_short_name(array_id: str) -> str:
    """Get short name for tensor from its array_id."""
    parts = [p for p in array_id.split("/") if p]
    return parts[-1] if parts else array_id


class _TensorGroup(NamedTuple):
    """An image tensor and the label sets addressed under it."""

    image: object  #: the image's catalog descriptor
    label_sets: List[object]  #: its sets, by array_id; empty for a plain tensor


def _group_tensors(tensors: Sequence) -> List[_TensorGroup]:
    """A source's tensors, with each label set filed under the image it annotates.

    The catalog lists a set as an ordinary tensor of the source -- there is no
    ``role`` column, by decision (biopb/biopb#1059) -- so the path is what says
    a tensor is one, and this is where the tree reads it.

    Grouping is what keeps "how many tensors" from becoming the wrong question
    for a browser: an ordinary image that gained an ``@ome`` set would otherwise
    read as a two-tensor source, lose its shape badge and stop opening on
    double-click. A set whose image the source does not list keeps a group of
    its own rather than vanishing -- it should not happen, but a tensor the
    catalog lists and the tree hides is the worse of the two failures.

    Mirror of the SPA's ``groupTensors`` (``web/packages/app/src/utils/sourceTree.ts``).
    """
    groups: Dict[str, _TensorGroup] = {}
    sets = []
    for tensor in tensors:
        address = split_label_array_id(tensor.array_id)
        if address is not None:
            sets.append((tensor, address.image_array_id))
            continue
        # dict order is insertion order, which leaves images in the order the
        # server listed them -- it puts image tensors first on purpose.
        groups.setdefault(tensor.array_id, _TensorGroup(tensor, []))
    for tensor, image_array_id in sets:
        group = groups.get(image_array_id)
        if group is not None:
            group.label_sets.append(tensor)
        else:
            groups[tensor.array_id] = _TensorGroup(tensor, [])
    for group in groups.values():
        group.label_sets.sort(key=lambda t: t.array_id)
    return list(groups.values())


def _sole_image(src: CatalogSource):
    """The source's one image tensor, or ``None`` when it has several.

    What ``len(src.tensors) == 1`` used to answer: whether this source opens as
    a single layer. Its label sets do not count -- they are added on their own
    rows, never implicitly with the image.
    """
    groups = _group_tensors(src.tensors)
    return groups[0].image if len(groups) == 1 else None


def _is_unresolved(src: CatalogSource) -> bool:
    """A source whose content has not been resolved yet, so shape/dtype are
    unknown until the server hydrates it (the cloud / synced-folder case).
    Resolving such a source downloads its whole file, so it is an explicit,
    blocking action rather than something browsing triggers.

    The server's own answer, not the empty field list it used to be inferred
    from: "no tensors" also describes a source that resolved cleanly and had
    nothing readable in it, and offering *that* a Resolve that can only
    succeed-and-change-nothing is the wrong branch (biopb/biopb#1032).
    """
    return not src.is_resolved


#: Reasons a source is unresolved other than a cloud placeholder.
_NOT_RECALL_REASONS = frozenset({"pending", "failed"})
#: Row suffix for the reasons that get one ("pending" gets a glyph instead).
_UNRESOLVED_BADGES = {"failed": "  [failed]"}


def _needs_recall(src: CatalogSource) -> bool:
    """An unresolved source whose resolve downloads a cloud file, so it needs the
    user's consent. Anything the server has not said is ``pending`` or ``failed``
    counts, including an older server that gives no reason at all."""
    return _is_unresolved(src) and src.unresolved_reason not in _NOT_RECALL_REASONS


def _unresolved_badge(src: CatalogSource) -> str:
    """Row suffix for a source the server failed to register. Empty for a
    cloud placeholder, a pending one (see :func:`_row_glyph`) and a resolved
    source."""
    if not _is_unresolved(src):
        return ""
    return _UNRESOLVED_BADGES.get(src.unresolved_reason, "")


def _is_empty_source(src: CatalogSource) -> bool:
    """A resolved source with nothing on it -- browsing into it would open an
    empty node, so the tree leaves it out rather than rendering a dead end.

    Gated on :func:`_is_unresolved` rather than ``tensors`` alone: an
    unresolved source also lists no tensors, but for the opposite reason --
    they are unknown, not absent -- and it still needs its row so a viewer can
    resolve it.

    ``len(src.tensors)`` is the raw catalog count, label sets included
    (biopb/biopb#1059); that is fine here, since zero of that raw count means
    there is truly nothing, image or label, to show.

    Mirror of the SPA's ``isEmptySource`` (``web/packages/app/src/utils/sourceTree.ts``).
    """
    return not _is_unresolved(src) and len(src.tensors) == 0


class _ResolveWorker(QThread):
    """Runs the blocking ``SourceList.resolve`` off the GUI thread.

    Resolving a cloud source downloads the whole file (a recall that can take
    minutes), so it must not run on the Qt event loop. This thread does the work
    and reports back via signals; the widget keeps a modal progress dialog up
    until one fires, so the user is blocked from other actions but the app stays
    painted. Server heartbeats are relayed via :attr:`progress`, and the dialog's
    Cancel button calls :meth:`request_cancel` — a cooperative stop checked at
    each heartbeat (so it takes effect within one heartbeat interval; the
    server-side recall finishes and is cached, so a later resolve coalesces).
    """

    resolved = Signal(object)  # the refreshed CatalogSource
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)  # ResolveProgress

    def __init__(self, sources: SourceList, source_id: str):
        super().__init__()
        self._sources = sources
        self._source_id = source_id
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the running resolve to stop (thread-safe, idempotent)."""
        self._cancel.set()

    def run(self):
        try:
            descriptor = self._sources.resolve(
                self._source_id,
                on_progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except ResolveCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        self.resolved.emit(descriptor)


# A dropped folder with more than this many filesystem entries prompts a
# confirmation before the recursive scan is sent (a footgun-stopper for dropping
# a home/root folder by mistake). Counted client-side: drag-drop is gated to a
# localhost server, so the client shares the server's disk. Coarse on purpose --
# it counts entries, not resulting sources.
_LARGE_DROP_ENTRY_THRESHOLD = 2000


def _is_onedrive_dir_name(name: str) -> bool:
    """True for a OneDrive root directory name (``OneDrive`` / ``OneDrive - <Org>``).

    A deliberate small copy of the server's discovery skip (biopb_tensor_server
    .discovery._is_skippable_system_dir): the server declines to walk these trees,
    which is exactly what makes a source added from inside one worth a heads-up.
    Kept independent rather than imported because the tensor-server package is not
    a runtime dependency of this widget.
    """
    low = name.lower()
    return low == "onedrive" or low.startswith(("onedrive -", "onedrive-"))


def _cloud_drop_warning(path: str) -> str | None:
    """Warning text if *path* sits in a cloud-synced folder, else ``None``.

    Name-based and best-effort: it inspects the path components only (never opens
    a file, so it cannot itself trigger a cloud recall). Detects OneDrive -- the
    common Windows case, and exactly the subtree the server's monitored-tree walk
    skips. Split on both separators so a Windows path survives whatever the drop
    delivered.
    """
    components = [c for c in re.split(r"[\\/]+", path) if c]
    if not any(_is_onedrive_dir_name(c) for c in components):
        return None
    name = os.path.basename(path.rstrip("/\\")) or path
    return (
        f"“{name}” is inside OneDrive, a cloud-synced folder.\n\n"
        "Cloud-synced source support is experimental.\n\n"
        "It will be indexed while its files are downloaded to this PC, but if "
        "OneDrive Files On-Demand later dehydrates them to free up space, reads "
        "can become slow or fail until Windows re-downloads them. Marking the "
        "files “Always keep on this device” avoids that.\n\n"
        "Add anyway?"
    )


def _dir_exceeds_entry_threshold(path: str) -> bool:
    """True if *path* is a directory holding more than the large-drop threshold.

    A non-directory (a single file/dataset) is never "large". Short-circuits at
    the threshold, so it stays cheap even on an enormous tree. This is the only
    size gate on a drop — the server does not re-check it (a direct SDK caller
    passing a path is trusted as explicit intent) — so a walk error (permission,
    race) is treated as "not large" and the drop proceeds unconfirmed rather
    than being blocked.
    """
    if not os.path.isdir(path):
        return False
    try:
        count = 0
        for _root, dirs, files in os.walk(path):
            count += len(dirs) + len(files)
            if count > _LARGE_DROP_ENTRY_THRESHOLD:
                return True
    except OSError:
        return False
    return False


#: Most matches a server-side search shows. The server's own cap is a safety
#: valve, not a page size: every row is an id to ship, a tree node to build and a
#: row to render, and nobody reads a hundred thousand results. One more than this
#: is asked for, so "there were more" is known without a count.
SERVER_QUERY_LIMIT = 2000

#: Deepest folder level a search opens by itself. A match deep in a big tree would
#: otherwise open every folder on the way down; the top level shows where the
#: matches are, and the user opens what they want.
AUTO_EXPAND_DEPTH = 1


def _search_sql(query: str) -> str:
    """The catalog query for a (lowercased, trimmed) search *query*: ids of the
    sources whose id, url or type contain it, in url order, one past the limit."""
    escaped = query.replace("\\", "\\\\").replace("'", "''")
    escaped = escaped.replace("%", "\\%").replace("_", "\\_")
    return (
        "SELECT source_id FROM sources WHERE "
        f"LOWER(source_id) LIKE '%{escaped}%' ESCAPE '\\' OR "
        f"LOWER(source_url) LIKE '%{escaped}%' ESCAPE '\\' OR "
        f"LOWER(source_type) LIKE '%{escaped}%' ESCAPE '\\' "
        f"ORDER BY source_url LIMIT {SERVER_QUERY_LIMIT + 1}"
    )


class _SearchWorker(QThread):
    """Runs one server-side search off the GUI thread.

    A search over a large catalog is a round trip plus a scan; run on the GUI
    thread it freezes the window on every pause in typing. Answers can arrive
    out of order, so each carries the generation it was asked for and the widget
    drops the superseded ones.
    """

    done = Signal(int, object, bool)  # generation, source ids, more matched
    failed = Signal(int)

    def __init__(self, client, query: str, generation: int):
        super().__init__()
        self._client = client
        self._query = query
        self._generation = generation

    def run(self):
        try:
            rows = self._client.query(_search_sql(self._query), format="records")
        except Exception:
            logger.exception("Server search failed")
            self.failed.emit(self._generation)
            return
        ids = [row["source_id"] for row in rows]
        self.done.emit(
            self._generation,
            set(ids[:SERVER_QUERY_LIMIT]),
            len(ids) > SERVER_QUERY_LIMIT,
        )


class _RoiWorker(QThread):
    """Fetches one annotation set of a tensor and turns it into layer specs, off
    the GUI thread -- the scale is a client call and a large set is a per-ROI
    Python loop. Only adding the layers is left to the GUI thread."""

    done = Signal(str, object, bool)  # set name, [RoiLayerSpec], truncated
    failed = Signal(str)

    def __init__(self, client, source_id: str, tensor_desc, set_name: str):
        super().__init__()
        self._client = client
        self._source_id = source_id
        self._tensor_desc = tensor_desc
        self._set_name = set_name

    def run(self):
        try:
            result = self._client.list_rois(self._tensor_desc.array_id, self._set_name)
            rois = list(result.rois)
            specs = []
            if rois:
                scale = image_scale(self._client, self._source_id, self._tensor_desc)
                specs = roi_layer_specs(rois, self._tensor_desc, scale=scale)
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        self.done.emit(self._set_name, specs, bool(getattr(result, "truncated", False)))


def _count(result, field: str) -> int:
    """A count field of an add result, 0 when an older server omits it."""
    return int(getattr(result, field, 0) or 0)


class _AddSourceWorker(QThread):
    """Runs ``SourceList.add`` for one dropped path off the GUI thread.

    Registering a dropped file/dir asks the server to discover + catalog it,
    which for a plain folder is a slow recursive walk that may add many sources,
    so it must not run on the Qt event loop. Presented *non-modally*:
    the user keeps using the viewer while sources appear. Per-source progress is
    relayed via :attr:`progress`; :meth:`request_cancel` cooperatively stops the
    walk (the client closes the stream; sources already registered stay).

    One drop == one path == one ``register_local_path`` call == one terminal result
    ``(added, refreshed, removed, failed)``. Multi-item drops are refused upstream
    (``_local_paths_from_mime``), so there is no cross-path aggregation here — a
    single call keeps the progress count monotone. An oversized folder is caught
    *before* this worker starts, by the widget's client-side confirm prompt
    (``_confirm_large_drop``), so the walk only runs once the user has agreed.
    """

    progress = Signal(object)  # AddSourceProgress
    done = Signal(object)  # (added, refreshed, removed, failed)
    failed = Signal(str)
    # What the server passed over because `cloud` was not set -- (offline
    # placeholder files, OneDrive directories); emitted after `done`, only when
    # either is non-zero.
    skipped = Signal(int, int)

    def __init__(self, sources: SourceList, path: str, cloud: bool = False):
        super().__init__()
        self._sources = sources
        self._path = path
        self._cloud = cloud
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the running add to stop (thread-safe, idempotent)."""
        self._cancel.set()

    def run(self):
        try:
            result = self._sources.add(
                self._path,
                cloud=self._cloud,
                on_progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        added = list(result.added)
        # `refreshed` rather than `already_present`: re-dropping a known path
        # rebuilds it, so "already present" would report a no-op that did not
        # happen. A rebuild that failed is reported in `failed` instead.
        refreshed = list(result.refreshed)
        removed = list(result.removed)
        failed = [(f.path, f.reason) for f in result.failed]
        self.done.emit((added, refreshed, removed, failed))
        # An older result has no such fields, which reads as nothing skipped.
        files = _count(result, "skipped_offline")
        dirs = _count(result, "skipped_cloud_dirs")
        if (files or dirs) and not self._cloud:
            self.skipped.emit(files, dirs)


class _RemoveSourceWorker(QThread):
    """Runs ``SourceList.remove`` for one dropped branch off the GUI thread.

    Removal is quick server-side (unregister N adapters), but a rescan may briefly
    hold the catalog lock, so it runs off the Qt event loop like the add worker.
    One [x] click == one ``deregister_local_path`` call == one terminal ``(removed,
    failed)`` tally. Only drag-dropped (``dnd://``) branches are ever removable, so
    there is nothing to cancel and no path aggregation.
    """

    done = Signal(object)  # (removed_ids, failed)
    failed = Signal(str)

    def __init__(self, sources: SourceList, root_url: str):
        super().__init__()
        self._sources = sources
        self._root_url = root_url

    def run(self):
        try:
            result = self._sources.remove(self._root_url)
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        removed = list(result.removed)
        failed = [(f.path, f.reason) for f in result.failed]
        self.done.emit((removed, failed))


def _human_bytes(n: int) -> str:
    """Compact human-readable byte size (e.g. ``4.2 GB``)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _build_tree(sources: Dict[str, CatalogSource]) -> _TreeNode:
    """Build hierarchical tree from sources based on source_url paths."""
    root = _TreeNode(node_id="", name="", node_type="folder", depth=0)

    for src in sources.values():
        parts = _get_path_parts(src.source_url)
        # A drag-dropped (dnd://) source's top-level node is a removable branch
        # root; tag it so the tree shows a remove [x], and record the dnd:// prefix
        # that deregister_local_path() targets (all of one drop's sources share it).
        is_dropped = src.source_url.startswith(_DND_URL_PREFIX)
        remove_root = (_DND_URL_PREFIX + parts[0]) if (is_dropped and parts) else None
        if not parts:
            # No path parts, add directly to root
            leaf = _TreeNode(
                node_id=src.source_id,
                name=src.source_id,
                node_type="source",
                depth=1,
                source=src,
            )
            if is_dropped:
                leaf.dropped = True
                leaf.remove_root = src.source_url
            root.children.append(leaf)
            continue

        # Navigate/create folder path
        current = root
        for i in range(len(parts) - 1):
            part = parts[i]
            child = next(
                (
                    c
                    for c in current.children
                    if c.node_type == "folder" and c.name == part
                ),
                None,
            )
            if not child:
                child = _TreeNode(
                    node_id=current.node_id + "/" + part,
                    name=part,
                    node_type="folder",
                    depth=current.depth + 1,
                )
                current.children.append(child)
            # The top-level folder of a dropped folder-branch is its removable root.
            if i == 0 and is_dropped:
                child.dropped = True
                child.remove_root = remove_root
            current = child

        # Add source as leaf
        source_name = parts[-1]
        leaf = _TreeNode(
            node_id=src.source_id,
            name=source_name,
            node_type="source",
            depth=current.depth + 1,
            source=src,
        )
        # A single-part dropped source (one dropped file/dataset) is itself the root.
        if is_dropped and len(parts) == 1:
            leaf.dropped = True
            leaf.remove_root = remove_root
        current.children.append(leaf)

    # Sort children: folders first, then sources, both alphabetically
    def sort_children(node: _TreeNode):
        node.children.sort(
            key=lambda c: (0 if c.node_type == "folder" else 1, c.name.lower())
        )
        for child in node.children:
            sort_children(child)

    sort_children(root)

    # Flatten paths: merge folders that have only one folder child
    def flatten_paths(node: _TreeNode):
        for child in node.children:
            if child.node_type == "folder":
                flatten_paths(child)
                # Flatten while single folder child
                while (
                    len(child.children) == 1 and child.children[0].node_type == "folder"
                ):
                    grandchild = child.children[0]
                    child.name = child.name + "/" + grandchild.name
                    child.node_id = grandchild.node_id
                    child.children = grandchild.children
                    for gc in child.children:
                        gc.depth = child.depth + 1
                    flatten_paths(child)

    flatten_paths(root)
    return root


# ==============================================================================
# Metadata Dialog
# ==============================================================================


def _is_empty_for_display(value) -> bool:
    """Check if a value is empty for display purposes.

    Filters out null, empty arrays, empty objects, and nested empty structures.
    """
    if value is None:
        return True
    if isinstance(value, list):
        if not value:
            return True
        return all(_is_empty_for_display(v) for v in value)
    if isinstance(value, dict):
        if not value:
            return True
        return all(_is_empty_for_display(v) for v in value.values())
    return isinstance(value, str) and not value.strip()


def _filter_empty_metadata(metadata: Dict) -> Dict:
    """Filter out empty items from metadata dict."""
    if not metadata:
        return {}

    filtered = {}
    for key, value in metadata.items():
        if not _is_empty_for_display(value):
            filtered[key] = value

    return filtered


# The pane under the tree: secondary information, deliberately quieter than the
# tree above it. Shared so the identifier rows sit flush with the lines below.
_MUTED_QSS = "color: #888; font-size: 11px;"

# Small enough to sit inside an 11px line without becoming the thing the eye
# lands on -- the identifier is what is being read, the button is how it leaves.
_COPY_BUTTON_QSS = "QPushButton { font-size: 10px; padding: 0px 5px; }"


def _make_selectable(label: QLabel):
    """Let *label*'s text be selected and copied.

    A QLabel is ``NoTextInteraction`` by default, which is right for a caption
    and wrong for anything a reader has to get out of the window intact
    (biopb/biopb#972).
    """
    label.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )


class _CopyableIdRow(QWidget):
    """One ``Name: value`` line of the info pane, with a button that copies it.

    The identifiers in this pane are arguments -- ``Tensor`` is what
    ``client.get_tensor(...)`` takes -- rather than captions, so they have to
    leave the screen exactly (biopb/biopb#972). Selecting the text works and is
    enabled here, but a drag that stops one character short of the end produces
    an ``array_id`` that is wrong in a way nothing downstream can catch. The
    button copies the value this row was *given*, never the rendered string, so
    no styling, eliding or wrapping can get between the two.
    """

    #: The value that reached the clipboard, for the caller to confirm.
    copied = Signal(str)

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._name = name
        self._value = ""

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setStyleSheet(_MUTED_QSS)
        _make_selectable(self._label)
        # Stretched, so the label owns the row's width and wraps inside it
        # rather than widening the dock. An identifier long enough to be worth
        # a button fills the line, which is what puts the button at the end of
        # it; on a short one the button sits at the pane's right edge, aligned
        # with the row below -- the failure mode worth having.
        row.addWidget(self._label, stretch=1)

        self._button = QPushButton("Copy")
        self._button.setStyleSheet(_COPY_BUTTON_QSS)
        self._button.setToolTip(f"Copy the {name.lower()} to the clipboard")
        # Top-aligned: a wrapped label is two lines tall and a centred button
        # would float away from the line whose value it copies.
        row.addWidget(self._button, alignment=Qt.AlignTop)
        self._button.clicked.connect(self._copy)

    def set_value(self, value: str):
        """Show *value*, and make it what the button copies."""
        self._value = value
        self._label.setText(f"{self._name}: {value}")

    def value(self) -> str:
        """What the button would copy right now."""
        return self._value

    def _copy(self):
        QApplication.clipboard().setText(self._value)
        self.copied.emit(self._value)


class MetadataDialog(QDialog):
    """Dialog to display source and tensor metadata."""

    def __init__(
        self,
        parent: QWidget,
        source: CatalogSource,
        tensor_id: str | None = None,
        metadata: Dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setMinimumSize(600, 500)
        self.resize(650, 600)

        layout = QVBoxLayout(self)

        # Compact header: source path - shape dtype
        header_layout = QHBoxLayout()

        # Source URL path (use stem)
        url_parts = _get_path_parts(source.source_url)
        url_display = "/" + "/".join(url_parts) if url_parts else source.source_id
        url_label = QLabel(url_display)
        url_label.setStyleSheet("color: #60a5fa; font-weight: bold;")
        _make_selectable(url_label)
        header_layout.addWidget(url_label)

        # Tensor info inline
        tensor_desc = None
        if tensor_id:
            tensor_desc = next(
                (t for t in source.tensors if t.array_id == tensor_id), None
            )
        elif len(source.tensors) == 1:
            tensor_desc = source.tensors[0]

        if tensor_desc:
            header_layout.addWidget(QLabel("—"))
            shape_str = _format_shape(tensor_desc.shape)
            shape_label = QLabel(shape_str)
            shape_label.setStyleSheet("color: #a78bfa;")
            _make_selectable(shape_label)
            header_layout.addWidget(shape_label)

            dtype_label = QLabel(tensor_desc.dtype)
            dtype_label.setStyleSheet("color: #fbbf24;")
            _make_selectable(dtype_label)
            header_layout.addWidget(dtype_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Metadata section
        layout.addWidget(QLabel("Metadata"))
        meta_text = QTextEdit()
        meta_text.setReadOnly(True)
        meta_text.setStyleSheet(
            "QTextEdit { background-color: #1e2435; color: #e2e8f0; font-family: monospace; }"
        )

        if metadata:
            # Filter empty items and format JSON with indentation
            filtered = _filter_empty_metadata(metadata)
            if filtered:
                formatted = json.dumps(filtered, indent=2)
                meta_text.setPlainText(formatted)
            else:
                meta_text.setPlainText("No metadata available")
        else:
            meta_text.setPlainText("No metadata available")

        layout.addWidget(meta_text)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ==============================================================================
# Tensor Browser Widget (Pure Qt)
# ==============================================================================

# Bottom message pane. One label shows all transient feedback and errors; its
# lifecycle is driven by a level:
#   "info"  — a one-shot outcome ("added 3 sources"). Self-clears after
#             _MESSAGE_AUTO_CLEAR_MS so a stale line does not linger (the d&d
#             summary was the motivating case).
#   "busy"  — an ongoing state ("Connecting…", "Indexing…"). Sticky: it reflects
#             a condition, so the flow that started it clears it when the
#             condition resolves, not a timer.
#   "error" — sticky until replaced or explicitly cleared (never times out).
_MESSAGE_AUTO_CLEAR_MS = 6000


def _callout_qss(text_hex: str, accent_hex: str, bg_rgba: str) -> str:
    """A left-accented callout stylesheet, so the pane reads as one component
    whose color alone signals severity (blue = status, red = error)."""
    return (
        "QLabel {"
        f" color: {text_hex};"
        " font-weight: bold;"
        f" background-color: {bg_rgba};"
        f" border-left: 3px solid {accent_hex};"
        " border-radius: 2px;"
        " padding: 4px 8px;"
        " }"
    )


# level -> stylesheet. "busy" shares the blue status look with "info".
_MESSAGE_STYLES = {
    "info": _callout_qss("#93c5fd", "#60a5fa", "rgba(96, 165, 250, 40)"),
    "busy": _callout_qss("#93c5fd", "#60a5fa", "rgba(96, 165, 250, 40)"),
    "error": _callout_qss("#fca5a5", "#ef4444", "rgba(239, 68, 68, 40)"),
}


class TensorBrowserWidget(QWidget):
    """Widget to browse and load tensors from a TensorFlight server."""

    # Emitted (via the source list's on_changed hook) when the background source
    # watcher re-lists the catalog from its daemon thread. A Qt signal — not a
    # direct call or QTimer — because the watcher fires off the Qt main thread;
    # the queued connection marshals the tree rebuild back onto it.
    _sources_changed = Signal(object)

    # Emitted (with the connect generation) from the background connect worker
    # when a connect attempt finishes. A Qt signal — not a direct call —
    # because the worker runs off the Qt main thread; the queued connection
    # marshals the tree render back onto it. See :meth:`_start_connect`.
    _connect_done = Signal(int)

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        connection: Connection | None = None,
        compute_scheduler: str | None = None,
    ):
        super().__init__()
        self._viewer = viewer
        # Shared with the MCP kernel when it hands one in, so a reconnect here
        # is the agent's next ``client`` too. Standalone, the widget owns it.
        self._conn = connection or Connection()
        self._list = SourceList(self._conn)
        # Set once a connect finds no control: the URL and token fields are
        # then how a plane is named, until the control answers again.
        self._manual = False
        # When set (MCP context), pin loaded layers' slice reads to a
        # single-process scheduler so the serial viewer shares the main-process
        # chunk cache instead of scattering across the cluster (issue #8). None
        # in the standalone napari plugin -> arrays passed through unchanged.
        self._compute_scheduler = compute_scheduler
        self._selected_source_id: str | None = None
        self._selected_tensor_id: str | None = None
        self._expanded_folders: Set[str] = set()
        # On the first unfiltered render, expand every top-level node so the
        # user lands on the first level of leaves instead of a wall of
        # collapsed roots. Flipped once, then ordinary expand-state tracking
        # (``_expanded_folders``) takes over so later rebuilds respect the
        # user's manual collapses.
        self._initial_expand_done: bool = False

        # Connect runs on a worker thread (see :meth:`_start_connect`).
        # ``_connecting`` is True while one is in flight (the watcher skips
        # re-rendering then). ``_connect_gen`` is a supersession token: each
        # new connect bumps it so a stale worker's result is dropped.
        self._connecting: bool = False
        self._connect_gen: int = 0

        # Set up widget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # In-flight resolve workers, owned by thread lifetime: a worker is held
        # here from start() until its `finished` fires (then discarded +
        # deleteLater'd). A set, not a single slot, so overlapping resolves can't
        # clobber each other's only ref and get the QThread GC'd / destroyed while
        # still running -- important once a non-modal progress/cancel lets two run.
        self._resolve_workers: set = set()
        # In-flight drag-drop add worker (at most one at a time) plus GC retention
        # to the ``finished`` signal, mirroring the resolve ownership rule.
        self._add_worker: _AddSourceWorker | None = None
        self._add_retain: set = set()
        # Dropped paths whose last add left offline files out. The name check
        # only knows OneDrive, so a re-drop of one of these is where the other
        # synced folders get asked about cloud mode. Session-only.
        self._skipped_cloud_paths: set = set()
        # In-flight dropped-branch remove worker (at most one at a time), same
        # ownership rule as the add worker.
        self._remove_worker: _RemoveSourceWorker | None = None
        self._remove_retain: set = set()
        # Server-side search: the generation of the query whose answer is wanted
        # (bumped by every new query and every clear, so a late answer is
        # dropped), the worker running it, and whether more sources matched than
        # the limit let through.
        self._search_generation = 0
        self._search_worker: _SearchWorker | None = None
        self._search_retain: set = set()
        self._search_more = False
        # In-flight ROI fetches, kept alive until they finish.
        self._roi_retain: set = set()
        self._setup_ui()

        # Self-heal the tree when the watcher re-lists a catalog listed
        # mid-index (issue #44). It runs on a daemon thread, so it reaches the
        # GUI through a queued signal.
        self._sources_changed.connect(self._on_sources_changed)
        self._list.on_changed = self._sources_changed.emit
        self._list.start_watch()

        # Render a background connect's outcome on the Qt main thread.
        self._connect_done.connect(self._on_connect_done)

        # Auto-connect on next event loop tick
        QTimer.singleShot(0, self._auto_connect)

    @property
    def _client(self):
        return self._conn.client

    @property
    def _connected(self) -> bool:
        return self._conn.client is not None

    @property
    def _sources(self):
        return self._list.sources

    @property
    def _use_server_query(self):
        return self._list.use_server_query

    def _setup_ui(self):
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Accept file/folder drops onto the widget (see dragEnterEvent/dropEvent).
        # The affordance and its enablement are surfaced by the drop-hint row
        # below -- a refused drag never reaches dropEvent, so the *reason* must
        # live in this always-visible label, not in a drop-time message.
        self.setAcceptDrops(True)

        # Compact connection summary row: a single clickable line showing the
        # server + state ("server_url — connected") with a trailing disclosure
        # caret. Clicking it toggles the advanced connection controls
        # (token/Connect/Refresh). Those controls are touched once at setup;
        # day-to-day the user only needs to see *that* they are connected, so
        # they are collapsed by default. The caret + pointing-hand cursor are the
        # affordance that the line is expandable.
        self._advanced_expanded = False
        self._status_summary = QLabel()
        self._status_summary.setWordWrap(True)
        self._status_summary.setCursor(Qt.PointingHandCursor)
        self._status_summary.setToolTip("Show/hide connection settings")
        # A QLabel has no clicked signal; route its click straight to the toggle.
        self._status_summary.mousePressEvent = lambda _event: self._toggle_advanced()
        layout.addWidget(self._status_summary)

        # Advanced connection panel — hidden until the summary line is clicked.
        # Holds Connect/Refresh and, only once a connect has found no control,
        # a URL and token to dial instead. While a control answers, it names
        # the plane and its credential, and nothing here is typed (#628).
        self._advanced_panel = QWidget()
        adv_layout = QVBoxLayout(self._advanced_panel)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(4)

        self._manual_panel = QWidget()
        manual_layout = QVBoxLayout(self._manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(4)
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("grpc://host:8815")
        self._url_input.returnPressed.connect(self._on_connect_clicked)
        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText("token (optional)")
        self._token_input.setEchoMode(QLineEdit.Password)
        self._token_input.returnPressed.connect(self._on_connect_clicked)
        for label, field in (
            ("Server:", self._url_input),
            ("Token:", self._token_input),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(field)
            manual_layout.addLayout(row)
        self._manual_panel.setVisible(False)
        adv_layout.addWidget(self._manual_panel)

        # Connect and Refresh buttons
        btn_layout = QHBoxLayout()
        self._connect_button = QPushButton("Connect")
        self._connect_button.clicked.connect(self._on_connect_clicked)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh)
        self._refresh_button.setEnabled(False)
        btn_layout.addWidget(self._connect_button)
        btn_layout.addWidget(self._refresh_button)
        adv_layout.addLayout(btn_layout)

        self._advanced_panel.setVisible(False)
        layout.addWidget(self._advanced_panel)
        self._update_status_summary()

        # Filter input (label and input on same row)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Search sources...")
        self._filter_input.textChanged.connect(self._on_filter_text_changed)
        filter_layout.addWidget(self._filter_input)
        layout.addLayout(filter_layout)
        # Under the search box on a large catalog: how many sources, that the
        # filter runs on the server, and whether the answer is partial.
        self._search_status = QLabel()
        self._search_status.setStyleSheet("color: gray; font-size: 11px;")
        self._search_status.setVisible(False)
        layout.addWidget(self._search_status)

        # Debounce timer for filter
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._apply_filter)

        # Tree widget - give it most of the space
        self._tree_widget = QTreeWidget()
        self._tree_widget.setHeaderHidden(True)
        # Column 0 holds the name/tree; a narrow column 1 holds the remove [x]
        # button on drag-dropped root rows only (empty, zero-width otherwise).
        self._tree_widget.setColumnCount(2)
        _header = self._tree_widget.header()
        _header.setStretchLastSection(False)
        _header.setSectionResizeMode(0, QHeaderView.Stretch)
        _header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tree_widget.setExpandsOnDoubleClick(False)
        self._tree_widget.setIndentation(12)
        self._tree_widget.setIconSize(QSize(_GLYPH_W, _GLYPH_H))
        self._tree_widget.setItemDelegateForColumn(
            0, _SuffixDelegate(self._tree_widget)
        )
        # Column 0 stretches to the viewport and row text elides (ElideRight is
        # the QTreeView default), so a horizontal scrollbar is never needed. Pin
        # it off: left ScrollBarAsNeeded, its show/hide toggles as the widest
        # row's text crosses the viewport width, and on non-overlay-scrollbar
        # platforms (Windows) the bar steals viewport height -- shifting every
        # row vertically on an otherwise-unchanged refresh (biopb/biopb#367).
        self._tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        self._tree_widget.itemClicked.connect(self._on_tree_item_clicked)
        self._tree_widget.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self._tree_widget.setStyleSheet("QTreeWidget { min-height: 300px; }")
        layout.addWidget(self._tree_widget, stretch=1)

        # Drag-drop affordance row: an always-visible hint reflecting whether a
        # drop is possible right now (connected + localhost), plus a Cancel button
        # shown only while an add is in flight. The hint is the ONLY place a
        # refused drop's reason can be shown (dragEnterEvent -> ignore never
        # reaches dropEvent), and it doubles as the non-modal progress line.
        drop_layout = QHBoxLayout()
        self._drop_hint_label = QLabel()
        self._drop_hint_label.setWordWrap(True)
        self._drop_hint_label.setStyleSheet(_MUTED_QSS)
        drop_layout.addWidget(self._drop_hint_label, stretch=1)
        self._add_cancel_btn = QPushButton("Cancel")
        self._add_cancel_btn.setFixedWidth(60)
        self._add_cancel_btn.setVisible(False)
        self._add_cancel_btn.clicked.connect(self._cancel_add)
        drop_layout.addWidget(self._add_cancel_btn)
        layout.addLayout(drop_layout)
        self._update_drop_hint()

        # Metadata display. Selectable throughout (biopb/biopb#972): this pane
        # exists to be read *out of* -- into a notebook cell, an issue, a chat
        # -- and a QLabel refuses a selection by default. The two identifier
        # lines are their own rows so each can carry a copy button, since a
        # drag-selection of an array_id has to land on exactly the right
        # character and silently does not say when it did not.
        self._metadata_pane = QWidget()
        metadata_layout = QVBoxLayout(self._metadata_pane)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(1)
        self._source_id_row = _CopyableIdRow("Source")
        self._tensor_id_row = _CopyableIdRow("Tensor")
        for id_row in (self._source_id_row, self._tensor_id_row):
            id_row.copied.connect(self._on_id_copied)
            metadata_layout.addWidget(id_row)
        self._metadata_label = QLabel()
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setStyleSheet(_MUTED_QSS)
        _make_selectable(self._metadata_label)
        metadata_layout.addWidget(self._metadata_label)
        self._metadata_pane.setVisible(False)
        layout.addWidget(self._metadata_pane)

        # Bottom message pane: a single callout that carries both transient
        # status/progress ("Indexing…", "added 3 sources") and inline errors,
        # colored by level and self-clearing per the rules on _MESSAGE_STYLES /
        # _MESSAGE_AUTO_CLEAR_MS above. It is styled as a left-accented callout
        # so it stays visually distinct from the grey metadata pane above.
        # _show_status / _show_error / _clear_status / _clear_error route here.
        self._message_level: str | None = None
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_message)
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setVisible(False)
        layout.addWidget(self._message_label)

    def _on_connect_clicked(self, *args):
        """Connect button handler: dial the typed URL, or ask the control again.

        With no URL typed this is a plain retry, which is what a user needs
        after starting the control.
        """
        url = self._url_input.text().strip() if self._manual else ""
        token = self._token_input.text().strip() or None
        self._start_connect(url or None, token)

    def _auto_connect(self):
        """Connect on startup using the resolved URL/token (no user prompt)."""
        self._start_connect()

    def _start_connect(self, url: str | None = None, token: str | None = None):
        """Connect, then list the catalog, on a worker thread.

        :meth:`Connection.connect` blocks through the plane's boot, so it must
        not run on the Qt main thread: the viewer stays responsive, and in the
        MCP context the widget lives in the kernel whose Qt loop
        ``start_kernel`` waits on. Completion is marshaled back via
        ``_connect_done``; a generation token drops the result of any connect
        the user has since superseded.
        """
        self._clear_error()
        self._clear_status()
        self._tree_widget.clear()
        self._metadata_pane.setVisible(False)
        self._selected_source_id = None
        self._selected_tensor_id = None

        self._connect_gen += 1
        gen = self._connect_gen
        self._connecting = True
        self._connect_button.setEnabled(False)
        self._update_status_summary()
        # The endpoint is unknown until the control names it, so on a first
        # connect there is no address to show yet.
        target = self._conn.url or "the data plane"
        self._show_status(f"Connecting to {target}…", sticky=True)

        def _worker():
            # connect() records its own failure on last_message; the list is
            # read here too, so the main thread only renders.
            try:
                self._list.clear()
                connected = self._conn.connect(url, token)
                if url is None:
                    # No control answered (the env URL would have set one).
                    self._manual = not connected and self._conn.url is None
                if connected:
                    self._list.update_health()
                    self._list.refresh()
            except Exception as exc:
                logger.exception("Connect worker failed")
                self._conn.client = None
                self._conn.last_message = f"Could not list the catalog: {exc}"
            finally:
                self._connect_done.emit(gen)

        threading.Thread(target=_worker, name="tbw-connect", daemon=True).start()

    def _on_connect_done(self, gen: int):
        """Render the outcome of a background connect (main thread).

        Queued from ``_connect_done``. A stale generation (the user retargeted a
        different server while this one was still connecting) is dropped; the
        superseding connect owns the UI.
        """
        if gen != self._connect_gen:
            return
        self._connecting = False
        self._connect_button.setEnabled(True)
        self._clear_status()
        self._update_status_summary()
        self._update_drop_hint()

        self._manual_panel.setVisible(self._manual)
        if not self._connected:
            if self._manual:
                # Open the panel so the fields that answer this are in view.
                self._advanced_expanded = True
                self._advanced_panel.setVisible(True)
                self._update_status_summary()
            self._show_error(
                self._conn.last_message or "Could not reach the biopb data plane."
            )
            self._tree_widget.clear()
            self._refresh_button.setEnabled(False)
            return

        sources = self._list.sources
        if not sources:
            # While the server is still indexing, keep Refresh enabled (more
            # sources are coming, and the watcher re-lists as they appear); a
            # genuinely empty server leaves it disabled, as before.
            indexing = self._show_empty_state()
            self._refresh_button.setEnabled(indexing)
            return

        if self._use_server_query:
            logger.info(
                "Large catalog (%d sources), server-side SQL filter enabled",
                len(sources),
            )
        self._update_search_chrome()

        self._build_and_display_tree()
        self._refresh_button.setEnabled(True)

    def _toggle_advanced(self):
        """Show/hide the full connection controls behind the summary line."""
        self._advanced_expanded = not self._advanced_expanded
        self._advanced_panel.setVisible(self._advanced_expanded)
        self._update_status_summary()

    def _update_status_summary(self):
        """Refresh the compact connection summary line.

        Renders ``<url> — <state> <caret>`` with a leading state glyph and a
        trailing disclosure caret, mirroring the live connection state
        (connecting / connected / disconnected) so the user can see they are
        connected without expanding the advanced panel. The caret signals that
        the line is clickable to reveal the connection settings.
        """
        # The URL is control-derived / config fallback (#413); it is embedded in a
        # rich-text QLabel, so escape it or a '&'/'<' would corrupt the markup.
        url = html.escape(self._conn.url or "(no server)")
        if self._connecting:
            glyph, color, state = "◌", "#888", "connecting…"
        elif self._connected:
            glyph, color, state = "●", "#4ade80", "connected"
        else:
            glyph, color, state = "○", "#f87171", "disconnected"
        caret = "▾" if self._advanced_expanded else "▸"
        self._status_summary.setText(
            f"<span style='color:{color}'>{glyph}</span> "
            f"<b>{url}</b> — <span style='color:{color}'>{state}</span> "
            f"<span style='color:#888'>{caret}</span>"
        )

    def _show_empty_state(self) -> bool:
        """Render the no-sources state, distinguishing indexing from empty.

        With progressive discovery the server reports ``SERVING`` while its
        data-folder scan is still running, so an empty catalog right after
        connect is often "not done indexing yet," not "nothing here." When the
        last-observed health says a full scan is in progress, show a transient
        grey status instead of an error -- the background source watcher re-lists
        the tree automatically as sources are found. Returns True in that case,
        False when the catalog is genuinely empty (an error is shown).
        """
        if self._list.scan_in_progress():
            self._clear_error()
            self._show_status(
                f"Indexing data folder… "
                f"({self._list.scan_source_count()} sources so far). "
                "The list updates automatically as sources are found.",
                sticky=True,
            )
            return True
        self._clear_status()
        self._show_error("No sources found on server")
        return False

    def _show_message(self, msg: str, *, level: str, sticky: bool):
        """Show *msg* in the bottom pane at *level* (see _MESSAGE_STYLES).

        Non-sticky messages self-clear after _MESSAGE_AUTO_CLEAR_MS; sticky ones
        persist until replaced or explicitly cleared. Each call restarts (or
        stops) the single-shot timer, so the visible message always owns it.
        """
        self._message_level = level
        self._message_label.setStyleSheet(_MESSAGE_STYLES[level])
        self._message_label.setText(msg)
        self._message_label.setVisible(True)
        self._message_timer.stop()
        if not sticky:
            self._message_timer.start(_MESSAGE_AUTO_CLEAR_MS)

    def _clear_message(self):
        """Clear the bottom pane regardless of level and cancel any timer."""
        self._message_timer.stop()
        self._message_level = None
        self._message_label.setVisible(False)
        self._message_label.setText("")

    def _show_error(self, msg: str):
        """Display an inline error (red, sticky until replaced/cleared)."""
        self._show_message(msg, level="error", sticky=True)

    def _clear_error(self):
        """Clear the pane only if it is currently showing an error.

        Callers sprinkle this before a new action to wipe a stale error; it must
        not knock out a busy/info status set by a concurrent flow (e.g. the
        background "Indexing…" line), so it is scoped to the error level.
        """
        if self._message_level == "error":
            self._clear_message()

    def _report_failure(self, title: str, message: str):
        """Modally report a failed *user-initiated* action (resolve/load).

        These actions are explicit, consenting gestures the user actively
        triggered and watched (a modal progress dialog, or a busy cursor during
        load), so their failure deserves an acknowledged modal box rather than
        the easily-missed, transient inline error pane (which is wiped by the
        next refresh/selection). Background errors (connect, refresh, list) stay
        on the inline pane -- issue #206.
        """
        QMessageBox.critical(self, title, message or "Unknown error")

    def _show_status(self, msg: str, *, sticky: bool = False):
        """Display a transient status/progress message (blue callout).

        Pass ``sticky=True`` for an *ongoing* state ("Connecting…", "Indexing…")
        whose owning flow clears it when the state resolves; leave it False for a
        one-shot outcome ("added 3 sources") that should self-clear.
        """
        self._show_message(msg, level="busy" if sticky else "info", sticky=sticky)

    def _clear_status(self):
        """Clear the pane only if it is currently showing a status (not an error)."""
        if self._message_level in ("info", "busy"):
            self._clear_message()

    # ------------------------------------------------------------------
    # Drag-and-drop: add a dropped local file/dir as a source and serve it.
    # ------------------------------------------------------------------

    def _can_accept_drop(self) -> tuple[bool, str]:
        """Whether a drop is possible right now, and the reason to display.

        The reason is the ambient affordance text -- a refused drag never reaches
        ``dropEvent``, so this string (shown in the drop-hint label) is the only
        place the user learns *why* a drop is unavailable. Drops are enabled only
        against a connected, **localhost** server: a dropped path is a client-side
        filesystem path, meaningful to the server only when they share a disk.
        """
        if self._connecting or not self._connected:
            return False, "Not connected — connect to add data by drag-drop"
        if not (self._conn.url and is_local_url(self._conn.url)):
            return False, "Connected to a remote server — drag-drop unavailable"
        if self._add_worker is not None:
            return False, "Adding data…"
        return True, "Drop image files or folders here to add them"

    def _update_drop_hint(self):
        """Refresh the ambient drop-hint text from the current connection state."""
        if self._add_worker is not None:
            return  # a live add owns the label (progress line)
        _, reason = self._can_accept_drop()
        self._drop_hint_label.setText(reason)

    @staticmethod
    def _local_paths_from_mime(mime) -> List[str]:
        """The single local path carried by a drag, wrapped in a list, or [].

        A drop is accepted only when it is *exactly one* local file/folder. A
        multi-item drag is refused (returns ``[]``, so the cursor shows "no
        drop") rather than partially accepted — the add pipeline handles one
        path per drop, and one folder still discovers many datasets in a single
        call, so the common case is unaffected. A non-file URL (e.g. a web link)
        is likewise rejected.
        """
        if not mime.hasUrls():
            return []
        urls = mime.urls()
        if len(urls) != 1:
            return []  # exactly one item per drop; multi-select is refused
        url = urls[0]
        if not url.isLocalFile():
            return []
        return [url.toLocalFile()]

    def dragEnterEvent(self, event):
        """Accept a drag only if it is all-local files onto a localhost server."""
        ok, _ = self._can_accept_drop()
        if ok and self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        """Register the single dropped local path as a source on the server."""
        ok, _ = self._can_accept_drop()
        paths = self._local_paths_from_mime(event.mimeData())
        if not ok or not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        path = paths[0]
        if not self._confirm_large_drop(path):
            return  # user declined scanning an oversized folder; nothing sent
        if not self._confirm_cloud_drop(path):
            return  # user declined adding from a cloud-synced folder
        # A confirmed cloud drop is the user's say-so to register the offline
        # placeholders too. Any other folder is told if the server passed some
        # over (`_on_add_skipped`) and asked about cloud mode if it is
        # dropped again.
        cloud = _cloud_drop_warning(path) is not None or self._confirm_cloud_redrop(
            path
        )
        self._start_add(path, cloud=cloud)

    def _confirm_cloud_redrop(self, path: str) -> bool:
        """Ask about cloud mode for a path whose last add left files out.

        False without asking for any other path. The server only accepts cloud
        mode on a folder it does not already hold, so this is the answer for a
        folder removed and dropped again; for one still registered the server
        refuses, and that message is shown as for any failed add.
        """
        if path not in self._skipped_cloud_paths:
            return False
        name = os.path.basename(path.rstrip("/\\")) or path
        resp = QMessageBox.question(
            self,
            "Add as cloud sources?",
            f"Offline files or OneDrive folders in “{name}” were left out the "
            "last time it was added.\n\nAdd it as a cloud folder this time, so they are "
            "included? Opening one will download it first.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return resp == QMessageBox.Yes

    def _confirm_large_drop(self, path: str) -> bool:
        """Ask before scanning a big folder; return True to proceed.

        Only a directory over the entry threshold prompts — a file or a small
        folder proceeds silently. The count runs *locally*: the drop UI is
        enabled only against a localhost server (``_can_accept_drop``), so the
        client shares the server's filesystem and can size the tree cheaply
        (short-circuiting at the threshold) before any scan is sent. This is a
        coarse footgun-stopper for dropping a home/root folder by mistake, kept
        client-side so the user stays in control instead of the server hard-
        rejecting; the walk itself still happens server-side once confirmed.
        """
        if not _dir_exceeds_entry_threshold(path):
            return True
        name = os.path.basename(path.rstrip("/")) or path
        resp = QMessageBox.question(
            self,
            "Add large folder?",
            f"“{name}” contains many files. Scan it and add all datasets "
            "found under it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return resp == QMessageBox.Yes

    def _confirm_cloud_drop(self, path: str) -> bool:
        """Warn before adding a source from a cloud-synced (OneDrive) folder.

        OneDrive "Files On-Demand" can evict a file's bytes to the cloud, leaving
        an offline placeholder: indexed fine while resident, but a later read
        recalls it -- slow, or failing outright when offline. The server keeps such
        a source registered across rescans (its walk skips OneDrive; the reconcile
        preserves already-registered claims there), so this is a heads-up about
        read behavior, not a block -- the default action is to proceed. Returns
        True to add, False to cancel.
        """
        warning = _cloud_drop_warning(path)
        if warning is None:
            return True
        resp = QMessageBox.question(
            self,
            "Add source from a cloud folder?",
            warning,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return resp == QMessageBox.Yes

    def _start_add(self, path: str, cloud: bool = False):
        """Spawn the off-GUI-thread add worker for *path* (non-modal)."""
        if self._add_worker is not None:
            return  # one add at a time
        self._clear_error()
        worker = _AddSourceWorker(self._list, path, cloud)
        self._add_worker = worker
        self._add_retain.add(worker)
        worker.progress.connect(self._on_add_progress)
        worker.done.connect(self._on_add_done)
        worker.failed.connect(self._on_add_failed)
        worker.skipped.connect(
            lambda files, dirs, p=path: self._on_add_skipped(p, files, dirs)
        )
        if cloud:
            # Done, not failed: a refused cloud drop is still worth asking again.
            worker.done.connect(
                lambda _payload, p=path: self._skipped_cloud_paths.discard(p)
            )
        worker.finished.connect(lambda w=worker: self._add_retain.discard(w))
        self._add_cancel_btn.setVisible(True)
        label = os.path.basename(path.rstrip("/")) or path
        self._drop_hint_label.setText(f"Adding {label}…")
        worker.start()

    def _cancel_add(self):
        """Cancel the in-flight add (keeps sources already registered)."""
        if self._add_worker is not None:
            self._add_worker.request_cancel()
            self._drop_hint_label.setText("Cancelling…")

    def _on_add_progress(self, progress):
        """Relay per-source add progress to the drop-hint line (count-up)."""
        count = progress.added_count
        path = progress.current_path or ""
        msg = f"Adding… {count} source{'' if count == 1 else 's'} registered"
        if os.path.isabs(path):
            # A real filesystem path being scanned -> show its basename.
            msg += f" (scanning {os.path.basename(path.rstrip('/'))})"
        elif path:
            # A status sentence (e.g. the catalog-lock wait heartbeat) -> show it
            # verbatim, not run through the "scanning {basename}" label.
            msg += f" ({path})"
        self._drop_hint_label.setText(msg)

    def _on_add_failed(self, msg: str):
        """Whole-request add failure (e.g. path not found on the server)."""
        self._add_worker = None
        self._add_cancel_btn.setVisible(False)
        self._update_drop_hint()
        self._report_failure("Add data failed", msg)

    def _on_add_skipped(self, path: str, files: int, dirs: int):
        """Tell the user what the drop left out: offline files and OneDrive
        directories (everything inside the latter is missing, and uncounted).
        Either one makes the path a potential cloud root.

        Information only. Re-sending the drop with ``cloud=True`` is not an
        offer to make: once a drop has registered anything its folder is a known
        root with cloud mode off, and the server refuses to switch that on
        afterwards ("Cannot switch cloud mode on inside ..."). The path is
        remembered, so dropping it again after removing it asks about cloud mode
        (`_confirm_cloud_redrop`).
        """
        self._skipped_cloud_paths.add(path)
        name = os.path.basename(path.rstrip("/\\")) or path
        parts = []
        if files:
            parts.append(
                f"{files} offline file{'' if files == 1 else 's'} (their contents "
                "are not on this PC)"
            )
        if dirs:
            parts.append(f"{dirs} OneDrive folder{'' if dirs == 1 else 's'}")
        QMessageBox.information(
            self,
            "Some files were not added",
            f"{' and '.join(parts)} in “{name}” were left out.\n\n"
            "To include them, remove the folder and drop it again.",
        )

    def _on_add_done(self, payload):
        """Terminal add tally: refresh, summarize, report failures."""
        added, refreshed, removed, failed = payload
        self._add_worker = None
        self._add_cancel_btn.setVisible(False)

        # Prompt sources appear immediately; the background watcher would also
        # catch up, but an explicit refresh is prompt. A rebuilt source changes
        # shape/dtype in place, so the tree needs re-rendering for that too.
        if added or refreshed or removed:
            try:
                self._refresh()
            except Exception:
                logger.exception("refresh after register_local_path failed")

        parts = []
        if added:
            parts.append(f"added {len(added)}")
        if refreshed:
            parts.append(f"{len(refreshed)} refreshed")
        if removed:
            parts.append(f"{len(removed)} removed")
        if failed:
            parts.append(f"{len(failed)} failed")
        self._show_status("Add data: " + (", ".join(parts) if parts else "nothing"))
        self._update_drop_hint()

        if failed:
            detail = "\n".join(
                f"• {os.path.basename(p.rstrip('/')) or p}: {reason}"
                for p, reason in failed
            )
            self._report_failure("Some items were not added", detail)

    def _refresh(self):
        """Refresh the source list from server."""
        self._clear_error()

        if not self._connected:
            self._show_error("Not connected")
            return

        try:
            sources = self._list.refresh()
        except Exception:
            # A failed re-list almost always means the server is gone. Drop the
            # shared client so the indicator, and the agent's next job, say so;
            # a reconnect restores both. Scoped to the re-list call: a later
            # render error is a client-side bug, not a lost server.
            self._conn.client = None
            self._conn.last_message = "Lost connection to server"
            self._list.clear()
            self._show_error("Refresh failed — lost connection to server")
            self._refresh_button.setEnabled(False)
            self._update_status_summary()
            logger.exception("Failed to refresh source list")
            return

        try:
            if not sources:
                self._show_empty_state()
                self._tree_widget.clear()
                return

            self._update_search_chrome()
            self._build_and_display_tree()
        except Exception:
            # The server answered; rendering the catalog is a client-side step,
            # so a failure here is not a lost connection -- report it without
            # dropping the client (the pre-#318 behavior for a render error).
            self._show_error("Refresh failed")
            logger.exception("Failed to display refreshed source list")

    def _on_sources_changed(self, sources):
        """Rebuild the tree after the background watcher re-lists (issue #44).

        Runs on the Qt main thread (queued from ``_sources_changed``). The
        connection has already swapped in the fresh catalog, so we just re-render
        — through ``_apply_filter`` so any active search text is preserved — and
        only while connected and not mid-(re)connect, to avoid fighting a
        concurrent connect that is about to repaint anyway.
        """
        if not self._connected or self._connecting:
            return
        self._clear_error()
        self._apply_filter()

    def closeEvent(self, event):
        """Stop the watcher; the list is this widget's alone."""
        self._list.on_changed = None
        self._list.stop_watch()
        self._search_generation += 1  # a search still out has nobody to tell
        super().closeEvent(event)

    def _build_and_display_tree(self, filtered_ids: Set[str] | None = None):
        """Build tree from sources and display in widget."""
        self._tree_widget.clear()

        if not self._sources:
            return

        # Build tree. Empty sources are dropped unconditionally, before a
        # search filter narrows further -- a source with nothing on it is
        # never worth a row, matching or not. A search builds the tree from the
        # matches alone, so its cost follows the result, not the catalog; an
        # empty match set is an empty tree, not the whole catalog.
        visible = {
            source_id: src
            for source_id, src in self._sources.items()
            if not _is_empty_source(src)
            and (filtered_ids is None or source_id in filtered_ids)
        }
        if not visible:
            return
        display_tree = _build_tree(visible)
        if filtered_ids is not None:
            # Open the top levels so the user sees where the matches are.
            level = display_tree.children
            for _ in range(AUTO_EXPAND_DEPTH):
                folders = [n for n in level if n.node_type == "folder"]
                self._expanded_folders.update(n.node_id for n in folders)
                level = [c for n in folders for c in n.children]

        # Populate tree widget
        for child in display_tree.children:
            self._add_tree_node(self._tree_widget, child)

        # First unfiltered render: seed every top-level node as expanded so the
        # first level of leaves is visible up front. Persisted via the normal
        # expand-state set so it survives rebuilds and the user can collapse it.
        if not self._initial_expand_done and filtered_ids is None:
            for child in display_tree.children:
                self._expanded_folders.add(child.node_id)
            self._initial_expand_done = True

        # Restore expanded state
        self._restore_expanded_state()
        # Restore the highlighted item: the tree is cleared and rebuilt on every
        # filter/refresh, which drops Qt's current-item even though we still track
        # the logical selection (issue #191).
        self._restore_selection()

    def _glyph_icon(self, glyph: str) -> QIcon:
        """A fixed-size icon drawing *glyph* ("" gives a transparent one), so a
        row with the glyph and a row without it keep their text aligned."""
        cache = self.__dict__.setdefault("_glyph_icons", {})
        if glyph not in cache:
            pix = QPixmap(_GLYPH_W, _GLYPH_H)
            pix.fill(Qt.GlobalColor.transparent)
            if glyph:
                painter = QPainter(pix)
                font = QFont(painter.font())
                font.setPixelSize(_GLYPH_H)
                font.setBold(True)
                painter.setFont(font)
                color = self._tree_widget.palette().color(
                    self._tree_widget.foregroundRole()
                )
                painter.setPen(QColor(color))
                painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
                painter.end()
            cache[glyph] = QIcon(pix)
        return cache[glyph]

    def _add_tree_node(self, parent, node: _TreeNode):
        """Add a tree node to the widget."""
        item = QTreeWidgetItem(parent)
        item.setData(0, Qt.ItemDataRole.UserRole, node.node_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, node.node_type)

        # Drag-dropped branch root: a remove [x] in column 1 (see the two-column
        # tree in _setup_ui). Re-created on every rebuild -- clear() drops widgets.
        if node.dropped and node.remove_root:
            self._tree_widget.setItemWidget(
                item, 1, self._make_remove_button(node.remove_root, node.name)
            )

        if node.node_type == "folder":
            item.setText(0, node.name)
            # The horizontal scrollbar is pinned off (biopb/biopb#367), so a long
            # or deeply-indented label that elides is only readable on hover.
            item.setToolTip(0, node.name)
            for child in node.children:
                self._add_tree_node(item, child)
        else:
            # Source node
            src = node.source
            assert src is not None
            # Label sets ride under their image rather than counting as
            # tensors of the source; see _group_tensors.
            groups = _group_tensors(src.tensors)
            suffix = ""
            if len(groups) == 1:
                # Shape for a single tensor
                suffix = _row_shape(groups[0].image.shape)
            elif len(groups) > 1:
                suffix = f"{len(groups)} tensors"
            if suffix:
                display_name = f"{node.name}  {suffix}"
            else:
                display_name = f"{node.name}{_unresolved_badge(src)}"

            # No residency indicator: drawing one cost a live stat walk per
            # source on every browse (biopb/biopb#1048).
            item.setText(0, node.name if suffix else display_name)
            item.setData(0, _SUFFIX_ROLE, suffix)
            # The horizontal scrollbar is pinned off (biopb/biopb#367), so the
            # full label -- which elides when it outgrows the panel -- is only
            # readable on hover.
            item.setToolTip(0, display_name)
            # Every source row gets an icon slot, blank unless it is a cloud
            # placeholder or still indexing, so the names stay aligned.
            glyph, note = "", ""
            if _needs_recall(src):
                glyph, note = _CLOUD_GLYPH, _CLOUD_TOOLTIP
            elif _is_unresolved(src) and src.unresolved_reason == "pending":
                glyph, note = _PENDING_GLYPH, _PENDING_TOOLTIP
            item.setIcon(0, self._glyph_icon(glyph))
            if note:
                item.setToolTip(0, f"{display_name}\n{note}")

            # Nested rows: one per image when the source has several, and one
            # per label set under the image it annotates. A set is always its
            # own row -- it is added as its own layer, never with the image
            # (the parent row's double-click opens the image alone).
            for group in groups:
                parent = item
                if len(groups) > 1:
                    parent = self._add_tensor_row(item, src, group.image)
                for label_set in group.label_sets:
                    self._add_tensor_row(parent, src, label_set, is_label=True)

    def _add_tensor_row(self, parent, src, tensor, *, is_label=False):
        """One tensor row under *parent*. Returns it, so sets can nest under it."""
        row = QTreeWidgetItem(parent)
        row.setData(0, Qt.ItemDataRole.UserRole, tensor.array_id)
        row.setData(0, Qt.ItemDataRole.UserRole + 1, "tensor")
        row.setData(0, Qt.ItemDataRole.UserRole + 2, src.source_id)
        row.setData(0, _SUFFIX_ROLE, _row_shape(tensor.shape))
        if is_label:
            address = split_label_array_id(tensor.array_id)
            # The set's own name, not the last path segment: a native pyramid
            # level would otherwise be what the row reads.
            name = address.name if address else _tensor_short_name(tensor.array_id)
            text = f"{_LABEL_GLYPH} {name}"
            row.setToolTip(0, f"Label set “{name}” — adds as a Labels layer")
        else:
            text = _tensor_short_name(tensor.array_id)
        row.setText(0, text)
        return row

    def _make_remove_button(self, remove_root: str, display_name: str) -> QPushButton:
        """A small [x] button that removes a drag-dropped branch at its root."""
        btn = QPushButton("✕")
        btn.setFlat(True)
        btn.setFixedSize(18, 18)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(f"Remove dropped source “{display_name}”")
        btn.clicked.connect(
            lambda _=False, r=remove_root, n=display_name: self._on_remove_dropped(r, n)
        )
        return btn

    def _on_remove_dropped(self, remove_root: str, display_name: str):
        """Confirm, then deregister a drag-dropped branch (off the GUI thread)."""
        resp = QMessageBox.question(
            self,
            "Remove dropped source?",
            f"Remove “{display_name}” from the browser?\n\n"
            "This unregisters the drag-dropped source(s) from the tensor server. "
            "The underlying files on disk are not deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return
        self._start_remove(remove_root, display_name)

    def _start_remove(self, root_url: str, display_name: str):
        """Spawn the off-GUI-thread remove worker (one at a time)."""
        if self._remove_worker is not None:
            return  # one removal at a time
        self._clear_error()
        worker = _RemoveSourceWorker(self._list, root_url)
        self._remove_worker = worker
        self._remove_retain.add(worker)
        worker.done.connect(self._on_remove_done)
        worker.failed.connect(self._on_remove_failed)
        worker.finished.connect(lambda w=worker: self._remove_retain.discard(w))
        self._show_status(f"Removing {display_name}…")
        worker.start()

    def _on_remove_done(self, payload):
        """Terminal remove tally: refresh so the row disappears, then summarize."""
        removed, failed = payload
        self._remove_worker = None
        if removed:
            try:
                self._refresh()
            except Exception:
                logger.exception("refresh after deregister_local_path failed")
        n = len(removed)
        self._show_status(
            f"Removed {n} source{'' if n == 1 else 's'}" if n else "Nothing to remove"
        )
        if failed:
            detail = "\n".join(f"• {sid}: {reason}" for sid, reason in failed)
            self._report_failure("Some sources were not removed", detail)

    def _on_remove_failed(self, msg: str):
        """Whole-request remove failure (e.g. the server refused a non-dnd root)."""
        self._remove_worker = None
        self._report_failure("Remove failed", msg)

    def _restore_expanded_state(self):
        """Restore expanded state for folders."""

        def restore_recursive(item: QTreeWidgetItem):
            item_id = item.data(0, Qt.ItemDataRole.UserRole)
            if item_id in self._expanded_folders:
                item.setExpanded(True)
            for i in range(item.childCount()):
                restore_recursive(item.child(i))

        for i in range(self._tree_widget.topLevelItemCount()):
            restore_recursive(self._tree_widget.topLevelItem(i))

    def _restore_selection(self):
        """Re-highlight the tracked source/tensor in a freshly rebuilt tree.

        Walk the new items, find the one matching ``_selected_source_id`` (and the
        tensor field when one is selected), make it the current item, and scroll it
        into view so a refresh -- notably after resolving a cloud source -- doesn't
        lose the user's place (issue #191). Prefer the tensor child when a field is
        selected, else fall back to the source node.
        """
        if self._selected_source_id is None:
            return

        target_source: QTreeWidgetItem | None = None
        target_tensor: QTreeWidgetItem | None = None

        def find_recursive(item: QTreeWidgetItem):
            nonlocal target_source, target_tensor
            node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if node_type == "tensor":
                if (
                    self._selected_tensor_id is not None
                    and item.data(0, Qt.ItemDataRole.UserRole + 2)
                    == self._selected_source_id
                    and item.data(0, Qt.ItemDataRole.UserRole)
                    == self._selected_tensor_id
                ):
                    target_tensor = item
            elif (
                node_type == "source"
                and item.data(0, Qt.ItemDataRole.UserRole) == self._selected_source_id
            ):
                target_source = item
            for i in range(item.childCount()):
                find_recursive(item.child(i))

        for i in range(self._tree_widget.topLevelItemCount()):
            find_recursive(self._tree_widget.topLevelItem(i))

        target = target_tensor or target_source
        if target is not None:
            self._tree_widget.setCurrentItem(target)
            self._tree_widget.scrollToItem(target)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int):
        """Handle tree item click."""
        self._clear_error()

        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if node_type == "folder":
            # Toggle expansion on click
            item_id = item.data(0, Qt.ItemDataRole.UserRole)
            expanded = not item.isExpanded()
            item.setExpanded(expanded)
            if expanded:
                self._expanded_folders.add(item_id)
            else:
                self._expanded_folders.discard(item_id)
            self._metadata_pane.setVisible(False)
            return

        # Determine selection
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            self._selected_source_id = source_id
            self._selected_tensor_id = tensor_id
        else:
            # Source item clicked
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            self._selected_source_id = source_id
            src = self._sources.get(source_id)
            sole = _sole_image(src) if src else None
            self._selected_tensor_id = sole.array_id if sole else None

            # Multi-tensor source: toggle its field list on click, like a folder
            if item.childCount() > 0:
                expanded = not item.isExpanded()
                item.setExpanded(expanded)
                if expanded:
                    self._expanded_folders.add(source_id)
                else:
                    self._expanded_folders.discard(source_id)

        self._update_metadata_display()

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int):
        """Handle tree item double-click - add tensor to viewer."""
        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        # Skip folders
        if node_type == "folder":
            return

        # Determine selection and add to viewer
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            self._selected_source_id = source_id
            self._selected_tensor_id = tensor_id
        else:
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            src = self._sources.get(source_id)
            if src and _is_unresolved(src):
                # Unresolved source: double-click resolves it rather than adding
                # to the viewer -- after consent for a cloud download, at once
                # for a pending one, and as a retry for a failed one.
                self._resolve_source(source_id)
                return
            sole = _sole_image(src) if src else None
            if sole is None:
                # Several images to choose from - don't add on double-click
                return
            self._selected_source_id = source_id
            self._selected_tensor_id = sole.array_id

        self._add_to_viewer()

    def _show_context_menu(self, pos):
        """Show context menu for tree items."""
        item = self._tree_widget.itemAt(pos)
        if not item:
            return

        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        # Skip folders
        if node_type == "folder":
            return

        # Determine selection for menu actions
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            is_multi_tensor_source = False
            is_unresolved_source = False
        else:
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            src = self._sources.get(source_id)
            is_unresolved_source = src is not None and _is_unresolved(src)
            sole = _sole_image(src) if src else None
            if sole is not None:
                tensor_id = sole.array_id
                is_multi_tensor_source = False
            else:
                # Several images, or an unresolved source. "View all" is
                # offered for the images, so it is their count that decides.
                tensor_id = None
                is_multi_tensor_source = (
                    src is not None and len(_group_tensors(src.tensors)) > 1
                )

        menu = QMenu(self)

        # Primary action: Resolve (unresolved/cloud), "View all" (multi-tensor),
        # or "View" (single tensor).
        if is_unresolved_source:
            assert src is not None
            if _needs_recall(src):
                label = "Resolve (downloads file)…"
            elif src.unresolved_reason == "failed":
                label = "Retry loading"
            else:
                label = "Load now"
            resolve_action = menu.addAction(label)
            resolve_action.triggered.connect(lambda: self._resolve_source(source_id))
        elif is_multi_tensor_source:
            view_action = menu.addAction("View all")
            view_action.triggered.connect(lambda: self._view_all_tensors(source_id))
        else:
            view_action = menu.addAction("View")
            if tensor_id:
                view_action.triggered.connect(
                    lambda: self._view_tensor(source_id, tensor_id)
                )
            else:
                view_action.setEnabled(False)

        # ROI annotations: the sets a tensor has come from the catalog, so the
        # entry is built from what is there and nothing is fetched until one is
        # picked. Absent when the server cannot say (no queryable rois table);
        # a label set has no entry in the catalog's map, so none either.
        if tensor_id and not is_unresolved_source:
            self._add_roi_actions(menu, source_id, tensor_id)

        # Metadata action
        meta_action = menu.addAction("Metadata")
        meta_action.triggered.connect(
            lambda: self._show_metadata_dialog(source_id, tensor_id)
        )

        menu.exec_(self._tree_widget.mapToGlobal(pos))

    def _resolve_source(self, source_id: str):
        """Resolve an unresolved source off the GUI thread.

        A cloud source (``needs_recall``) downloads its whole file, so we (1)
        take explicit consent via a modal warning. A ``pending`` source is only
        waiting for the server to register it -- nothing is downloaded -- and a
        ``failed`` one is a retry, so both skip the warning. Then (2) run the
        blocking resolve in a worker thread behind a modal progress dialog — the user is blocked from other
        actions but the UI stays painted — and (3) on success repopulate the tree
        from the now-resolved field list. The repopulate is necessary because
        resolution does not change the server ``source_count``, so the background
        watcher won't pick it up (issue #44); we refresh explicitly here.
        """
        src = self._sources.get(source_id)
        if not src or not self._connected:
            return

        parts = _get_path_parts(src.source_url)
        name = parts[-1] if parts else source_id

        recall = _needs_recall(src)
        if recall:
            confirm = QMessageBox.warning(
                self,
                "Resolve cloud source",
                f"Resolving “{name}” downloads the entire file from remote "
                f"storage.\n\nThis may take several minutes, use local disk "
                f"space, and will not work offline. Continue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if confirm != QMessageBox.Ok:
                return

        self._clear_error()

        # Modal, indeterminate (no byte-level progress available), with a working
        # Cancel button. The label is refreshed from server heartbeats with
        # elapsed time + target size so the user can judge whether to wait. We
        # manage close ourselves (autoClose/autoReset off) so that hitting Cancel
        # shows a "Cancelling…" state and the dialog stays up until the worker
        # confirms the stop — which also blocks a second resolve in the meantime.
        verb = "Resolving" if recall else "Loading"
        progress = QProgressDialog(f"{verb} “{name}”…", "Cancel", 0, 0, self)
        progress.setWindowTitle(verb)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        worker = _ResolveWorker(self._list, source_id)
        # Own the worker by its thread lifetime (held until `finished`), so a
        # later/overlapping resolve can't drop its only ref and have the QThread
        # destroyed mid-run.
        self._resolve_workers.add(worker)

        def _on_progress(p):
            elapsed = int(p.elapsed_seconds)
            label = f"{verb} “{p.target_name or name}”… {elapsed}s"
            if p.target_bytes:
                label += f" ({_human_bytes(p.target_bytes)})"
            progress.setLabelText(label)

        def _on_resolved(_descriptor):
            progress.close()
            # Pin the just-resolved source as the logical selection so the rebuild
            # below re-highlights it and scrolls it into view -- otherwise the user
            # loses track of it in the refreshed list (issue #191). Set here (not
            # only on click) because double-click/context-menu resolve can fire
            # without a prior single-click selection. Drop any tensor sub-selection
            # so we land on the source node itself.
            self._selected_source_id = source_id
            self._selected_tensor_id = None
            # The connection snapshot was refreshed by resolve_source(); re-render
            # through _apply_filter so any active search text is preserved and the
            # resolved source now shows its shape badge / field children.
            self._apply_filter()

        def _on_failed(message):
            progress.close()
            self._report_failure("Resolve failed", message)

        def _on_cancelled():
            # User-initiated stop; the server recall continues + caches, so a
            # later resolve coalesces. Close quietly, no error banner.
            progress.close()

        def _on_cancel_clicked():
            # Cooperative request; keep the dialog up showing "Cancelling…" until
            # the worker confirms (within one heartbeat) via the cancelled signal.
            progress.setLabelText(f"Cancelling “{name}”…")
            worker.request_cancel()

        worker.progress.connect(_on_progress)
        worker.resolved.connect(_on_resolved)
        worker.failed.connect(_on_failed)
        worker.cancelled.connect(_on_cancelled)
        progress.canceled.connect(_on_cancel_clicked)
        worker.finished.connect(lambda: self._resolve_workers.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        # Modal: blocks here (event loop still runs, dialog stays painted) until a
        # slot above calls progress.close(). The worker's queued signal is
        # delivered inside this exec_, so a fast finish can't deadlock.
        progress.exec_()

    def _add_roi_actions(self, menu, source_id: str, tensor_id: str):
        """ "Load ROI annotations" for a tensor: one entry for a single set, a
        submenu with one entry per set for several. A reserved (``@``) set is the
        server's own -- read-only -- and says so."""
        sets = (self._list.roi_sets or {}).get(tensor_id)
        src = self._sources.get(source_id)
        tensor_desc = next(
            (t for t in (src.tensors if src else ()) if t.array_id == tensor_id), None
        )
        if not sets or tensor_desc is None:
            return

        target = menu
        if len(sets) > 1:
            target = menu.addMenu("Load ROI annotations")
        for name, count in sorted(sets.items()):
            tag = "  (read-only)" if name.startswith("@") else ""
            text = f"{name}{tag}  ({count:,})"
            if len(sets) == 1:
                text = f"Load ROI annotations: {text}"
            target.addAction(text).triggered.connect(
                lambda _=False, n=name: self._load_rois(source_id, tensor_desc, n)
            )

    def _load_rois(self, source_id: str, tensor_desc, set_name: str):
        """Fetch one annotation set, then add it as Points / Shapes layers."""
        if not self._client:
            return
        self._clear_error()
        QApplication.setOverrideCursor(Qt.BusyCursor)
        worker = _RoiWorker(self._client, source_id, tensor_desc, set_name)
        self._roi_retain.add(worker)
        worker.done.connect(self._on_rois_fetched)
        worker.failed.connect(self._on_rois_failed)
        worker.finished.connect(lambda w=worker: self._roi_retain.discard(w))
        worker.start()

    def _on_rois_failed(self, message: str):
        QApplication.restoreOverrideCursor()
        self._report_failure("Could not load ROI annotations", message)

    def _on_rois_fetched(self, set_name: str, specs, truncated: bool):
        QApplication.restoreOverrideCursor()
        if not specs:
            self._show_status(f"ROI set “{set_name}” is empty.")
            return
        try:
            layers = add_roi_layers(self._viewer, specs)
        except Exception as exc:
            logger.exception("Failed to add ROI layers")
            self._report_failure("Could not load ROI annotations", str(exc))
            return
        note = f"Loaded {len(layers)} ROI layer{'' if len(layers) == 1 else 's'}."
        if truncated:
            note += " The server returned a partial set (truncated)."
        self._show_status(note)

    def _view_tensor(self, source_id: str, tensor_id: str):
        """Add single tensor to viewer."""
        self._selected_source_id = source_id
        self._selected_tensor_id = tensor_id
        self._add_to_viewer()

    def _view_all_tensors(self, source_id: str):
        """Add all tensors from a source to viewer."""
        src = self._sources.get(source_id)
        if not src or not self._client:
            return

        url_parts = _get_path_parts(src.source_url)
        stem = url_parts[-1] if url_parts else source_id

        # Show busy cursor during loading
        QApplication.setOverrideCursor(Qt.BusyCursor)

        try:
            # Images only. A label set is added on its own row, never with the
            # image: "View all" means every picture this source holds, and a
            # mask silently laid over one is not that.
            for group in _group_tensors(src.tensors):
                tensor = group.image
                try:
                    tensor_name = _tensor_short_name(tensor.array_id)
                    layer_name = f"{stem}/{tensor_name}"
                    # Shared build-pyramid -> wrap -> OME scale -> add_image
                    # pipeline (also used by the MCP add_tensor).
                    add_tensor_layer(
                        self._viewer,
                        self._client,
                        source_id,
                        tensor.array_id,
                        tensor,
                        name=layer_name,
                        compute_scheduler=self._compute_scheduler,
                    )
                    logger.info(
                        "Added tensor layer '%s' from source '%s'",
                        layer_name,
                        source_id,
                    )
                except Exception:
                    logger.exception("Failed to load tensor %s", tensor.array_id)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_metadata_dialog(self, source_id: str, tensor_id: str | None):
        """Show metadata dialog for source/tensor."""
        src = self._sources.get(source_id)
        if not src:
            return

        # Fetch metadata from server
        metadata = None
        if self._client:
            try:
                metadata = self._client.get_source_metadata(source_id)
            except Exception:
                logger.warning("Failed to fetch metadata for %s", source_id)

        dialog = MetadataDialog(self, src, tensor_id, metadata)
        dialog.exec_()

    def _on_id_copied(self, value: str):
        """Confirm a copy in the message pane.

        Naming the value rather than saying "Copied": the two rows sit one line
        apart and the button that was pressed is not visible in the outcome, so
        an unnamed confirmation cannot be told from the wrong one.
        """
        self._show_status(f"Copied {value}")

    def _update_metadata_display(self):
        """Update metadata display for selected tensor."""
        if not self._selected_tensor_id or not self._selected_source_id:
            self._metadata_pane.setVisible(False)
            return

        src = self._sources.get(self._selected_source_id)
        if not src:
            self._metadata_pane.setVisible(False)
            return

        # Find tensor descriptor
        tensor_desc = next(
            (t for t in src.tensors if t.array_id == self._selected_tensor_id),
            None,
        )
        if not tensor_desc:
            self._metadata_pane.setVisible(False)
            return

        shape_str = _format_shape(tensor_desc.shape)
        dims_str = (
            ", ".join(tensor_desc.dim_labels) if tensor_desc.dim_labels else "N/A"
        )
        self._source_id_row.set_value(self._selected_source_id)
        self._tensor_id_row.set_value(self._selected_tensor_id)
        lines = [
            f"Shape: {shape_str}",
            f"Dtype: {tensor_desc.dtype}",
            f"Dims: {dims_str}",
        ]
        # No chunk grid here. These entries come from the catalog listing,
        # and the transfer grid belongs to the tensor GetFlightInfo binds,
        # which is authoritative for it (biopb/biopb#684, biopb/biopb#812).
        # This used to be a `if tensor_desc.chunk_shape:` row, unreachable
        # since the listing has always left the field empty; the catalog
        # struct no longer has one to test (biopb/biopb#1032).
        self._metadata_label.setText("\n".join(lines))
        self._metadata_pane.setVisible(True)

    def _on_filter_text_changed(self, _text: str):
        """Handle filter text change with debounce."""
        self._filter_timer.start(300)  # 300ms debounce

    def _apply_filter(self):
        """Apply the current filter to the tree."""
        query = self._filter_input.text().strip().lower()

        if not query:
            # Clear filter; an answer still in flight is no longer wanted.
            self._search_generation += 1
            self._search_more = False
            self._update_search_chrome()
            self._build_and_display_tree()
            return

        if self._use_server_query and self._client:
            # Server-side SQL query for large catalogs
            self._apply_server_filter(query)
        else:
            # Client-side filter
            self._search_generation += 1
            self._search_more = False
            self._update_search_chrome()
            self._apply_client_filter(query)

    def _apply_server_filter(self, query: str):
        """Ask the server for the matches, off the GUI thread. The tree stays as
        it is until the answer lands."""
        self._search_generation += 1
        worker = _SearchWorker(self._client, query, self._search_generation)
        self._search_worker = worker
        self._search_retain.add(worker)
        worker.done.connect(self._on_search_done)
        worker.failed.connect(self._on_search_failed)
        worker.finished.connect(lambda w=worker: self._search_retain.discard(w))
        self._update_search_chrome(searching=True)
        worker.start()

    def _on_search_done(self, generation: int, ids, more: bool):
        if generation != self._search_generation:
            return  # superseded while in flight
        self._search_more = more
        self._update_search_chrome()
        self._build_and_display_tree(filtered_ids=ids)

    def _on_search_failed(self, generation: int):
        if generation != self._search_generation:
            return
        # Fall back to filtering the listing we already hold.
        self._search_more = False
        self._update_search_chrome()
        self._apply_client_filter(self._filter_input.text().strip().lower())

    def _update_search_chrome(self, searching: bool = False):
        """The search box's placeholder and the line under it."""
        if not self._use_server_query:
            self._filter_input.setPlaceholderText("Search sources...")
            self._search_status.setVisible(False)
            return
        self._filter_input.setPlaceholderText("Search (SQL filter)")
        text = f"{len(self._sources):,} sources • Server-side filter"
        if searching:
            text += " • Searching…"
        elif self._search_more:
            text += f" • First {SERVER_QUERY_LIMIT:,} matches shown, refine the search"
        self._search_status.setText(text)
        self._search_status.setVisible(True)

    def _apply_client_filter(self, query: str):
        """Apply client-side filter."""
        matching_ids: Set[str] = set()
        for src in self._sources.values():
            hay = f"{src.source_id} {src.source_url} {src.source_type}".lower()
            if query in hay:
                matching_ids.add(src.source_id)

        self._build_and_display_tree(filtered_ids=matching_ids)

    def _add_to_viewer(self):
        """Add selected tensor as dask array to viewer."""
        self._clear_error()

        if self._client is None:
            self._show_error("Not connected to server")
            return

        if not self._selected_source_id or not self._selected_tensor_id:
            self._show_error("No tensor selected")
            return

        src = self._sources.get(self._selected_source_id)
        if not src:
            self._show_error("Source not found")
            return

        # Find tensor descriptor
        tensor_desc = next(
            (t for t in src.tensors if t.array_id == self._selected_tensor_id),
            None,
        )
        if not tensor_desc:
            self._show_error("Tensor descriptor not found")
            return

        try:
            # Show busy cursor during loading
            QApplication.setOverrideCursor(Qt.BusyCursor)

            # Build layer name: source_url.stem[/tensor_short_name]
            url_parts = _get_path_parts(src.source_url)
            stem = url_parts[-1] if url_parts else self._selected_source_id

            sole = _sole_image(src)
            if sole is not None and sole.array_id == self._selected_tensor_id:
                layer_name = stem
            else:
                tensor_name = _tensor_short_name(self._selected_tensor_id)
                layer_name = f"{stem}/{tensor_name}"

            # Shared build-pyramid -> wrap -> OME scale -> add_image pipeline
            # (also used by the MCP add_tensor).
            add_tensor_layer(
                self._viewer,
                self._client,
                self._selected_source_id,
                self._selected_tensor_id,
                tensor_desc,
                name=layer_name,
                compute_scheduler=self._compute_scheduler,
            )
            logger.info(
                "Added tensor layer '%s' from source '%s'",
                layer_name,
                self._selected_source_id,
            )

        except Exception:
            self._report_failure("Load failed", "Failed to load tensor")
            logger.exception(
                "Failed to get tensor %s from %s",
                self._selected_tensor_id,
                self._selected_source_id,
            )
        finally:
            QApplication.restoreOverrideCursor()
