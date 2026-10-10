"""Tests for the TensorBrowserWidget connect flow.

The widget connects its ``Connection`` and lists the catalog on a worker
thread, then renders the outcome on the Qt main thread via the
``_connect_done`` signal. These tests drive that flow deterministically: the
worker thread is *captured* rather than really spawned, so the test runs it
explicitly and can assert both the in-flight ("Connecting…") and completed
states. The connection and the source list are fakes.

The widget only stores the viewer and hands it to ``add_tensor_layer``, so a
stand-in viewer is used: building a real napari viewer (a Qt/OpenGL window) for
every test was slow and segfaulted intermittently inside napari on CI.
"""

from unittest.mock import MagicMock

import pytest


class TestGetPathParts:
    """source_url -> tree path parts, incl. remote authority roots (biopb/biopb#297)."""

    @staticmethod
    def _parts(url):
        from biopb_napari_widget.tensor_browser._widget import _get_path_parts

        return _get_path_parts(url)

    def test_local_file_url_is_just_its_path(self):
        assert self._parts("file:///home/jiyu/data/img.tif") == [
            "home",
            "jiyu",
            "data",
            "img.tif",
        ]

    def test_bare_posix_path(self):
        assert self._parts("/data/cells/img.tif") == ["data", "cells", "img.tif"]

    def test_remote_grpc_endpoint_is_the_root(self):
        assert self._parts("grpc://mantis-060:8815/labs/Yu/exp1/img.ome.tif") == [
            "grpc://mantis-060:8815",
            "labs",
            "Yu",
            "exp1",
            "img.ome.tif",
        ]

    def test_remote_alias_endpoint_root(self):
        assert self._parts("grpc://lab/data/x.tif") == ["grpc://lab", "data", "x.tif"]

    def test_s3_bucket_nests_under_endpoint(self):
        assert self._parts("s3://bucket/key/img.zarr") == [
            "s3://bucket",
            "key",
            "img.zarr",
        ]

    def test_windows_drive_letter_dropped(self):
        assert self._parts("file:///C:/Users/me/img.tif") == ["Users", "me", "img.tif"]

    def test_empty_url(self):
        assert self._parts("") == []

    def test_dnd_single_source_strips_scheme_to_own_root(self):
        # A drag-dropped source's "dnd://" origin scheme is stripped for display,
        # so it renders under a clean top-level root (same as a scheme-less
        # re-root), not a literal "dnd://exp.zarr" node.
        assert self._parts("dnd://exp.zarr") == ["exp.zarr"]

    def test_dnd_folder_children_nest_under_stripped_root(self):
        assert self._parts("dnd://my_experiment/sub/b.tif") == [
            "my_experiment",
            "sub",
            "b.tif",
        ]

    def test_dnd_basename_with_colon_not_misparsed_as_port(self):
        # String-strip (not urlparse) so a basename with a colon can't misparse
        # as a netloc/port.
        assert self._parts("dnd://exp:2.zarr") == ["exp:2.zarr"]


class TestBuildTreeDroppedTagging:
    """_build_tree tags a drag-dropped branch's root node so the UI shows [x]."""

    @staticmethod
    def _src(source_id, source_url):
        from biopb_napari_widget._catalog import CatalogSource, CatalogTensor

        return CatalogSource(
            source_id=source_id,
            source_url=source_url,
            tensors=(CatalogTensor(array_id=source_id, shape=(8, 8), dtype="uint8"),),
        )

    @staticmethod
    def _tree(sources):
        from biopb_napari_widget.tensor_browser._widget import _build_tree

        return _build_tree({s.source_id: s for s in sources})

    def test_single_file_drop_leaf_is_tagged(self):
        root = self._tree([self._src("s1", "dnd://exp.zarr")])
        (leaf,) = root.children
        assert leaf.node_type == "source"
        assert leaf.dropped is True
        assert leaf.remove_root == "dnd://exp.zarr"

    def test_folder_drop_top_folder_is_tagged(self):
        root = self._tree(
            [
                self._src("a", "dnd://my_experiment/a.zarr"),
                self._src("b", "dnd://my_experiment/sub/b.zarr"),
            ]
        )
        (folder,) = root.children
        assert folder.node_type == "folder"
        assert folder.dropped is True
        assert folder.remove_root == "dnd://my_experiment"
        # Children (the actual sources) are NOT individually tagged.
        assert all(not _leaf_dropped(c) for c in folder.children)

    def test_dropped_root_survives_path_flattening(self):
        # A drop whose only content is nested gets its single-child folders
        # flattened; the dropped tag + remove_root must ride along on the merged
        # node (flatten mutates the node in place, preserving the attributes).
        root = self._tree([self._src("x", "dnd://exp/sub/deep/img.zarr")])
        (node,) = root.children
        assert node.dropped is True
        assert node.remove_root == "dnd://exp"
        assert node.name.startswith("exp")  # flattened, e.g. "exp/sub/deep"

    def test_configured_source_is_not_tagged(self):
        root = self._tree([self._src("c", "file:///data/cells/img.zarr")])
        assert all(not n.dropped for n in _walk(root))


def _leaf_dropped(node):
    return getattr(node, "dropped", False)


def _walk(node):
    for child in node.children:
        yield child
        yield from _walk(child)


@pytest.fixture
def widget(qapp, monkeypatch):
    from qtpy.QtCore import QTimer

    from biopb_napari_widget.tensor_browser import _widget as widget_mod
    from biopb_napari_widget.tensor_browser._widget import TensorBrowserWidget

    viewer = MagicMock(name="viewer")
    conn = MagicMock()
    conn.url = "grpc://localhost:8815"
    # Default outcome: a connect that resolved to "not connected" (down). Tests
    # that exercise a successful connect use _connected_with.
    conn.client = None
    conn.last_message = ""
    conn.connect.return_value = False

    # Capture connect workers instead of spawning real threads so the tests run
    # them explicitly (and can assert the in-flight state before completion).
    workers = []

    class _FakeThread:
        def __init__(self, target=None, args=(), name=None, daemon=None):
            self._run = lambda: target(*args)

        def start(self):
            workers.append(self._run)

    monkeypatch.setattr(widget_mod.threading, "Thread", _FakeThread)
    # Neutralize the auto-connect-on-construction tick — the tests drive connect
    # explicitly for determinism.
    monkeypatch.setattr(QTimer, "singleShot", lambda *a, **k: None)

    w = TensorBrowserWidget(viewer, connection=conn)
    workers.clear()  # the source watcher's thread, captured at construction
    listing = MagicMock()
    listing.sources = {}
    listing.use_server_query = False
    # Default: server is not mid-scan, so an empty catalog renders as a genuine
    # "no sources" error (progressive-discovery indexing case is opt-in per test).
    listing.scan_in_progress.return_value = False
    listing.scan_source_count.return_value = 0
    w._list = listing
    # Isolate the render from tree building (which needs real descriptors).
    w._build_and_display_tree = MagicMock()
    return w, conn, workers


def _connected_with(w, sources, *, use_server_query=False):
    """Make the next connect succeed and list *sources*."""

    def _connect(url=None, token=None):
        w._conn.client = MagicMock()
        return True

    def _refresh():
        w._list.sources = sources
        w._list.use_server_query = use_server_query
        return sources

    w._conn.connect.side_effect = _connect
    w._list.refresh.side_effect = _refresh


class TestTreeLayoutStability:
    """The tree pins its horizontal scrollbar off so a content-width change on
    an otherwise-unchanged refresh can't toggle the bar and shift rows
    vertically on non-overlay-scrollbar platforms (biopb/biopb#367)."""

    def test_horizontal_scrollbar_pinned_off(self, widget):
        from qtpy.QtCore import Qt

        w, _, _ = widget
        assert w._tree_widget.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


class TestConnect:
    def test_shows_connecting_then_builds_tree(self, widget):
        w, conn, workers = widget
        _connected_with(w, {"a": object()})

        w._auto_connect()

        # In flight: status shown, button disabled, worker captured not yet run.
        assert w._connecting is True
        assert not w._connect_button.isEnabled()
        assert w._message_level == "busy"  # "Connecting…" is an ongoing state
        assert "Connecting" in w._message_label.text()
        assert len(workers) == 1
        w._build_and_display_tree.assert_not_called()

        workers.pop(0)()  # run the worker -> connect + list + render

        conn.connect.assert_called_once()
        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert w._message_label.isHidden()  # status cleared once connected
        assert w._connect_button.isEnabled()
        assert w._connecting is False

    def test_down_shows_error_no_prompt(self, widget):
        w, conn, workers = widget
        # connect tried, failed, recorded the friendly reason. There is no
        # dialog anymore — the failure just renders inline.
        conn.last_message = (
            "Cannot reach tensor server at grpc://localhost:8815 — is it running?"
        )

        w._auto_connect()
        workers.pop(0)()

        conn.connect.assert_called_once()
        assert w._message_level == "error"
        assert not w._message_label.isHidden()
        assert "Cannot reach" in w._message_label.text()
        assert not w._refresh_button.isEnabled()
        assert w._connecting is False
        assert w._connect_button.isEnabled()

    def test_down_uses_generic_message_without_last_message(self, widget):
        w, conn, workers = widget
        conn.last_message = ""  # nothing recorded -> generic fallback

        w._auto_connect()
        workers.pop(0)()

        # The fallback names no endpoint: without a recorded reason there is no
        # resolved address to name either (#628).
        assert w._message_level == "error"
        assert "Could not reach the biopb data plane" in w._message_label.text()

    def test_empty_catalog_shows_error(self, widget):
        w, conn, workers = widget
        _connected_with(w, {})  # connected, but no sources

        w._auto_connect()
        workers.pop(0)()

        assert w._message_level == "error"
        assert "No sources found" in w._message_label.text()
        assert not w._refresh_button.isEnabled()
        w._build_and_display_tree.assert_not_called()

    def test_empty_catalog_while_indexing_shows_status(self, widget):
        # Progressive discovery: SERVING but the scan is still running. An empty
        # catalog is "not done indexing yet", not an error -- show grey status,
        # keep Refresh enabled (more sources are coming), no error label.
        w, conn, workers = widget
        _connected_with(w, {})
        w._list.scan_in_progress.return_value = True
        w._list.scan_source_count.return_value = 7

        w._auto_connect()
        workers.pop(0)()

        # One pane, showing a sticky "busy" status -- not an error.
        assert w._message_level == "busy"
        assert not w._message_label.isHidden()
        assert "Indexing" in w._message_label.text()
        assert "7 sources so far" in w._message_label.text()
        assert w._refresh_button.isEnabled()
        w._build_and_display_tree.assert_not_called()

    def test_large_catalog_enables_sql_filter(self, widget):
        w, conn, workers = widget
        _connected_with(w, {"a": object()}, use_server_query=True)

        w._auto_connect()
        workers.pop(0)()

        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert "SQL filter" in w._filter_input.placeholderText()

    def test_connect_button_asks_the_control_again(self, widget):
        w, conn, workers = widget
        # While a control may answer, nothing is typed (#628): the fields stay
        # hidden, and a typed value would not be used anyway.
        assert w._manual_panel.isHidden()
        w._url_input.setText("grpc://typed:1")

        w._on_connect_clicked()
        workers.pop(0)()

        conn.connect.assert_called_once_with(None, None)

    def test_no_control_offers_a_url_and_token(self, widget):
        w, conn, workers = widget
        conn.url = None  # what a connect that found no control leaves behind
        conn.last_message = "No biopb control plane is running"

        w._auto_connect()
        workers.pop(0)()

        assert not w._manual_panel.isHidden()
        assert not w._advanced_panel.isHidden()
        assert "No biopb control" in w._message_label.text()

    def test_a_typed_url_is_dialed_with_the_typed_token(self, widget):
        w, conn, workers = widget
        w._manual = True
        w._url_input.setText(" grpc://typed:1 ")
        w._token_input.setText("tok")

        w._on_connect_clicked()
        workers.pop(0)()

        conn.connect.assert_called_once_with("grpc://typed:1", "tok")


class TestConnectionSummary:
    def test_advanced_panel_collapsed_by_default(self, widget):
        w, _conn, _workers = widget
        # The full connection controls start hidden behind the summary line.
        # (isHidden reflects the explicit visibility flag even when the top-level
        # widget is never shown, as in these headless tests.)
        assert w._advanced_panel.isHidden()
        assert not w._advanced_expanded
        # The collapsed caret is part of the (clickable) summary line.
        assert "▸" in w._status_summary.text()

    def test_clicking_summary_reveals_and_hides_panel(self, widget):
        w, _conn, _workers = widget
        w._toggle_advanced()  # what the summary line's mousePressEvent calls
        assert not w._advanced_panel.isHidden()
        assert w._advanced_expanded
        assert "▾" in w._status_summary.text()
        w._toggle_advanced()
        assert w._advanced_panel.isHidden()
        assert not w._advanced_expanded
        assert "▸" in w._status_summary.text()

    def test_summary_shows_url_and_state_across_lifecycle(self, widget):
        w, conn, workers = widget
        _connected_with(w, {"a": object()})

        # Before connecting: disconnected.
        assert conn.url in w._status_summary.text()
        assert "disconnected" in w._status_summary.text()

        w._auto_connect()
        assert "connecting" in w._status_summary.text()  # in flight

        workers.pop(0)()  # run worker -> connected
        assert "connected" in w._status_summary.text()
        assert "disconnected" not in w._status_summary.text()

    def test_summary_escapes_url_for_rich_text(self, widget):
        w, conn, _workers = widget
        # The summary is a rich-text QLabel; a URL with markup-significant chars
        # must be escaped, not injected raw.
        conn.url = "grpc://h?a=1&b=<x>"
        w._update_status_summary()
        text = w._status_summary.text()
        assert "&amp;" in text and "&lt;x&gt;" in text
        assert "<x>" not in text

    def test_stale_generation_is_dropped(self, widget):
        w, conn, workers = widget
        _connected_with(w, {"a": object()})

        w._auto_connect()  # gen 1, worker captured
        stale_worker = workers.pop(0)
        w._auto_connect()  # gen 2 supersedes; new worker captured
        workers.pop(0)()  # gen 2 completes and renders
        w._build_and_display_tree.assert_called_once()

        # The superseded (gen 1) worker finishing now must NOT re-render.
        w._build_and_display_tree.reset_mock()
        stale_worker()
        w._build_and_display_tree.assert_not_called()

    def test_worker_signals_completion_even_if_listing_raises(self, widget):
        w, conn, workers = widget
        # The worker must still signal completion (and not die) if the listing
        # after a connect raises.
        _connected_with(w, {})
        w._list.refresh.side_effect = RuntimeError("boom")

        w._auto_connect()
        workers.pop(0)()  # must not raise

        assert w._connecting is False
        assert w._connect_button.isEnabled()
        assert w._message_level == "error"
        assert not w._message_label.isHidden()


class TestRefreshFailure:
    def test_failed_refresh_marks_disconnected_and_updates_indicator(self, widget):
        w, conn, _workers = widget
        # Start from a connected state, then make the re-list blow up (server
        # gone): the shared client is dropped so the indicator says so.
        conn.client = MagicMock()
        w._list.refresh.side_effect = RuntimeError("unreachable")

        w._refresh()

        assert conn.client is None
        assert not w._refresh_button.isEnabled()
        assert w._message_level == "error"
        assert "lost connection" in w._message_label.text().lower()
        assert "disconnected" in w._status_summary.text()

    def test_render_error_does_not_mark_disconnected(self, widget):
        w, conn, _workers = widget
        # The server answered fine (refresh returned sources), but building the
        # tree blows up -- a client-side bug, not a lost server. The connection
        # must stay up and the indicator must not flip to disconnected; the
        # error is reported without dropping the client (that is scoped to the
        # re-list call, not the render).
        client = conn.client = MagicMock()
        w._list.refresh.return_value = {"a": object()}
        w._build_and_display_tree.side_effect = RuntimeError("render boom")

        w._refresh()

        assert conn.client is client
        assert w._message_level == "error"
        assert "lost connection" not in w._message_label.text().lower()


class TestMessagePane:
    """The unified bottom pane's level + auto-clear lifecycle (biopb/biopb#312)."""

    def test_info_self_clears_while_busy_and_error_are_sticky(self, widget):
        w, _conn, _workers = widget

        # A one-shot outcome ("added N") is info and arms the auto-clear timer.
        w._show_status("added 3 sources")
        assert w._message_level == "info"
        assert not w._message_label.isHidden()
        assert w._message_timer.isActive()

        # An ongoing state is a sticky "busy" status -- no timer.
        w._show_status("Indexing…", sticky=True)
        assert w._message_level == "busy"
        assert not w._message_timer.isActive()

        # An error is sticky (no timer) and replaces the status.
        w._show_error("boom")
        assert w._message_level == "error"
        assert "boom" in w._message_label.text()
        assert not w._message_timer.isActive()

    def test_clears_are_scoped_to_their_own_level(self, widget):
        w, _conn, _workers = widget

        # _clear_status must not wipe a live error...
        w._show_error("boom")
        w._clear_status()
        assert w._message_level == "error"
        assert not w._message_label.isHidden()

        # ...and _clear_error must not wipe a live status.
        w._show_status("Indexing…", sticky=True)
        w._clear_error()
        assert w._message_level == "busy"
        assert not w._message_label.isHidden()

        # Each clear still works on its own level.
        w._clear_status()
        assert w._message_level is None
        assert w._message_label.isHidden()

    def test_auto_clear_hides_and_resets(self, widget):
        w, _conn, _workers = widget
        w._show_status("added 3 sources")  # info -> timer armed
        assert w._message_timer.isActive()

        w._clear_message()  # what the timer's timeout invokes
        assert w._message_level is None
        assert w._message_label.isHidden()
        assert w._message_label.text() == ""
        assert not w._message_timer.isActive()


class TestSourcesChangedGuard:
    """The background source watcher's re-render is suppressed mid-connect."""

    def test_skipped_while_connecting(self, widget):
        w, conn, workers = widget
        conn.client = MagicMock()
        w._connecting = True
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        # A connect in flight owns the repaint; don't fight it.
        w._apply_filter.assert_not_called()

    def test_renders_when_idle(self, widget):
        w, conn, workers = widget
        conn.client = MagicMock()
        w._connecting = False
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        w._apply_filter.assert_called_once()

    def test_skipped_when_disconnected(self, widget):
        w, conn, workers = widget
        conn.client = None
        w._connecting = False
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        w._apply_filter.assert_not_called()


class TestRowShape:
    @pytest.mark.parametrize(
        "shape,expect",
        [
            ([1, 1, 5, 512, 512], "5×512×512"),
            ([1, 3, 512, 512], "3×512×512"),
            ([512, 512], "512×512"),
            ([5, 1, 512], "5×1×512"),  # only *leading* singletons go
            ([1, 1, 1], "1"),
        ],
    )
    def test_leading_singletons_are_squeezed(self, shape, expect):
        from biopb_napari_widget.tensor_browser._widget import _row_shape

        assert _row_shape(shape) == expect

    def test_row_keeps_the_shape_as_a_suffix_not_in_its_text(self, widget):
        from biopb_napari_widget.tensor_browser._widget import _SUFFIX_ROLE, _TreeNode

        w, _, _ = widget
        src = _source("a", tensors=["a"])
        w._add_tree_node(
            w._tree_widget,
            _TreeNode(
                node_id="a", name="a.zarr", node_type="source", depth=0, source=src
            ),
        )
        item = w._tree_widget.topLevelItem(0)
        assert item.text(0) == "a.zarr"
        assert item.data(0, _SUFFIX_ROLE) == "8×8"
        assert "[" not in item.toolTip(0)


class TestCloudGlyph:
    """A `needs_recall` source row carries a cloud glyph; every source row has
    the same icon slot so the names stay aligned."""

    def _row(self, w, name, **kw):
        from biopb_napari_widget.tensor_browser._widget import _TreeNode

        src = _source(name, tensors=[], is_resolved=False, **kw)
        node = _TreeNode(
            node_id=name, name=name, node_type="source", depth=0, source=src
        )
        w._add_tree_node(w._tree_widget, node)
        return w._tree_widget.topLevelItem(w._tree_widget.topLevelItemCount() - 1)

    def test_glyph_only_for_needs_recall(self, widget):
        w, _, _ = widget
        cloud = self._row(w, "a", unresolved_reason="needs_recall")
        legacy = self._row(w, "b", unresolved_reason=None)
        pending = self._row(w, "c", unresolved_reason="pending")
        failed = self._row(w, "d", unresolved_reason="failed")
        assert "download" in cloud.toolTip(0)
        # An older server gives no reason; that still means a cloud file.
        assert "download" in legacy.toolTip(0)
        for row in (pending, failed):
            assert "download" not in row.toolTip(0)
        assert "indexed" in pending.toolTip(0)
        assert "indexed" not in cloud.toolTip(0)

    def test_icon_slot_is_the_same_size_for_every_source_row(self, widget):
        w, _, _ = widget
        cloud = self._row(w, "a", unresolved_reason="needs_recall")
        pending = self._row(w, "c", unresolved_reason="pending")
        plain = self._row(w, "d", unresolved_reason="failed")  # no glyph
        rows = (cloud, pending, plain)
        size = lambda it: it.icon(0).availableSizes()[0]  # noqa: E731
        assert all(not r.icon(0).isNull() for r in rows)
        assert size(cloud) == size(pending) == size(plain)
        # Only the plain row's slot is blank; a glyph draws something. (The two
        # glyphs are not compared with each other: a font may lack either.)
        image = lambda it: it.icon(0).pixmap(size(it)).toImage()  # noqa: E731
        assert image(cloud) != image(plain)
        assert image(pending) != image(plain)

    def test_glyph_gone_after_resolve(self, widget):
        w, _, _ = widget
        row = self._row(w, "a", unresolved_reason="needs_recall")
        assert "download" in row.toolTip(0)
        from biopb_napari_widget.tensor_browser._widget import _TreeNode

        src = _source("a", tensors=[], is_resolved=True)
        w._add_tree_node(
            w._tree_widget,
            _TreeNode(node_id="a", name="a", node_type="source", depth=0, source=src),
        )
        resolved = w._tree_widget.topLevelItem(1)
        assert "download" not in resolved.toolTip(0)


class TestRemoveButton:
    """`_add_tree_node` puts a remove [x] in column 1 for dropped roots only."""

    def _node(self, *, dropped, source_url="dnd://exp.zarr", name="exp.zarr"):
        from biopb_napari_widget._catalog import CatalogSource, CatalogTensor
        from biopb_napari_widget.tensor_browser._widget import _TreeNode

        src = CatalogSource(
            source_id="s",
            source_url=source_url,
            tensors=(CatalogTensor(array_id="s", shape=(10, 10), dtype="uint8"),),
        )
        node = _TreeNode(
            node_id="s", name=name, node_type="source", depth=0, source=src
        )
        if dropped:
            node.dropped = True
            node.remove_root = source_url
        return node

    def test_dropped_root_gets_remove_button(self, widget):
        from qtpy.QtWidgets import QPushButton

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(dropped=True))
        item = w._tree_widget.topLevelItem(0)
        assert isinstance(w._tree_widget.itemWidget(item, 1), QPushButton)

    def test_non_dropped_row_has_no_button(self, widget):
        w, _, _ = widget
        w._add_tree_node(
            w._tree_widget, self._node(dropped=False, source_url="/data/c.zarr")
        )
        item = w._tree_widget.topLevelItem(0)
        assert w._tree_widget.itemWidget(item, 1) is None

    def test_button_click_routes_to_remove_with_branch_root(self, widget, monkeypatch):
        w, _, _ = widget
        called = {}
        monkeypatch.setattr(
            w, "_on_remove_dropped", lambda r, n: called.update(root=r, name=n)
        )
        w._add_tree_node(w._tree_widget, self._node(dropped=True))
        item = w._tree_widget.topLevelItem(0)
        w._tree_widget.itemWidget(item, 1).click()
        assert called == {"root": "dnd://exp.zarr", "name": "exp.zarr"}

    def test_confirm_yes_starts_remove(self, widget, monkeypatch):
        from biopb_napari_widget.tensor_browser import _widget as m

        w, _, _ = widget
        monkeypatch.setattr(
            m.QMessageBox, "question", lambda *a, **k: m.QMessageBox.Yes
        )
        started = {}
        monkeypatch.setattr(
            w, "_start_remove", lambda r, n: started.update(root=r, name=n)
        )
        w._on_remove_dropped("dnd://exp", "exp")
        assert started == {"root": "dnd://exp", "name": "exp"}

    def test_confirm_no_does_not_remove(self, widget, monkeypatch):
        from biopb_napari_widget.tensor_browser import _widget as m

        w, _, _ = widget
        monkeypatch.setattr(m.QMessageBox, "question", lambda *a, **k: m.QMessageBox.No)
        w._start_remove = MagicMock()
        w._on_remove_dropped("dnd://exp", "exp")
        w._start_remove.assert_not_called()


class TestHidesEmptySources:
    """`_build_and_display_tree` drops a resolved source with nothing on it --
    it would only be a row that opens an empty list."""

    @staticmethod
    def _src(source_id, *, tensors, is_resolved=True):
        from biopb_napari_widget._catalog import CatalogSource, CatalogTensor

        return CatalogSource(
            source_id=source_id,
            source_url=f"{source_id}.zarr",
            is_resolved=is_resolved,
            tensors=tuple(
                CatalogTensor(array_id=t, shape=(8, 8), dtype="uint8") for t in tensors
            ),
        )

    @staticmethod
    def _render(widget, sources):
        from biopb_napari_widget.tensor_browser._widget import TensorBrowserWidget

        w, _, _ = widget
        w._list.sources = {s.source_id: s for s in sources}
        # The fixture stubs this method out; call the real one directly.
        TensorBrowserWidget._build_and_display_tree(w)
        return [
            w._tree_widget.topLevelItem(i).text(0)
            for i in range(w._tree_widget.topLevelItemCount())
        ]

    def test_a_resolved_empty_source_gets_no_row(self, widget):
        names = self._render(
            widget,
            [self._src("empty", tensors=[]), self._src("full", tensors=["full"])],
        )
        assert names == ["full.zarr"]

    def test_an_unresolved_source_still_gets_its_row(self, widget):
        # Its empty tensor list means "unknown", not "nothing" -- it still
        # needs a row so a viewer can resolve it.
        names = self._render(
            widget, [self._src("cloud", tensors=[], is_resolved=False)]
        )
        assert names == ["cloud.zarr"]


class TestRestoreSelection:
    """`_restore_selection` re-highlights the tracked row in a rebuilt tree (#191)."""

    def _source_node(self, source_id, tensors):
        from biopb_napari_widget._catalog import CatalogSource, CatalogTensor
        from biopb_napari_widget.tensor_browser._widget import _TreeNode

        src = CatalogSource(
            source_id=source_id,
            source_url=f"/data/{source_id}.zarr",
            tensors=tuple(
                CatalogTensor(array_id=tid, shape=(8, 8), dtype="uint8")
                for tid in tensors
            ),
        )
        return _TreeNode(
            node_id=source_id,
            name=f"{source_id}.zarr",
            node_type="source",
            depth=0,
            source=src,
        )

    def _build(self, w, *source_nodes):
        w._tree_widget.clear()
        for node in source_nodes:
            w._add_tree_node(w._tree_widget, node)

    def test_reselects_source_node_after_rebuild(self, widget):
        w, _, _ = widget
        self._build(
            w,
            self._source_node("a", ["a"]),
            self._source_node("b", ["b"]),
        )
        # No current item right after a rebuild.
        assert w._tree_widget.currentItem() is None

        w._selected_source_id = "b"
        w._selected_tensor_id = None
        w._restore_selection()

        current = w._tree_widget.currentItem()
        assert current is not None
        assert current.data(0, _user_role()) == "b"

    def test_reselects_tensor_child_when_field_selected(self, widget):
        from qtpy.QtCore import Qt

        w, _, _ = widget
        self._build(w, self._source_node("multi", ["multi/f0", "multi/f1"]))

        w._selected_source_id = "multi"
        w._selected_tensor_id = "multi/f1"
        w._restore_selection()

        current = w._tree_widget.currentItem()
        assert current is not None
        assert current.data(0, Qt.ItemDataRole.UserRole + 1) == "tensor"
        assert current.data(0, _user_role()) == "multi/f1"

    def test_no_selection_leaves_current_unset(self, widget):
        w, _, _ = widget
        self._build(w, self._source_node("a", ["a"]))
        w._selected_source_id = None
        w._restore_selection()
        assert w._tree_widget.currentItem() is None


def _user_role():
    from qtpy.QtCore import Qt

    return Qt.ItemDataRole.UserRole


def _source(
    source_id, *, tensors, source_type="", is_resolved=True, unresolved_reason=None
):
    """A catalog row. ``is_resolved`` is independent of ``tensors`` on purpose:
    the two are what biopb/biopb#1032 stopped conflating."""
    from biopb_napari_widget._catalog import CatalogSource, CatalogTensor

    return CatalogSource(
        source_id=source_id,
        source_url=f"/cloud/{source_id}.zarr",
        source_type=source_type,
        is_resolved=is_resolved,
        unresolved_reason=unresolved_reason,
        tensors=tuple(
            CatalogTensor(array_id=tid, shape=(8, 8), dtype="uint8") for tid in tensors
        ),
    )


class _Sig:
    """Minimal stand-in for a Qt signal: connect()/emit() on the calling thread."""

    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def emit(self, *args):
        for cb in self._cbs:
            cb(*args)


class _FakeProgress:
    """Non-blocking QProgressDialog stand-in (exec_ returns immediately)."""

    def __init__(self, *a, **k):
        self.closed = False
        self.label = ""
        self.canceled = _Sig()  # user Cancel button -> request_cancel

    def setWindowTitle(self, *a):
        pass

    def setLabelText(self, text):
        self.label = text

    setWindowModality = setMinimumDuration = setCancelButton = setValue = (
        setAutoClose
    ) = setAutoReset = lambda self, *a: None

    def close(self):
        self.closed = True

    def exec_(self):
        pass


class TestUnresolvedHelper:
    def test_reads_the_servers_flag(self):
        from biopb_napari_widget.tensor_browser._widget import _is_unresolved

        assert _is_unresolved(_source("c", tensors=[], is_resolved=False))
        assert not _is_unresolved(_source("c", tensors=["c"]))
        assert not _is_unresolved(_source("c", tensors=["c/a", "c/b"]))

    def test_resolved_but_empty_is_not_unresolved(self):
        """A source that resolved and had nothing readable in it. The old
        ``len(tensors) == 0`` proxy called this unresolved and offered a
        Resolve that could only succeed and change nothing
        (biopb/biopb#1032)."""
        from biopb_napari_widget.tensor_browser._widget import _is_unresolved

        assert not _is_unresolved(_source("c", tensors=[]))


class TestUnresolvedReasonHelpers:
    def test_only_a_cloud_or_unexplained_source_needs_consent(self):
        from biopb_napari_widget.tensor_browser._widget import _needs_recall

        def src(reason, resolved=False):
            return _source(
                "c", tensors=[], is_resolved=resolved, unresolved_reason=reason
            )

        assert _needs_recall(src("needs_recall"))
        assert _needs_recall(src(None))  # older server: is_resolved alone
        assert not _needs_recall(src("pending"))
        assert not _needs_recall(src("failed"))
        assert not _needs_recall(src(None, resolved=True))

    def test_badge_only_for_pending_and_failed(self):
        from biopb_napari_widget.tensor_browser._widget import _unresolved_badge

        def src(reason):
            return _source("c", tensors=[], is_resolved=False, unresolved_reason=reason)

        assert _unresolved_badge(src("pending")) == ""
        assert "failed" in _unresolved_badge(src("failed"))
        assert _unresolved_badge(src("needs_recall")) == ""
        assert _unresolved_badge(src(None)) == ""


class TestEmptySourceHelper:
    def test_resolved_with_nothing_on_it_is_empty(self):
        from biopb_napari_widget.tensor_browser._widget import _is_empty_source

        assert _is_empty_source(_source("c", tensors=[]))

    def test_not_empty_once_it_lists_a_tensor(self):
        from biopb_napari_widget.tensor_browser._widget import _is_empty_source

        assert not _is_empty_source(_source("c", tensors=["c"]))

    def test_never_empty_while_unresolved(self):
        """An unresolved source's empty list means "unknown", not "nothing" --
        it still needs its row so a viewer can resolve it."""
        from biopb_napari_widget.tensor_browser._widget import _is_empty_source

        assert not _is_empty_source(_source("c", tensors=[], is_resolved=False))


class TestResolveAction:
    """Double-click / context-menu on an unresolved source resolves it off-thread."""

    def _arm(self, widget, monkeypatch, *, accept, outcome, reason=None):
        """Wire a widget so _resolve_source runs without Qt threads/modals.

        ``accept`` chooses the warning-dialog answer; ``outcome`` is the fake
        worker's result: ("resolved", desc) or ("failed", message).
        """
        from biopb_napari_widget.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.client = MagicMock()
        w._list.sources = {
            "cloud_x": _source(
                "cloud_x", tensors=[], is_resolved=False, unresolved_reason=reason
            )
        }

        answer = widget_mod.QMessageBox.Ok if accept else widget_mod.QMessageBox.Cancel
        monkeypatch.setattr(
            widget_mod.QMessageBox, "warning", staticmethod(lambda *a, **k: answer)
        )
        monkeypatch.setattr(widget_mod, "QProgressDialog", _FakeProgress)

        started = {"n": 0}

        class _FakeWorker:
            def __init__(self, conn_, source_id):
                self.resolved = _Sig()
                self.failed = _Sig()
                self.finished = _Sig()
                self.cancelled = _Sig()
                self.progress = _Sig()

            def request_cancel(self):
                pass

            def start(self):
                started["n"] += 1
                kind, payload = outcome
                sig = getattr(self, kind)
                sig.emit(payload) if payload is not None else sig.emit()

            def deleteLater(self):
                pass

        monkeypatch.setattr(widget_mod, "_ResolveWorker", _FakeWorker)
        w._apply_filter = MagicMock()
        w._show_error = MagicMock()
        w._report_failure = MagicMock()
        return w, started

    def test_declined_warning_does_nothing(self, widget, monkeypatch):
        w, started = self._arm(
            widget, monkeypatch, accept=False, outcome=("resolved", object())
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 0  # no worker spawned
        w._apply_filter.assert_not_called()

    def test_accepted_resolves_then_repopulates(self, widget, monkeypatch):
        # A resolved descriptor: repopulate the tree.
        w, started = self._arm(
            widget,
            monkeypatch,
            accept=True,
            outcome=("resolved", _source("cloud_x", tensors=["cloud_x"])),
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 1  # resolve ran off-thread
        w._apply_filter.assert_called_once()  # tree repopulated from fresh catalog
        w._show_error.assert_not_called()
        # The resolved source is pinned as the selection so the rebuild re-selects
        # it and the user doesn't lose track of it (issue #191).
        assert w._selected_source_id == "cloud_x"
        assert w._selected_tensor_id is None

    @pytest.mark.parametrize("reason", ["pending", "failed"])
    def test_pending_or_failed_resolves_without_the_cloud_warning(
        self, widget, monkeypatch, reason
    ):

        w, started = self._arm(
            widget,
            monkeypatch,
            accept=False,  # a warning, if shown, would be declined
            outcome=("resolved", _source("cloud_x", tensors=["cloud_x"])),
            reason=reason,
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 1
        w._apply_filter.assert_called_once()

    def test_needs_recall_still_asks(self, widget, monkeypatch):
        w, started = self._arm(
            widget,
            monkeypatch,
            accept=False,
            outcome=("resolved", object()),
            reason="needs_recall",
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 0

    def test_failure_surfaces_error(self, widget, monkeypatch):
        # A failed user-initiated resolve reports via a modal box, not the
        # easily-missed inline pane (issue #206).
        w, _ = self._arm(
            widget, monkeypatch, accept=True, outcome=("failed", "offline")
        )
        w._resolve_source("cloud_x")
        w._report_failure.assert_called_once()
        assert "offline" in w._report_failure.call_args[0][1]
        w._show_error.assert_not_called()

    def test_cancelled_closes_quietly(self, widget, monkeypatch):
        # A user-cancelled resolve is not an error: no banner, no repopulate (the
        # server recall finishes + caches, so a later resolve coalesces).
        w, started = self._arm(
            widget, monkeypatch, accept=True, outcome=("cancelled", None)
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 1
        w._show_error.assert_not_called()
        w._apply_filter.assert_not_called()

    def test_overlapping_workers_are_each_retained(self, widget, monkeypatch):
        # Two in-flight workers must not clobber each other's only ref (which
        # would let a still-running QThread be GC'd / destroyed mid-run). We hold
        # them by thread lifetime in a set, discarded on `finished`.
        from biopb_napari_widget.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.client = MagicMock()
        w._list.sources = {"cloud_x": _source("cloud_x", tensors=[], is_resolved=False)}
        monkeypatch.setattr(
            widget_mod.QMessageBox,
            "warning",
            staticmethod(lambda *a, **k: widget_mod.QMessageBox.Ok),
        )
        monkeypatch.setattr(widget_mod, "QProgressDialog", _FakeProgress)

        made = []

        class _PendingWorker:
            """Starts but never emits resolved/failed/finished (stays in flight)."""

            def __init__(self, conn_, source_id):
                self.resolved = _Sig()
                self.failed = _Sig()
                self.finished = _Sig()
                self.cancelled = _Sig()
                self.progress = _Sig()
                made.append(self)

            def request_cancel(self):
                pass

            def start(self):
                pass

            def deleteLater(self):
                pass

        monkeypatch.setattr(widget_mod, "_ResolveWorker", _PendingWorker)

        w._resolve_source("cloud_x")
        w._resolve_source("cloud_x")

        # Both workers are alive (neither dropped); discard happens on `finished`.
        assert len(made) == 2
        assert set(made) == w._resolve_workers
        for worker in made:  # finishing one removes only itself
            worker.finished.emit()
        assert w._resolve_workers == set()

    def test_double_click_routes_unresolved_to_resolve(self, widget, monkeypatch):
        from biopb_napari_widget.tensor_browser._widget import _TreeNode

        w, conn, _ = widget
        w._list.sources = {"cloud_x": _source("cloud_x", tensors=[], is_resolved=False)}
        w._add_tree_node(
            w._tree_widget,
            _TreeNode(
                node_id="cloud_x",
                name="cloud_x.zarr",
                node_type="source",
                depth=0,
                source=w._list.sources["cloud_x"],
            ),
        )
        item = w._tree_widget.topLevelItem(0)
        w._resolve_source = MagicMock()
        w._add_to_viewer = MagicMock()

        w._on_tree_item_double_clicked(item, 0)

        w._resolve_source.assert_called_once_with("cloud_x")
        w._add_to_viewer.assert_not_called()  # unresolved never hits the add path


class TestAddToViewer:
    """`_add_to_viewer` loads the selected tensor behind a busy cursor."""

    def _arm(self, widget):
        # `_client`/`_sources` are read-only views onto the connection.
        w, conn, _ = widget
        conn.client = MagicMock()
        w._list.sources = {"m": _source("m", tensors=["m"], source_type="zarr")}
        w._selected_source_id = "m"
        w._selected_tensor_id = "m"
        w._show_error = MagicMock()
        w._report_failure = MagicMock()
        return w

    def test_load_failure_reports_modally(self, widget, monkeypatch):
        # A failed view/load is user-initiated too, so it reports modally rather
        # than on the easily-missed inline pane (#206 consistency).
        from biopb_napari_widget.tensor_browser import _widget as widget_mod

        w = self._arm(widget)
        monkeypatch.setattr(
            widget_mod,
            "add_tensor_layer",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        w._add_to_viewer()
        w._report_failure.assert_called_once()
        assert "Failed to load tensor" in w._report_failure.call_args[0][1]
        w._show_error.assert_not_called()


class TestInfoPaneIsReadableOut:
    """The info pane exists to be read *out of* (biopb/biopb#972).

    `Tensor:` is the argument to `client.get_tensor(...)`, so the pane's job is
    not finished when it renders the identifier -- it is finished when the
    identifier can leave the window intact.
    """

    def _select(self, widget, source_id="ome-tiff_8cc0", tensor="Image:0"):
        w, conn, _ = widget
        array_id = f"{source_id}/{tensor}"
        w._list.sources = {source_id: _source(source_id, tensors=[array_id])}
        w._selected_source_id = source_id
        w._selected_tensor_id = array_id
        w._update_metadata_display()
        return w, source_id, array_id

    def test_the_identifiers_get_their_own_rows(self, widget):
        w, source_id, array_id = self._select(widget)

        assert w._source_id_row.value() == source_id
        assert w._tensor_id_row.value() == array_id
        # ...and are no longer buried in the block below them, or the copy
        # button would be copying one of two things the pane shows.
        assert "Source:" not in w._metadata_label.text()
        assert "Tensor:" not in w._metadata_label.text()
        assert "Shape:" in w._metadata_label.text()

    def test_every_line_can_be_selected(self, widget):
        from qtpy.QtCore import Qt

        w, _, _ = self._select(widget)

        for label in (
            w._metadata_label,
            w._source_id_row._label,
            w._tensor_id_row._label,
        ):
            flags = label.textInteractionFlags()
            assert flags & Qt.TextSelectableByMouse
            assert flags & Qt.TextSelectableByKeyboard

    def test_copy_puts_the_identifier_on_the_clipboard(self, widget):
        from qtpy.QtWidgets import QApplication

        w, source_id, array_id = self._select(widget)

        w._tensor_id_row._button.click()
        assert QApplication.clipboard().text() == array_id

        w._source_id_row._button.click()
        assert QApplication.clipboard().text() == source_id

    def test_copy_takes_the_value_not_the_rendered_line(self, widget):
        # The label carries a "Tensor: " prefix and may wrap; copying what is
        # drawn would hand over neither the id nor anything that fails loudly.
        w, _, array_id = self._select(widget)

        assert w._tensor_id_row.value() == array_id
        assert w._tensor_id_row._label.text() == f"Tensor: {array_id}"

    def test_a_copy_says_what_it_copied(self, widget):
        w, _, array_id = self._select(widget)
        w._show_status = MagicMock()

        w._tensor_id_row._button.click()

        # Named, because the two rows are one line apart and the outcome does
        # not otherwise say which button was pressed.
        w._show_status.assert_called_once_with(f"Copied {array_id}")

    def test_the_whole_pane_hides_together(self, widget):
        w, _, _ = self._select(widget)
        assert w._metadata_pane.isVisible() or w._metadata_pane.isVisibleTo(w)

        w._selected_tensor_id = None
        w._update_metadata_display()

        assert not w._metadata_pane.isVisibleTo(w)


class TestGroupTensors:
    """A source's tensors with its label sets filed under their image.

    The catalog lists a set as an ordinary tensor (biopb/biopb#1059), so
    without this an ordinary image that gained an ``@ome`` set reads as a
    two-tensor source: no shape badge, and no double-click open.
    """

    @staticmethod
    def _group(*array_ids):
        from biopb_napari_widget.tensor_browser._widget import _group_tensors

        tensors = [MagicMock(array_id=a, shape=[8, 8]) for a in array_ids]
        return _group_tensors(tensors)

    def test_a_set_does_not_count_as_a_tensor_of_the_source(self):
        groups = self._group("src0", "src0/@labels/@ome")
        assert len(groups) == 1
        assert groups[0].image.array_id == "src0"
        assert [s.array_id for s in groups[0].label_sets] == ["src0/@labels/@ome"]

    def test_sets_file_under_their_own_image(self):
        groups = self._group(
            "src0/A", "src0/B", "src0/B/@labels/nuclei", "src0/A/@labels/cells"
        )
        assert [g.image.array_id for g in groups] == ["src0/A", "src0/B"]
        assert [s.array_id for s in groups[0].label_sets] == ["src0/A/@labels/cells"]
        assert [s.array_id for s in groups[1].label_sets] == ["src0/B/@labels/nuclei"]

    def test_images_keep_the_order_the_server_listed_them(self):
        # The server puts image tensors first on purpose: tensors[0] is the
        # source's picture.
        groups = self._group("src0/Z", "src0/A")
        assert [g.image.array_id for g in groups] == ["src0/Z", "src0/A"]

    def test_sets_are_sorted_by_id(self):
        groups = self._group("src0", "src0/@labels/nuclei", "src0/@labels/@ome")
        assert [s.array_id for s in groups[0].label_sets] == [
            "src0/@labels/@ome",
            "src0/@labels/nuclei",
        ]

    def test_an_orphan_set_keeps_its_own_row(self):
        # Should not happen -- the server registers a set on its parent -- but a
        # tensor the catalog lists and the tree hides is the worse failure.
        groups = self._group("src0/@labels/nuclei")
        assert [g.image.array_id for g in groups] == ["src0/@labels/nuclei"]
        assert groups[0].label_sets == []


class TestSoleImage:
    """Whether a source opens as a single layer -- sets never count."""

    @staticmethod
    def _sole(*array_ids):
        from biopb_napari_widget.tensor_browser._widget import _sole_image

        src = MagicMock()
        src.tensors = [MagicMock(array_id=a, shape=[8, 8]) for a in array_ids]
        return _sole_image(src)

    def test_an_image_with_sets_is_still_sole(self):
        sole = self._sole("src0", "src0/@labels/@ome", "src0/@labels/nuclei")
        assert sole.array_id == "src0"

    def test_two_images_have_no_sole(self):
        assert self._sole("src0/A", "src0/B") is None

    def test_no_tensors(self):
        assert self._sole() is None


class TestSearch:
    """Search on a large catalog follows biopb's web viewer: the query runs off the
    GUI thread, is capped, drops superseded answers, and says what it did."""

    def _real_tree(self, w, sources):
        from biopb_napari_widget.tensor_browser._widget import TensorBrowserWidget

        w._list.sources = sources
        w._build_and_display_tree = lambda **kw: (
            TensorBrowserWidget._build_and_display_tree(w, **kw)
        )

    def _rows(self, w):
        tree = w._tree_widget
        out = []

        def walk(item):
            out.append(item.text(0))
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(tree.topLevelItemCount()):
            walk(tree.topLevelItem(i))
        return out

    def test_sql_is_capped_ordered_and_escaped(self):
        from biopb_napari_widget.tensor_browser._widget import (
            SERVER_QUERY_LIMIT,
            _search_sql,
        )

        sql = _search_sql("it's_50%")
        assert f"LIMIT {SERVER_QUERY_LIMIT + 1}" in sql
        assert "ORDER BY source_url" in sql
        assert "it''s\\_50\\%" in sql
        assert sql.count("ESCAPE '\\'") == 3

    def test_worker_reports_ids_and_whether_more_matched(self, qapp):
        from biopb_napari_widget.tensor_browser._widget import (
            SERVER_QUERY_LIMIT,
            _SearchWorker,
        )

        client = MagicMock()
        client.query.return_value = [{"source_id": f"s{i}"} for i in range(5)]
        got = []
        worker = _SearchWorker(client, "q", 7)
        worker.done.connect(lambda g, ids, more: got.append((g, ids, more)))
        worker.run()
        assert got == [(7, {f"s{i}" for i in range(5)}, False)]

        client.query.return_value = [
            {"source_id": f"s{i}"} for i in range(SERVER_QUERY_LIMIT + 1)
        ]
        got.clear()
        worker.run()
        g, ids, more = got[0]
        assert len(ids) == SERVER_QUERY_LIMIT and more is True

    def test_worker_failure_is_reported(self, qapp):
        from biopb_napari_widget.tensor_browser._widget import _SearchWorker

        client = MagicMock()
        client.query.side_effect = RuntimeError("boom")
        failed = []
        worker = _SearchWorker(client, "q", 3)
        worker.failed.connect(failed.append)
        worker.run()
        assert failed == [3]

    def test_a_superseded_answer_is_dropped(self, widget):
        w, _, _ = widget
        w._search_generation = 5
        w._on_search_done(4, {"a"}, True)
        w._build_and_display_tree.assert_not_called()
        assert w._search_more is False
        w._on_search_done(5, {"a"}, True)
        w._build_and_display_tree.assert_called_once_with(filtered_ids={"a"})
        assert w._search_more is True

    def test_server_search_runs_on_a_worker_and_leaves_the_tree_alone(
        self, widget, monkeypatch
    ):
        from biopb_napari_widget.tensor_browser import _widget as widget_mod

        w, _, _ = widget
        w._list.use_server_query = True
        w._conn.client = MagicMock()
        started = []
        monkeypatch.setattr(
            widget_mod._SearchWorker, "start", lambda self: started.append(self)
        )
        w._filter_input.setText("abc")
        w._apply_filter()
        assert len(started) == 1
        assert w._search_generation == 1
        w._build_and_display_tree.assert_not_called()  # stays until the answer
        assert "Searching" in w._search_status.text()

        w._apply_filter()
        assert w._search_generation == 2  # the first answer is now stale

    def test_clearing_the_box_cancels_the_search(self, widget):
        w, _, _ = widget
        w._search_generation = 3
        w._search_more = True
        w._filter_input.setText("")
        w._apply_filter()
        assert w._search_generation == 4 and w._search_more is False
        w._build_and_display_tree.assert_called_once_with()

    def test_a_failed_search_falls_back_to_the_listing(self, widget):
        w, _, _ = widget
        w._list.sources = {"a": _source("a", tensors=["a"])}
        w._filter_input.setText("a")
        w._search_generation = 2
        w._on_search_failed(1)  # stale: ignored
        w._build_and_display_tree.assert_not_called()
        w._on_search_failed(2)
        w._build_and_display_tree.assert_called_once_with(filtered_ids={"a"})

    def test_status_line_only_on_a_large_catalog(self, widget):
        from biopb_napari_widget.tensor_browser._widget import SERVER_QUERY_LIMIT

        w, _, _ = widget
        w._list.sources = {"a": object(), "b": object()}
        w._list.use_server_query = False
        w._update_search_chrome()
        assert w._search_status.isHidden()
        w._list.use_server_query = True
        w._search_more = True
        w._update_search_chrome()
        assert not w._search_status.isHidden()
        text = w._search_status.text()
        assert "2 sources" in text and f"First {SERVER_QUERY_LIMIT:,}" in text

    def test_no_matches_is_an_empty_tree_not_the_whole_catalog(self, widget):
        w, _, _ = widget
        self._real_tree(w, {"a": _source("a", tensors=["a"])})
        w._build_and_display_tree(filtered_ids=set())
        assert w._tree_widget.topLevelItemCount() == 0
        w._build_and_display_tree()
        assert w._tree_widget.topLevelItemCount() == 1

    def test_only_the_top_level_opens_by_itself(self, widget):
        from biopb_napari_widget._catalog import CatalogSource, CatalogTensor

        w, _, _ = widget

        def src(sid, url):
            return CatalogSource(
                source_id=sid,
                source_url=url,
                tensors=(CatalogTensor(array_id=sid, shape=(8, 8), dtype="uint8"),),
            )

        self._real_tree(
            w,
            {
                "m": src("m", "/lab/exp1/plateA/m.tif"),
                "k": src("k", "/lab/exp1/plateB/k.tif"),
                "n": src("n", "/lab/x/n.tif"),
            },
        )
        w._build_and_display_tree(filtered_ids={"m", "k"})
        # The top-level folder is open; the folders below it are not, and the
        # non-matching branch is not built at all.
        opened = w._expanded_folders
        assert opened == {"/lab/exp1"}
        assert not any(r.startswith("n.tif") for r in self._rows(w))


class TestRoiAnnotations:
    """The ROI entry comes from the catalog's set names: absent when the server
    cannot say, one entry for one set, a submenu for several; picking one fetches
    just that set and adds Points / Shapes layers."""

    def _menu(self, w, sets):
        from qtpy.QtWidgets import QMenu

        w._list.roi_sets = sets
        menu = QMenu()
        w._add_roi_actions(menu, "a", "a")
        return menu

    def test_server_without_roi_support_has_no_entry(self, widget):
        w, _, _ = widget
        assert self._menu(w, None).actions() == []

    def test_a_tensor_without_sets_has_no_entry(self, widget):
        w, _, _ = widget
        assert self._menu(w, {"other": {"s": 1}}).actions() == []

    def test_one_set_is_one_entry(self, widget):
        w, _, _ = widget
        actions = self._menu(w, {"a": {"nuclei": 12}}).actions()
        assert len(actions) == 1
        assert "nuclei" in actions[0].text() and "12" in actions[0].text()

    def test_several_sets_make_a_submenu_and_mark_reserved_ones(self, widget):
        w, _, _ = widget
        menu = self._menu(w, {"a": {"nuclei": 12, "@ome": 3}})  # keep it alive
        (top,) = menu.actions()
        texts = [a.text() for a in top.menu().actions()]
        assert len(texts) == 2
        assert any("@ome" in t and "read-only" in t for t in texts)
        assert any("nuclei" in t and "read-only" not in t for t in texts)

    def test_picking_a_set_fetches_that_set(self, widget, monkeypatch):
        from biopb_napari_widget.tensor_browser import _widget as widget_mod

        w, _, _ = widget
        w._conn.client = MagicMock()
        started = []
        monkeypatch.setattr(
            widget_mod._RoiWorker, "start", lambda self: started.append(self)
        )
        monkeypatch.setattr(
            widget_mod.QApplication, "setOverrideCursor", lambda *a: None
        )
        menu = self._menu(w, {"a": {"x": 1, "y": 2}})
        menu.actions()[0].menu().actions()[1].trigger()  # menu stays referenced
        assert len(started) == 1 and started[0]._set_name == "y"

    def test_worker_fetches_by_array_id_and_set(self, qapp):
        from biopb_napari_widget.tensor_browser._widget import _RoiWorker

        client = MagicMock()
        got = []
        worker = _RoiWorker(client, "arr", "@ome")
        worker.done.connect(got.append)
        worker.run()
        client.list_rois.assert_called_once_with("arr", "@ome")
        assert len(got) == 1

    def _result(self, n=3, truncated=False):
        from biopb.image import ROI, Point, RoiAnnotation

        rois = [
            RoiAnnotation(
                roi_id=f"r{i}",
                array_id="a",
                set_name="nuclei",
                label="L",
                roi=ROI(point=Point(x=i + 1.0, y=2.0)),
            )
            for i in range(n)
        ]
        return MagicMock(rois=rois, truncated=truncated)

    def _fetched(self, w, result):
        w._list.sources = {"a": _source("a", tensors=["a"])}
        w._conn.client = MagicMock()
        w._conn.client.get_physical_scale.return_value = None
        w._report_failure = MagicMock()
        w._show_message = MagicMock()
        w._on_rois_fetched("a", "a", "nuclei", result)

    def test_a_fetched_set_becomes_a_layer(self, widget):
        w, _, _ = widget
        self._fetched(w, self._result())
        w._viewer.add_points.assert_called_once()
        assert w._viewer.add_points.call_args.kwargs["name"] == "nuclei"

    def test_an_empty_set_says_so(self, widget):
        w, _, _ = widget
        self._fetched(w, self._result(n=0))
        w._viewer.add_points.assert_not_called()
        assert "empty" in w._show_message.call_args.args[0]

    def test_truncation_is_reported(self, widget):
        w, _, _ = widget
        self._fetched(w, self._result(truncated=True))
        assert "truncated" in w._show_message.call_args.args[0]

    def test_a_failure_to_fetch_is_reported(self, widget):
        w, _, _ = widget
        w._report_failure = MagicMock()
        w._on_rois_failed("annotations disabled")
        w._report_failure.assert_called_once()
