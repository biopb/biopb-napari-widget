"""The catalog as the Tensor Browser shows it, and what keeps it current.

Qt-free, so it runs and tests without a display. It reads through the shared
:class:`biopb.tensor.Connection` and holds the one cache in the picture: the
list of sources the tree is drawn from.

Threading: the watcher re-lists from its own daemon thread and only ever
*rebinds* ``sources`` to a fresh dict, so a reader on the Qt thread sees the
whole old list or the whole new one.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Dict

from .._catalog import CatalogSource, source_from_row, sources_from_rows

logger = logging.getLogger(__name__)

#: Catalogs larger than this filter server-side, in SQL.
SERVER_QUERY_THRESHOLD = 1000

#: The watcher's poll interval backs off from the first to the second while
#: the source count holds, and snaps back on a change.
WATCH_MIN_INTERVAL_S = 2.0
WATCH_MAX_INTERVAL_S = 60.0

#: The columns ``_catalog`` decodes.
_BASE_COLUMNS = "source_id, source_url, source_type, is_resolved, tensors"


def _sources_sql(client) -> str:
    """The listing query. An SDK with ``source_row_columns`` adds
    ``unresolved_reason`` when the server's schema has it; an older one gets the
    base columns, and a row without a reason reads as a cloud source."""
    columns = _BASE_COLUMNS
    try:
        probed = client.source_row_columns()
    except Exception:  # noqa: BLE001 - absent or failing probe: base columns
        logger.debug("source_row_columns unavailable; using base columns")
    else:
        if isinstance(probed, str) and probed:
            columns = probed
    return f"SELECT {columns} FROM sources ORDER BY source_id"


#: One row per (tensor, set): what annotations the server holds. Reserved
#: (``@``) sets are in it -- the SQL surface does not hide them the way an
#: unqualified ``list_rois`` does.
_ROI_SETS_SQL = (
    "SELECT array_id, set_name, count(*) AS n FROM rois GROUP BY array_id, set_name"
)


def roi_sets_from_rows(rows) -> Dict[str, Dict[str, int]]:
    """``{array_id: {set_name: count}}`` from the ``rois`` aggregate rows."""
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        out.setdefault(row["array_id"], {})[row["set_name"]] = int(row["n"])
    return out


def _accepts_cloud(register) -> bool:
    try:
        params = inspect.signature(register).parameters
    except (TypeError, ValueError):
        return True  # cannot tell; let the SDK answer
    return "cloud" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class SourceList:
    """The sources the tree shows, re-listed when the server's count moves."""

    def __init__(self, connection) -> None:
        self._conn = connection
        self.sources: Dict[str, CatalogSource] = {}
        #: Which annotation sets each tensor has, ``{array_id: {set: count}}``,
        #: or ``None`` when the server has no queryable ``rois`` table (an older
        #: server, or annotations switched off) -- the ROI feature is then off.
        #: Re-read with the listing, so it is as current as the tree; another
        #: client's new annotations show on the next refresh.
        self.roi_sets: Dict[str, Dict[str, int]] | None = None
        self._roi_unsupported = False
        # The last health answer, so the paint thread can tell "still indexing"
        # from "empty" without a round trip.
        self.last_health: dict | None = None
        # Called from the watcher thread with the fresh dict after a re-list.
        self.on_changed = None
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()

    def _client(self):
        client = self._conn.client
        if client is None:
            raise RuntimeError("Not connected")
        return client

    @property
    def use_server_query(self) -> bool:
        return len(self.sources) > SERVER_QUERY_THRESHOLD

    def clear(self) -> None:
        self.sources = {}
        self.roi_sets = None
        self._roi_unsupported = False  # a new connection may be a newer server
        self.last_health = None

    def refresh(self) -> Dict[str, CatalogSource]:
        """Re-list the whole catalog from the server."""
        client = self._client()
        rows = client.query(_sources_sql(client), format="records")
        self.sources = {s.source_id: s for s in sources_from_rows(rows)}
        self._refresh_roi_sets(client)
        return self.sources

    def _refresh_roi_sets(self, client) -> None:
        """Read which sets each tensor has, if the server will say.

        A server without the table refuses the query; that is remembered for the
        connection rather than retried at every re-list. Any other failure (a
        dropped call) keeps the last answer and is tried again next time.
        """
        if self._roi_unsupported:
            return
        try:
            self.roi_sets = roi_sets_from_rows(
                client.query(_ROI_SETS_SQL, format="records")
            )
        except Exception as exc:  # noqa: BLE001 - the listing must not fail on this
            if "rois" in str(exc).lower():
                logger.info("Server has no queryable rois table; ROI loading is off")
                self._roi_unsupported = True
                self.roi_sets = None
            else:
                logger.debug("Reading the ROI sets failed", exc_info=True)

    def update_health(self) -> dict | None:
        """Ask the server how it is; kept for :meth:`scan_in_progress`."""
        health = self._client().health_check()
        self.last_health = health if isinstance(health, dict) else None
        return self.last_health

    def scan_in_progress(self) -> bool:
        h = self.last_health
        return bool(h.get("full_scan_in_progress")) if h else False

    def registration_pending(self) -> int:
        """Sources the server has found and not yet registered; 0 on a server
        whose health has no such field."""
        h = self.last_health
        try:
            return int(h.get("registration_pending") or 0) if h else 0
        except (TypeError, ValueError):
            return 0

    def scan_source_count(self) -> int:
        h = self.last_health
        try:
            return int(h.get("source_count") or 0) if h else 0
        except (TypeError, ValueError):
            return 0

    # --- the server's verbs, followed by a re-list ---------------------------

    def resolve(self, source_id: str, *, on_progress=None, should_cancel=None):
        """Resolve a cloud source (downloads it; call off the GUI thread)."""
        row = self._client().resolve_source(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )
        self.refresh()
        # The row this resolve committed, not the refreshed entry, which a
        # concurrent rescan could have re-registered underneath. It also lands in
        # the listing if the re-list still shows the source unresolved (a pending
        # source registers without moving the count), so the tree can repaint
        # from it.
        resolved = source_from_row(row)
        current = self.sources.get(resolved.source_id)
        if resolved.is_resolved and (current is None or not current.is_resolved):
            self.sources = {**self.sources, resolved.source_id: resolved}
        return resolved

    def add(self, path: str, *, cloud=False, on_progress=None, should_cancel=None):
        """Register *path*; *cloud* also registers its offline placeholders.

        ``cloud`` is only sent when set, and only to an SDK whose ``register_local_path``
        takes it: against an older one the drop proceeds without it, as it did
        before the keyword existed, rather than failing on an unknown argument.
        """
        client = self._client()
        kwargs = {}
        if cloud:
            if _accepts_cloud(client.register_local_path):
                kwargs["cloud"] = True
            else:
                logger.warning(
                    "SDK register_local_path has no `cloud`; adding without it"
                )
        result = client.register_local_path(
            path, on_progress=on_progress, should_cancel=should_cancel, **kwargs
        )
        self.refresh()
        return result

    def remove(self, root_url: str):
        result = self._client().deregister_local_path(root_url)
        self.refresh()
        return result

    # --- the watcher ---------------------------------------------------------

    def start_watch(
        self,
        min_interval: float = WATCH_MIN_INTERVAL_S,
        max_interval: float = WATCH_MAX_INTERVAL_S,
    ) -> None:
        """Re-list whenever the server's ``source_count`` or
        ``registration_pending`` changes.

        A catalog listed while the server was still indexing then fills itself
        in. A thread, not a ``QTimer``: ``health_check`` blocks. Idempotent.
        """
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(min_interval, max(max_interval, min_interval)),
            name="biopb-source-watch",
            daemon=True,
        )
        self._watch_thread.start()

    def stop_watch(self) -> None:
        self._watch_stop.set()

    def _watch_loop(self, min_interval: float, max_interval: float) -> None:
        """``last_count`` is the count the list was last reconciled against;
        ``None`` while disconnected, so the first connected poll compares
        against what was listed at connect and catches a mid-index partial.

        A registration changes neither the count nor the scan flag, so while the
        server reports ``registration_pending`` the pending figure is watched
        too, and the poll does not back off to a standstill. A source that
        gains tensors any other way is not caught.
        """
        interval = min_interval
        last_count: int | None = None
        last_pending = 0
        while not self._watch_stop.wait(interval):
            if self._conn.client is None:
                last_count = None
                interval = min_interval
                continue
            try:
                health = self.update_health()
            except Exception:  # noqa: BLE001 - transient; back off
                interval = min(interval * 2, max_interval)
                continue
            count = health.get("source_count") if health else None
            if count is None:
                interval = max_interval
                continue
            if last_count is None:
                last_count = len(self.sources)
            pending = self.registration_pending()
            if count != last_count or pending != last_pending:
                self._relist()
                last_count = count
                last_pending = pending
                interval = min_interval
            elif pending:
                interval = min_interval  # registering: keep the tree filling in
            else:
                interval = min(interval * 2, max_interval)

    def _relist(self) -> None:
        try:
            sources = self.refresh()
        except Exception:  # noqa: BLE001 - keep watching through a blip
            logger.exception("Source watch: re-list failed")
            return
        logger.info("Source watch: re-listed %d sources", len(sources))
        callback = self.on_changed
        if callback is not None:
            try:
                callback(sources)
            except Exception:  # noqa: BLE001
                logger.exception("Source watch: on_changed hook failed")
