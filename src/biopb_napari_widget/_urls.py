"""Whether a server URL points at this machine.

Owned here rather than imported from ``biopb`` (which is dropping
``is_local_url``). The drag-and-drop gate uses it: a dropped path is a
client-side filesystem path, meaningful to the server only when they share a
disk.
"""

from __future__ import annotations

from urllib.parse import urlparse

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_local_url(url: str) -> bool:
    """Whether *url* points at this machine.

    A URL with no host (e.g. a bare path) counts as local; an unparseable one
    does not.
    """
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host is None or host in _LOCAL_HOSTS
