#!/usr/bin/env python3
"""
coastal_sst_data -- network hardening: bounded waits, bounded retries.

The small JSON endpoints (tides, datum, ERDDAP) already retried with backoff. Everything
that actually MOVES DATA did not: `earthaccess.open/download`, the STAC searches,
`copernicusmarine.open_dataset`, and every windowed `/vsicurl` COG read had neither a
timeout nor a retry. Two consequences, both of which we have now seen in the audit:

  * A stalled TCP connection hangs the run FOREVER. There is no timeout to trip.
  * A single transient 503 permanently loses that scene/tile. The caller logs a warning
    and moves on, and -- because the skip guard treats a written output as done -- the day
    is never retried on a later run either. One blip becomes a permanent hole in the cube.

So: every remote read gets a deadline, and every remote read gets a small number of
retries with exponential backoff.

WHAT WE DO NOT RETRY matters as much as what we do. A 401/403/404 is an ANSWER, not a
hiccup: the credentials are wrong, or the granule is not there. Retrying it five times with
backoff burns a minute to arrive at the same reply, and worse, buries the message that tells
you how to fix it. Only transient failures (timeouts, connection resets, 5xx, 429) are
retried; everything else is raised immediately.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

# Bulk-transfer defaults. Generous -- a windowed COG read over a slow link is not a fault --
# but FINITE, which is the whole point.
HTTP_TIMEOUT_S = 120        # a single stalled read may not hang the run forever
CONNECT_TIMEOUT_S = 30
MAX_RETRY = 4
RETRY_DELAY_S = 2.0         # doubled each attempt: 2s, 4s, 8s

# Status codes worth trying again. 429 = rate-limited (backoff is exactly right); 5xx = the
# server is having a moment. Everything else in 4xx is a considered answer.
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def setup_gdal_env() -> None:
    """Give GDAL/curl a deadline and a retry policy for remote COG reads.

    Every module that reads a `/vsicurl` COG must call this. ECOSTRESS set the efficiency
    knobs but no timeouts; landsat_pc, landcover_esa and bathymetry set nothing at all, so
    their windowed reads could hang indefinitely on a stalled connection.

    `setdefault`, so an operator who has tuned these in the environment keeps their values.
    """
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", str(HTTP_TIMEOUT_S))
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", str(CONNECT_TIMEOUT_S))
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", str(MAX_RETRY))
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", str(int(RETRY_DELAY_S)))
    # Efficiency knobs for windowed reads of remote COGs (were ECOSTRESS-only).
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status behind an exception, if it carries one."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status", None) or getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def is_transient(exc: BaseException) -> bool:
    """Is this worth trying again, or is it the server's final answer?

    Errs toward NOT retrying when the exception names a definite status: a 404 retried four
    times is four times the wait for the same answer, and it hides the message that says
    what to fix.
    """
    status = _status_of(exc)
    if status is not None:
        return status in TRANSIENT_STATUS

    # No status to go on (socket errors, fsspec/h5 wrappers, GDAL's RuntimeError). Fall back
    # to the exception type and its text -- a timeout or a reset connection is transient by
    # definition.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(w in text for w in ("unauthorized", "forbidden", "not found", "no such file",
                               "401", "403", "404", "credential", "authentication")):
        return False
    return any(w in text for w in ("timeout", "timed out", "connection", "reset", "broken "
                                   "pipe", "temporarily", "try again", "503", "502", "500",
                                   "429", "curl", "ssl", "eof"))


def retry(fn, *, what: str, attempts: int = MAX_RETRY, delay: float = RETRY_DELAY_S,
          sleep=time.sleep):
    """Call `fn()`, retrying TRANSIENT failures with exponential backoff.

        granules = net.retry(lambda: earthaccess.search_data(...), what="MUR search")

    Raises the last exception if every attempt fails, so the caller's own error handling
    (and the run report's failure tally) still sees a real failure rather than a None.
    """
    wait = delay
    for i in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:                 # noqa: BLE001 -- re-raised below
            if isinstance(exc, KeyboardInterrupt):
                raise
            if not is_transient(exc):
                # A permanent answer. Do not spend a minute rediscovering it.
                raise
            if i == attempts:
                log.warning("    %s: giving up after %d attempt(s) (%s)", what, attempts, exc)
                raise
            log.warning("    %s: transient failure (%s); retry %d/%d in %.0fs",
                        what, exc, i, attempts - 1, wait)
            sleep(wait)
            wait *= 2
