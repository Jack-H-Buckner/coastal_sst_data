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

THE THIRD CATEGORY. That rule has one hole, and it cost a whole Landsat date range before it
was found: an EXPIRED credential is a 403 too, and re-reading the same dead URL really is
pointless -- but re-reading it with a FRESH credential is not. So there are three answers, not
two:

  * transient   -- try again, after a backoff.                     (`is_transient`)
  * auth        -- try again ONCE, but only after replacing the
                   credential; if it still fails, it was never
                   an expiry.                                      (`is_auth_error`)
  * final       -- everything else. Raise now.

`retry` stays auth-blind unless the caller hands it a `refresh` callable, because only the
caller knows WHICH credential is behind the call and how to mint another. With no `refresh`
the behaviour is exactly what it always was. See `auth.refresher`, which returns a ready-made
one, and `processes.landsat_pc.read_scene`, where this pattern was first worked out.
"""

from __future__ import annotations

import logging
import os
import re
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

# "Your credential is not acceptable", as opposed to "the server is busy" or "it isn't there".
AUTH_STATUS = {401, 403}

# Auth-shaped WORDINGS, for the many failures that carry no status at all: GDAL's
# RuntimeError, fsspec and h5 wrappers, botocore. Matched against `err_text`, which has
# already walked the cause chain and scrubbed path-like tokens.
#
# UNAMBIGUOUS ONLY. Every entry here names a credential; none of them can be produced by a
# granule that is merely absent. That is what makes this tier safe to apply everywhere.
AUTH_MARKERS = ("unauthorized", "forbidden", "access denied", "accessdenied",
                "authenticationfailed", "authentication failed", "authorizationfailure",
                "invalid credential", "invalid_grant", "credentials were rejected",
                "token expired", "expired token", "signature expired",
                "signature not valid", "invalid signature", "permission denied")

# The EXTRA wordings a dead PRESIGNED URL arrives in. /vsicurl cannot always tell a 403 on the
# header request from a file that is simply not there, so an expired SAS token surfaces as
# "does not exist" -- and so does a granule that genuinely does not exist. These are therefore
# AMBIGUOUS, and opt-in rather than default: use them only where the credential is per-URL, so
# that a false positive costs one extra read of a scene that was already failing.
#
# The asymmetry is deliberate and was learned expensively (see processes.landsat_pc, which
# carried this list privately until every backend needed it): a false positive costs one read,
# while a false negative costs the scene AND every scene after it -- which is exactly how an
# expiring Landsat token once masqueraded as the date range simply ending.
SIGNED_URL_MARKERS = AUTH_MARKERS + (
    "signature", "expired", "not recognized as", "no such file",
    "does not exist in the file system")

# A status quoted in prose, when no exception attribute carried it. WORD-BOUNDED, which is the
# whole game: `err_text` only scrubs tokens containing a slash, so a bare scene id like
# `LC08_L2SP_046027_20240403_02_T1` survives it -- and a plain `"403" in text` would read that
# April 3rd acquisition date as Forbidden. `_` and digits are word characters, so `\b403\b`
# does not match inside `_20240403_`, while `HTTP response code: 403` does.
_STATUS_WORD = re.compile(r"\b(401|403)\b")

# How many times `retry` will replace the credential and try again before accepting that the
# credential was never the problem. One: a fresh credential either works or it does not.
AUTH_ATTEMPTS = 1


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


_PATHY = re.compile(r"\S*[/\\]\S*")


def err_text(exc: BaseException) -> str:
    """An exception's text -- its whole cause chain, paths and URLs removed -- for matching on.

    Paths go because they are full of digits that are not statuses. GDAL puts the entire href
    in every message it raises, and a Landsat href carries the acquisition date
    (`LC08_L2SP_046027_20240403_02_T1`) plus a random SAS token: without this, the "403"
    inside an April 3rd scene reads as Forbidden, and a reset connection on that day becomes a
    permanent answer. A date is not a status.

    The cause chain goes IN because rioxarray/fsspec wrap the failure that actually happened:
    the outer message says "could not open", and the word that decides is one level down.
    """
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or exc.__context__
    return _PATHY.sub("<path>", " | ".join(parts)).lower()


def is_transient(exc: BaseException) -> bool:
    """Is this worth trying again, or is it the server's final answer?

    Errs toward NOT retrying when the exception names a definite status: a 404 retried four
    times is four times the wait for the same answer, and it hides the message that says
    what to fix.
    """
    status = _status_of(exc)
    if status is not None:
        return status in TRANSIENT_STATUS

    # No status to go on (socket errors, fsspec/h5 wrappers, GDAL's RuntimeError). Read the
    # TEXT first, then fall back to the exception TYPE.
    #
    # That order used to be reversed, and it quietly inverted this module's whole promise.
    # `rasterio.errors.RasterioIOError` subclasses OSError, so the isinstance fallback fired
    # before this list was ever consulted, and EVERY 403 GDAL reported was retried four times
    # with backoff as a "hiccup" -- the exact opposite of "a 403 is an ANSWER". An expired
    # Planetary Computer signature is what that looks like in practice.
    text = err_text(exc)
    if any(w in text for w in ("unauthorized", "forbidden", "not found", "no such file",
                               "401", "403", "404", "credential", "authentication")):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return any(w in text for w in ("timeout", "timed out", "connection", "reset", "broken "
                                   "pipe", "temporarily", "try again", "503", "502", "500",
                                   "429", "curl", "ssl", "eof"))


def is_auth_error(exc: BaseException, markers: tuple[str, ...] = AUTH_MARKERS) -> bool:
    """Does this failure look like a credential the server would accept a REPLACEMENT for?

    The third category. `is_transient` answers "will the same request work if I ask again";
    this answers "will it work if I ask again with a NEW credential". An expired Earthdata
    token and a dead Planetary Computer signature are both 403s that `is_transient` rightly
    refuses to retry, and both are fixed by re-authenticating rather than by waiting.

    True is not a promise that the credential expired -- only that replacing it is worth one
    attempt. `markers` widens the vocabulary where the credential is per-URL and an expiry can
    surface as a missing file; see SIGNED_URL_MARKERS.
    """
    status = _status_of(exc)
    if status is not None:
        # A named status is definitive in BOTH directions: 403 is auth, 404 is not, and prose
        # in the body does not override the number the server actually sent. Without this, a
        # chatty 500 from an auth service would read as a credential failure.
        return status in AUTH_STATUS
    text = err_text(exc)
    if _STATUS_WORD.search(text):
        return True
    return any(w in text for w in markers)


def retry(fn, *, what: str, attempts: int = MAX_RETRY, delay: float = RETRY_DELAY_S,
          sleep=time.sleep, refresh=None, auth_attempts: int = AUTH_ATTEMPTS,
          markers: tuple[str, ...] = AUTH_MARKERS):
    """Call `fn()`, retrying TRANSIENT failures with exponential backoff.

        granules = net.retry(lambda: earthaccess.search_data(...), what="MUR search")

    Raises the last exception if every attempt fails, so the caller's own error handling
    (and the run report's failure tally) still sees a real failure rather than a None.

    `refresh` opts into the third category (see the module docstring): a zero-argument
    callable that replaces the credential behind `fn`, typically `auth.refresher(backend,
    settings)`. When one is supplied, an auth-shaped failure costs a refresh and ONE more
    attempt instead of ending the call:

        net.retry(lambda: earthaccess.open([g])[0], what="MUR granule",
                  refresh=auth.refresher("earthdata", settings))

    That retry deliberately does NOT consume an attempt from `attempts` and does NOT sleep the
    backoff. The transient budget exists for a server having a moment; this is a different
    failure with a different fix, and a fresh credential either works on the spot or the
    credential was never the problem. Bounded by `auth_attempts` so a permanently-rejected
    credential still costs two requests, not `attempts`.

    With no `refresh` -- every call site that has not opted in -- the behaviour is exactly what
    it was before the parameter existed.
    """
    wait = delay
    used_auth = 0
    i = 0
    while True:
        i += 1
        try:
            return fn()
        except BaseException as exc:                 # noqa: BLE001 -- re-raised below
            if isinstance(exc, KeyboardInterrupt):
                raise
            if refresh is not None and used_auth < auth_attempts and is_auth_error(exc, markers):
                used_auth += 1
                log.warning("    %s: credential rejected (%s); refreshing and retrying",
                            what, exc)
                try:
                    refresh()
                except BaseException as rexc:                    # noqa: BLE001
                    if isinstance(rexc, KeyboardInterrupt):
                        raise
                    # Re-raise the ORIGINAL read failure, not the refusal. `rep.fail(item, exc)`
                    # records this string in the run report, and "budget exhausted" there would
                    # hide which granule was actually lost.
                    log.error("    %s: credential refresh did not happen (%s); failing on the "
                              "original error", what, rexc)
                    raise exc from None
                i -= 1             # the refresh is not one of the transient attempts
                continue
            if not is_transient(exc):
                if used_auth:
                    # We already replaced the credential and the answer did not change. Say so:
                    # this is the line that distinguishes "the token expired" from "these
                    # credentials are not allowed to read this", which look identical otherwise.
                    log.warning("    %s: still rejected after a credential refresh (%s) -- "
                                "this is not an expiry; check the credentials themselves",
                                what, exc)
                # A permanent answer. Do not spend a minute rediscovering it.
                raise
            if i >= attempts:
                log.warning("    %s: giving up after %d attempt(s) (%s)", what, attempts, exc)
                raise
            log.warning("    %s: transient failure (%s); retry %d/%d in %.0fs",
                        what, exc, i, attempts - 1, wait)
            sleep(wait)
            wait *= 2
