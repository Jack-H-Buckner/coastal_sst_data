"""Network hardening. The bulk-data paths had neither a timeout nor a retry: a stalled
connection hung the run forever, and one transient 503 permanently lost a scene."""

import os

import pytest
from rasterio.errors import RasterioIOError

from coastal_sst_data import net

# A real signed Planetary Computer asset href. Note what is inside it: the acquisition date
# `20240403` (which contains "403"), and a SAS token whose random base64 can contain any
# 3-digit run at all. Every GDAL error message quotes the whole thing.
_HREF = ("/vsicurl/https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/"
         "standard/oli-tirs/2024/046/027/LC08_L2SP_046027_20240403_02_T1/"
         "LC08_L2SP_046027_20240403_02_T1_ST_B10.TIF"
         "?st=2024-04-03T00%3A00%3A00Z&se=2024-04-04T00%3A00%3A00Z&sp=rl&sig=4o3Xq404z")


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _http(status):
    exc = RuntimeError(f"server said {status}")
    exc.response = _Resp(status)
    return exc


# --------------------------------------------------------------------------- #
# What is worth retrying -- and what is a final answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [500, 502, 503, 504, 429, 408])
def test_transient_statuses_are_retried(status):
    assert net.is_transient(_http(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_permanent_statuses_are_NOT_retried(status):
    """A 404 retried four times with backoff is four times the wait for the same answer --
    and it buries the message that tells you what to fix."""
    assert not net.is_transient(_http(status))


def test_socket_errors_are_transient():
    assert net.is_transient(TimeoutError("timed out"))
    assert net.is_transient(ConnectionError("connection reset by peer"))


def test_credential_errors_are_not_transient():
    assert not net.is_transient(RuntimeError("401 Unauthorized: bad credentials"))
    assert not net.is_transient(RuntimeError("HTTP 403 Forbidden"))


def test_a_rasterio_403_is_a_final_answer():
    """The bug that made a Landsat AoI stop halfway through its date range.

    `RasterioIOError` subclasses OSError, so the isinstance fallback used to fire BEFORE the
    deny-list was consulted -- and an expired Planetary Computer signature, which is a 403,
    was retried four times with backoff as if it were a hiccup."""
    exc = RasterioIOError(f"'{_HREF}': HTTP response code: 403")
    assert isinstance(exc, OSError)          # the MRO trap that caused it, pinned
    assert not net.is_transient(exc)


def test_a_rasterio_connection_reset_is_still_transient():
    """The reorder must not over-correct: a GDAL read that died mid-stream is worth retrying."""
    assert net.is_transient(
        RasterioIOError(f"'{_HREF}': CURL error: Recv failure: Connection reset by peer"))


def test_a_date_in_an_href_is_not_a_status_code():
    """`_20240403_` contains "403", and a SAS token can contain anything. Matching on the raw
    message would read an April 3rd scene's reset connection as Forbidden -- a date-shaped bug
    in the fix for a date-shaped bug."""
    assert "403" in _HREF
    assert net.is_transient(RasterioIOError(f"'{_HREF}': CURL error: Connection reset by peer"))


def test_a_bare_oserror_is_still_transient():
    """The type fallback survives the reorder -- it just no longer pre-empts the deny-list."""
    assert net.is_transient(OSError("something broke"))


def test_the_cause_chain_decides():
    """rioxarray wraps the failure that actually happened. Without walking the chain the outer
    "connection" would win and a dead signature would be retried."""
    try:
        try:
            raise RasterioIOError("HTTP response code: 403")
        except RasterioIOError as inner:
            raise RuntimeError("connection to the asset failed") from inner
    except RuntimeError as exc:
        assert not net.is_transient(exc)


# --------------------------------------------------------------------------- #
# retry()
# --------------------------------------------------------------------------- #
def test_retry_returns_the_first_success():
    calls = []
    assert net.retry(lambda: calls.append(1) or "ok", what="x", sleep=lambda s: None) == "ok"
    assert len(calls) == 1


def test_retry_recovers_from_a_transient_failure():
    """The case that used to permanently lose a scene."""
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _http(503)
        return "granule"

    assert net.retry(flaky, what="x", sleep=lambda s: None) == "granule"
    assert len(calls) == 3


def test_retry_gives_up_and_reraises():
    """The caller's own error handling -- and the run report's failure tally -- must still
    see a real failure, not a None."""
    def always():
        raise _http(503)

    with pytest.raises(RuntimeError, match="503"):
        net.retry(always, what="x", attempts=3, sleep=lambda s: None)


def test_retry_does_not_waste_time_on_a_permanent_failure():
    calls = []

    def gone():
        calls.append(1)
        raise _http(404)

    with pytest.raises(RuntimeError):
        net.retry(gone, what="x", attempts=5, sleep=lambda s: None)
    assert len(calls) == 1          # asked once, got a final answer, stopped


def test_retry_does_not_burn_attempts_on_an_expired_signature():
    """Re-reading a dead signed URL cannot succeed. The re-sign belongs one level up, in the
    caller that can mint a new signature -- see landsat_pc.read_scene."""
    calls = []

    def forbidden():
        calls.append(1)
        raise RasterioIOError(f"'{_HREF}': HTTP response code: 403")

    with pytest.raises(RasterioIOError):
        net.retry(forbidden, what="x", attempts=4, sleep=lambda s: None)
    assert len(calls) == 1


def test_retry_backs_off_exponentially():
    waits = []

    def always():
        raise _http(503)

    with pytest.raises(RuntimeError):
        net.retry(always, what="x", attempts=4, delay=2.0, sleep=waits.append)
    assert waits == [2.0, 4.0, 8.0]     # 3 sleeps before the 4th (final) attempt


def test_keyboard_interrupt_is_never_retried():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        net.retry(interrupted, what="x", sleep=lambda s: None)


# --------------------------------------------------------------------------- #
# is_auth_error -- the THIRD category: not "wait and try again", but "try again
# with a different credential"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_are_auth_errors(status):
    assert net.is_auth_error(_http(status))


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_other_statuses_are_not_auth_errors(status):
    """A status the server actually sent is definitive. Only 401/403 say "your credential"."""
    assert not net.is_auth_error(_http(status))


def test_a_500_mentioning_authentication_is_not_an_auth_error():
    """A known status must WIN over the prose. An auth service having an outage returns 5xx
    with "authentication" all over the body; refreshing a perfectly good credential because
    of it would burn the budget on someone else's problem."""
    exc = _http(500)
    exc.args = ("authentication service temporarily unavailable",)
    assert not net.is_auth_error(exc)
    assert net.is_transient(exc)            # it is a hiccup, and still retried as one


def test_a_rasterio_403_is_an_auth_error_and_still_not_transient():
    """Both classifiers must agree on a dead signature: not worth waiting for, worth
    re-signing for. If it were transient too, it would be retried against the dead URL."""
    exc = RasterioIOError(f"'{_HREF}': HTTP response code: 403")
    assert net.is_auth_error(exc)
    assert not net.is_transient(exc)


def test_a_date_in_a_scene_id_is_not_a_403():
    """The word-boundary test, and the reason `"403" in text` is not good enough.

    `err_text` only scrubs tokens containing a slash, so a BARE scene id survives it -- and
    `LC08_L2SP_046027_20240403_02_T1` contains "403". Matching it would refresh the credential
    on every failure of every April 3rd scene: a date-shaped bug in the fix for a
    date-shaped bug, one layer down from the one already pinned above.
    """
    exc = RuntimeError("LC08_L2SP_046027_20240403_02_T1 read failed")
    assert "403" in str(exc)
    assert not net.is_auth_error(exc)


def test_signed_url_markers_are_opt_in():
    """GDAL reports a 403 on a HEAD request as a file that is not there -- and so does a
    granule that is genuinely not there. Ambiguous, so only the per-URL callers opt in."""
    exc = RasterioIOError("'/vsicurl/https://x/y.tif': does not exist in the file system")
    assert not net.is_auth_error(exc)
    assert net.is_auth_error(exc, net.SIGNED_URL_MARKERS)


def test_the_cause_chain_decides_for_auth_too():
    try:
        try:
            raise RasterioIOError("HTTP response code: 403")
        except RasterioIOError as inner:
            raise RuntimeError("could not open the asset") from inner
    except RuntimeError as exc:
        assert net.is_auth_error(exc)


# --------------------------------------------------------------------------- #
# retry(refresh=...) -- replacing the credential, exactly once
# --------------------------------------------------------------------------- #
def test_refresh_fires_once_and_the_retry_succeeds():
    calls, refreshed, waits = [], [], []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise _http(401)
        return "ok"

    got = net.retry(flaky, what="x", sleep=waits.append,
                    refresh=lambda: refreshed.append(1))
    assert got == "ok"
    assert len(calls) == 2 and len(refreshed) == 1
    # A fresh credential works on the spot or it does not. Backing off first would just be
    # dead time in a run that is already hours long.
    assert waits == []


def test_the_auth_retry_does_not_spend_the_transient_budget():
    """The headline property. `attempts` is the budget for a server having a moment; an
    expired credential is a different failure with a different fix, and must not eat it."""
    calls, waits = [], []
    seq = [_http(401), _http(503), _http(503), "ok"]

    def flaky():
        calls.append(1)
        got = seq[len(calls) - 1]
        if got == "ok":
            return got
        raise got

    assert net.retry(flaky, what="x", attempts=3, sleep=waits.append,
                     refresh=lambda: None) == "ok"
    assert len(calls) == 4          # 1 auth + all 3 transient attempts still available
    assert waits == [2.0, 4.0]


def test_a_refreshed_credential_still_rejected_is_final():
    """The diagnosis the whole feature exists to deliver: if a NEW credential is rejected
    the same way, the problem was never expiry."""
    calls, refreshed = [], []

    def forbidden():
        calls.append(1)
        raise _http(403)

    with pytest.raises(RuntimeError):
        net.retry(forbidden, what="x", attempts=5, sleep=lambda s: None,
                  refresh=lambda: refreshed.append(1))
    assert len(calls) == 2 and len(refreshed) == 1     # not `attempts` times


def test_without_a_refresh_the_behaviour_is_unchanged():
    """Every call site that has not opted in must behave exactly as it did before."""
    calls = []

    def forbidden():
        calls.append(1)
        raise _http(401)

    with pytest.raises(RuntimeError):
        net.retry(forbidden, what="x", attempts=4, sleep=lambda s: None)
    assert len(calls) == 1


def test_a_refresh_that_raises_surfaces_the_ORIGINAL_error():
    """`rep.fail(item, exc)` records this string in the run report. "refresh budget
    exhausted" there would hide which granule was actually lost."""
    def forbidden():
        raise _http(401)

    def refuse():
        raise ValueError("auth_strategy='interactive' cannot re-authenticate")

    with pytest.raises(RuntimeError, match="server said 401"):
        net.retry(forbidden, what="x", sleep=lambda s: None, refresh=refuse)


def test_markers_are_threaded_through_retry():
    """A per-URL caller opts into the wider vocabulary; the default caller does not."""
    def missing():
        raise RasterioIOError("'/vsicurl/https://x/y.tif': no such file")

    default_refreshes, signed_refreshes = [], []
    with pytest.raises(RasterioIOError):
        net.retry(missing, what="x", attempts=1, sleep=lambda s: None,
                  refresh=lambda: default_refreshes.append(1))
    with pytest.raises(RasterioIOError):
        net.retry(missing, what="x", attempts=1, sleep=lambda s: None,
                  refresh=lambda: signed_refreshes.append(1), markers=net.SIGNED_URL_MARKERS)
    assert default_refreshes == [] and len(signed_refreshes) == 1


def test_keyboard_interrupt_is_never_treated_as_an_auth_error():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        net.retry(interrupted, what="x", sleep=lambda s: None, refresh=lambda: None)


# --------------------------------------------------------------------------- #
# GDAL env
# --------------------------------------------------------------------------- #
def test_setup_gdal_env_sets_a_deadline_and_a_retry_policy(monkeypatch):
    for k in ("GDAL_HTTP_TIMEOUT", "GDAL_HTTP_CONNECTTIMEOUT",
              "GDAL_HTTP_MAX_RETRY", "GDAL_HTTP_RETRY_DELAY"):
        monkeypatch.delenv(k, raising=False)

    net.setup_gdal_env()

    # Without these, a windowed COG read on a stalled connection hangs the run forever.
    assert os.environ["GDAL_HTTP_TIMEOUT"] == str(net.HTTP_TIMEOUT_S)
    assert os.environ["GDAL_HTTP_CONNECTTIMEOUT"] == str(net.CONNECT_TIMEOUT_S)
    assert int(os.environ["GDAL_HTTP_MAX_RETRY"]) >= 1
    assert int(os.environ["GDAL_HTTP_RETRY_DELAY"]) >= 1


def test_setup_gdal_env_does_not_override_an_operator(monkeypatch):
    monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "999")
    net.setup_gdal_env()
    assert os.environ["GDAL_HTTP_TIMEOUT"] == "999"


# --------------------------------------------------------------------------- #
# requests deadline
#
# `setup_gdal_env` reaches only GDAL. Herbie (the HRRR backend) fetches its index and its
# GRIB byte-ranges with plain `requests` and passes no timeout on all but one call, so
# without this patch a stalled mirror hangs the stage forever -- and on a thread pool it
# holds a worker nothing can cancel.
# --------------------------------------------------------------------------- #
def _reset_requests_patch(monkeypatch):
    """Undo the module-level install so each test observes a fresh one."""
    import requests
    monkeypatch.setattr(requests.Session, "request", requests.Session.request)
    monkeypatch.setattr(net, "_requests_patched", False)


def test_setup_requests_timeout_fills_in_a_missing_deadline(monkeypatch):
    import requests
    _reset_requests_patch(monkeypatch)
    seen = {}
    monkeypatch.setattr(requests.Session, "request",
                        lambda self, method, url, **kw: seen.update(kw) or "resp")

    net.setup_requests_timeout()
    requests.Session().request("GET", "http://example.invalid/")

    assert seen["timeout"] == (net.CONNECT_TIMEOUT_S, net.HTTP_TIMEOUT_S)


def test_setup_requests_timeout_keeps_an_explicit_one(monkeypatch):
    """`insitu_ioos._get`, `tides._get_json` and `datum._http_json` all pass their own
    timeout, chosen for their own endpoint. Clobbering those would be a regression."""
    import requests
    _reset_requests_patch(monkeypatch)
    seen = {}
    monkeypatch.setattr(requests.Session, "request",
                        lambda self, method, url, **kw: seen.update(kw) or "resp")

    net.setup_requests_timeout()
    requests.Session().request("GET", "http://example.invalid/", timeout=7)

    assert seen["timeout"] == 7


def test_setup_requests_timeout_is_idempotent(monkeypatch):
    """Called from every met/met_overpass acquire(); stacking wrappers per call would
    rebuild the chain on every stage and make the deadline harder to reason about."""
    import requests
    _reset_requests_patch(monkeypatch)
    net.setup_requests_timeout()
    once = requests.Session.request
    net.setup_requests_timeout()
    assert requests.Session.request is once
