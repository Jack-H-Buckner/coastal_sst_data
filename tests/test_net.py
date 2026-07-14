"""Network hardening. The bulk-data paths had neither a timeout nor a retry: a stalled
connection hung the run forever, and one transient 503 permanently lost a scene."""

import os

import pytest

from coastal_sst_data import net


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
