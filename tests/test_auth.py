"""Runtime auth layer: backend dispatch, requirement resolution over a project,
and the verify() preflight. Offline -- the real login handlers are stubbed, so
no network or credentials are touched."""

import pytest

from coastal_sst_data.config import DataProduct, parse_config
from coastal_sst_data import auth


def _project(products, *, earthdata=False, gee=False):
    """A valid project selecting `products`, with optional auth blocks."""
    auth_block = {}
    if earthdata:
        auth_block["earthdata"] = {"auth_strategy": "netrc"}
    if gee:
        auth_block["gee"] = {"project": "p"}
    d = {
        "name": "t", "output_dir": "o",
        "time": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "products": products,
        "regions": [{"name": "r", "areas": [
            {"name": "a", "center_lat": 45.5, "center_lon": -123.9,
             "buffer_ns_km": 8, "buffer_ew_km": 8}]}],
    }
    if auth_block:
        d["auth"] = auth_block
    return parse_config(d)


def _fail(msg):
    def handler(settings):
        raise RuntimeError(msg)
    return handler


# ---------------------------------------------------------------------------
# authenticate: dispatch to a registered handler.
# ---------------------------------------------------------------------------
def test_authenticate_dispatches_to_handler(monkeypatch):
    seen = []
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: seen.append(s))
    auth.authenticate("earthdata", "SETTINGS")
    assert seen == ["SETTINGS"]


def test_authenticate_unknown_backend_raises():
    with pytest.raises(ValueError, match="no auth handler"):
        auth.authenticate("nope", None)


# ---------------------------------------------------------------------------
# required_backends: which backends a project's selected products need.
# ---------------------------------------------------------------------------
def test_required_backends_dedups_and_skips_public():
    # mur+ecostress -> earthdata (once); landsat default pc -> none; bathymetry -> none
    proj = _project({"mur": None, "ecostress": None, "landsat": None, "bathymetry": None},
                    earthdata=True)
    b = auth.required_backends(proj)
    assert set(b) == {"earthdata"}
    assert b["earthdata"] is proj.auth.earthdata      # the actual settings object


def test_required_backends_respects_products_filter():
    proj = _project({"mur": None, "bathymetry": None}, earthdata=True)
    assert set(auth.required_backends(proj, products=[DataProduct.bathymetry])) == set()
    assert set(auth.required_backends(proj, products=[DataProduct.mur])) == {"earthdata"}


# ---------------------------------------------------------------------------
# verify: a real-connection preflight (handlers stubbed here).
# ---------------------------------------------------------------------------
def test_verify_all_ok(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": lambda s: None})
    assert auth.verify(_project({"mur": None}, earthdata=True)) == {"earthdata": "ok"}


def test_verify_public_only_needs_nothing():
    assert auth.verify(_project({"bathymetry": None})) == {}


def test_verify_raises_on_bad_credentials(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": _fail("bad netrc")})
    with pytest.raises(RuntimeError, match="earthdata"):
        auth.verify(_project({"mur": None}, earthdata=True))


def test_verify_reports_all_failures(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_HANDLERS",
                        {"earthdata": _fail("e-boom"), "gee": _fail("g-boom")})
    proj = _project({"mur": None, "landcover": {"source": "gee"}},
                    earthdata=True, gee=True)
    with pytest.raises(RuntimeError) as exc:
        auth.verify(proj)
    # both failing backends named -> user fixes everything at once
    assert "earthdata" in str(exc.value) and "gee" in str(exc.value)


def test_verify_only_checks_requested_products(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": _fail("should not run")})
    proj = _project({"mur": None, "bathymetry": None}, earthdata=True)
    # verifying only the public product must NOT touch earthdata
    assert auth.verify(proj, products=[DataProduct.bathymetry]) == {}

# ---------------------------------------------------------------------------
# Credential lifetime: the state auth.py did not have.
#
# A preflight is not enough on a multi-hour run. These pin the three things that
# make a mid-run credential replaceable: knowing its AGE, forcing a genuinely
# fresh login, and refusing to do either when it would do harm.
# ---------------------------------------------------------------------------
@pytest.fixture
def clock(monkeypatch):
    """A hand-advanced monotonic clock, so ages can be tested without sleeping."""
    now = [1000.0]
    monkeypatch.setattr(auth, "_clock", lambda: now[0])
    return now


@pytest.fixture
def handlers(monkeypatch):
    """Stub login + refresh handlers that record the order they ran in."""
    order = []
    monkeypatch.setitem(auth.AUTH_HANDLERS, "earthdata", lambda s: order.append("login"))
    monkeypatch.setitem(auth.REFRESH_HANDLERS, "earthdata", lambda s: order.append("refresh"))
    return order


NETRC = {"auth_strategy": "netrc"}


def test_login_records_when_so_the_age_is_knowable(clock, handlers):
    auth.login("earthdata", NETRC)
    assert auth.age("earthdata") == 0.0
    clock[0] += 600
    assert auth.age("earthdata") == 600.0


def test_age_is_none_for_a_backend_that_never_logged_in():
    assert auth.age("earthdata") is None


def test_verify_seeds_the_session_registry(monkeypatch):
    """So a credential's age is counted from the preflight that actually minted it, and the
    first stage does not immediately log in a second time."""
    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": lambda s: None})
    auth.verify(_project({"mur": None}, earthdata=True))
    assert auth.age("earthdata") is not None


def test_login_reuses_a_credential_that_is_still_fresh(clock, handlers):
    """A stage's run() logs in at its top, and how many times run() is ENTERED is an
    orchestration detail -- one call per AoI rather than one per product is what makes AoIs
    parallelisable. Without this, that split becomes N x M logins against one account in a
    few seconds, which a service is entitled to read as abuse."""
    assert auth.login("earthdata", NETRC) is True
    assert auth.login("earthdata", NETRC) is False
    assert auth.login("earthdata", NETRC) is False
    assert handlers == ["login"]                      # ONE round-trip, not three


def test_login_re_authenticates_once_the_credential_is_old(clock, handlers):
    """Reuse is bounded by the same MAX_AGE_S that `ensure_fresh` enforces, so reusing a
    recorded session is never a weaker guarantee than minting a new one."""
    auth.login("earthdata", NETRC)
    clock[0] += auth.MAX_AGE_S + 1
    assert auth.login("earthdata", NETRC) is True
    assert handlers == ["login", "login"]


def test_login_force_always_authenticates(clock, handlers):
    auth.login("earthdata", NETRC)
    assert auth.login("earthdata", NETRC, force=True) is True
    assert handlers == ["login", "login"]


def test_verify_connects_even_when_a_session_is_recorded(monkeypatch, clock):
    """The whole promise of a preflight is that it CONNECTS. If `verify` reused a recorded
    session it would answer "your credentials worked earlier in this process", which is not
    the question the user asked when they ran `coastal-sst-data verify`."""
    calls = []
    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": lambda s: calls.append(1)})
    project = _project({"mur": None}, earthdata=True)

    auth.verify(project)
    auth.verify(project)          # same process, credential still young

    assert len(calls) == 2


def test_refresh_forces_a_login_the_plain_handler_would_not(clock, handlers):
    """The registries are separate because `earthaccess.login()` a second time is a NO-OP --
    it returns the same stale Auth without contacting EDL. A refresh that reused
    AUTH_HANDLERS would log success and change nothing."""
    auth.login("earthdata", NETRC)
    clock[0] += 3600
    assert auth.refresh("earthdata", NETRC) is True
    assert handlers == ["login", "refresh"]
    assert auth.age("earthdata") == 0.0          # the clock restarted


def test_ensure_fresh_is_a_noop_inside_the_ttl(clock, handlers):
    auth.login("earthdata", NETRC)
    clock[0] += 60
    assert auth.ensure_fresh("earthdata", NETRC, max_age_s=1800) is False
    assert handlers == ["login"]


def test_ensure_fresh_replaces_a_stale_credential(clock, handlers):
    auth.login("earthdata", NETRC)
    clock[0] += 2000
    assert auth.ensure_fresh("earthdata", NETRC, max_age_s=1800) is True
    assert handlers == ["login", "refresh"]


def test_the_rate_limit_stops_a_login_storm(clock, handlers):
    """A dead credential does not fail once -- it fails on every one of the hundreds of
    granules still to come, and each one asks for a refresh."""
    auth.login("earthdata", NETRC)
    assert auth.refresh("earthdata", NETRC) is True
    clock[0] += 5                                 # well inside MIN_REFRESH_INTERVAL_S
    assert auth.refresh("earthdata", NETRC) is False
    assert handlers == ["login", "refresh"]       # not twice


def test_the_budget_stops_refreshing_forever(clock, handlers, monkeypatch):
    """A run needing more than the budget is not experiencing expiry, and quietly
    re-logging-in forever would hide whatever it IS experiencing."""
    monkeypatch.setattr(auth, "MAX_REFRESHES", 2)
    monkeypatch.setattr(auth, "MIN_REFRESH_INTERVAL_S", 0.0)
    auth.login("earthdata", NETRC)
    auth.refresh("earthdata", NETRC)
    auth.refresh("earthdata", NETRC)
    with pytest.raises(auth.CredentialRefreshError, match="budget exhausted"):
        auth.refresh("earthdata", NETRC)
    assert handlers.count("refresh") == 2


def test_interactive_is_never_re_authenticated(clock, handlers):
    """A run blocked overnight on a password prompt is worse than one that failed."""
    auth.login("earthdata", {"auth_strategy": "interactive"})
    with pytest.raises(auth.CredentialRefreshError, match="interactive"):
        auth.refresh("earthdata", {"auth_strategy": "interactive"})
    assert "refresh" not in handlers
    # and the message must say what to do about it
    with pytest.raises(auth.CredentialRefreshError, match="netrc"):
        auth.refresh("earthdata", {"auth_strategy": "interactive"})


def test_ensure_fresh_swallows_a_refusal_but_refresh_raises(clock, handlers):
    """A reactive refresh runs after a read already failed, so raising costs nothing. A
    proactive one runs while everything is working, and must never be what stops a run."""
    auth.login("earthdata", {"auth_strategy": "interactive"})
    clock[0] += 5000
    assert auth.ensure_fresh("earthdata", {"auth_strategy": "interactive"}) is False


def test_the_strategy_is_remembered_so_deep_call_sites_need_no_settings(clock, handlers):
    """`modis._fetch_download` builds a refresher without its `eff` bag in scope."""
    auth.login("earthdata", {"auth_strategy": "interactive"})
    with pytest.raises(auth.CredentialRefreshError, match="interactive"):
        auth.refresh("earthdata")            # settings omitted


def test_refresher_is_a_zero_arg_callable_for_net_retry(clock, handlers):
    auth.login("earthdata", NETRC)
    r = auth.refresher("earthdata", NETRC)
    r()
    assert handlers == ["login", "refresh"]


def test_refresher_is_none_for_a_backend_that_cannot_be_refreshed():
    """None, not a no-op: `net.retry(refresh=None)` is byte-identical to the old path, so a
    public product's call site keeps failing fast exactly as it did."""
    assert auth.refresher("nope") is None
    assert auth.refresher(None) is None


def test_refresh_of_an_unknown_backend_refuses():
    with pytest.raises(auth.CredentialRefreshError, match="no refresh handler"):
        auth.refresh("nope", NETRC)


def test_pc_is_refreshable_without_being_a_login_backend():
    """Planetary Computer is anonymous -- nothing to log in to -- but its SAS token is still
    a credential that expires. It must therefore NOT be in AUTH_HANDLERS (required_backends
    would try `getattr(project.auth, "pc")`), yet must be refreshable."""
    assert "pc" not in auth.AUTH_HANDLERS
    assert "pc" in auth.REFRESH_HANDLERS
    assert auth.refresher("pc") is not None


def test_pc_refresh_evicts_the_token_cache(monkeypatch):
    """`planetary_computer.sign` re-fetches only below 60s of nominal TTL, so a token Azure
    has just REJECTED can be handed straight back. Clearing the cache is the force flag."""
    pc_sas = pytest.importorskip("planetary_computer.sas")
    monkeypatch.setitem(pc_sas.TOKEN_CACHE, "https://example/token", "dead")
    auth.refresh("pc")
    assert "https://example/token" not in pc_sas.TOKEN_CACHE


def test_configure_applies_the_project_policy():
    proj = _project({"mur": None}, earthdata=True)
    proj.auth.max_age_s = 60.0
    proj.auth.max_refreshes = 3
    auth.configure(proj.auth)
    assert auth.MAX_AGE_S == 60.0 and auth.MAX_REFRESHES == 3
    auth.configure(None)                     # restores the defaults
    assert auth.MAX_AGE_S == auth._DEFAULT_MAX_AGE_S


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])

def test_concurrent_first_logins_collapse_to_one(monkeypatch, clock):
    """The check and the login must be ONE atomic step. Checked separately, a cold start
    lets every worker read "no session", all decide to authenticate, and the account sees
    the storm this whole mechanism exists to prevent."""
    import threading
    import time as _time

    calls, lock = [], threading.Lock()

    def slow_login(settings):
        _time.sleep(0.05)              # wide enough for the others to reach the check
        with lock:
            calls.append(1)

    monkeypatch.setattr(auth, "AUTH_HANDLERS", {"earthdata": slow_login})

    threads = [threading.Thread(target=auth.login, args=("earthdata", NETRC))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert calls == [1]                # ONE round-trip, six callers
