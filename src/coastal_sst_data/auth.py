#!/usr/bin/env python3
"""
coastal_sst_data -- runtime authentication.

The config (coastal_sst_data.config) declares WHICH backend each product needs
(AUTH_REQUIREMENTS / required_backend) and holds only NON-SECRET settings. This
module is the runtime side:

  * authenticate(backend, settings) -- log in to one backend.
  * verify(project)                 -- a PREFLIGHT that actually CONNECTS to every
                                       backend the selected products require, so
                                       bad/missing credentials fail in seconds --
                                       before the pipeline downloads anything.

Secrets are never here or in the config: earthaccess reads ~/.netrc or the
EARTHDATA_USERNAME/PASSWORD env vars; Earth Engine uses a service-account key
file (path only, in config) or the user's `earthengine authenticate` token. A
new backend registers a handler here and follows the same "secret stays external"
rule.

CREDENTIALS EXPIRE, AND RUNS ARE LONG. A preflight is necessary and not sufficient: a
multi-year AoI is hours of downloading, and an EDL token or a SAS signature minted at t=0 is
dead well before the last granule. Worse, the failure does not look like a failure -- scenes
are processed in date order, so an expiring token looks exactly like the data running out
partway through the range. So this module also owns:

  * login(backend, settings)   -- authenticate AND record WHEN, which is the state that makes
                                  every question below answerable.
  * refresh / refresher        -- force a genuinely fresh login; `refresher` returns the
                                  zero-argument callable `net.retry(refresh=...)` wants.
  * ensure_fresh               -- the proactive half: re-login before a credential gets old
                                  enough to die mid-item, called at loop boundaries.

FORCING A FRESH LOGIN IS NOT THE SAME AS LOGGING IN, and getting that wrong is silent.
`earthaccess.Auth.login` short-circuits on `self.authenticated`, so calling
`earthaccess.login()` a second time returns the SAME stale object without contacting EDL at
all -- a "refresh" that logs success and changes nothing. Hence AUTH_HANDLERS (log in) and
REFRESH_HANDLERS (log in again, for real) are separate registries, and every entry in the
second one is written against the specific library's escape hatch.

Usage (preflight only -- verify credentials and exit):
    python -m coastal_sst_data.auth --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

from . import logctx
from .config import Project, load_config, required_backend

log = logging.getLogger(__name__)

# Proactive TTL: a credential older than this is replaced at the next loop boundary rather
# than being allowed to die mid-item. Well under the ~30-60 min a PC SAS token or an EDL
# token typically lives, so the reactive path stays the exception rather than the rule.
_DEFAULT_MAX_AGE_S = 1800.0
MAX_AGE_S = _DEFAULT_MAX_AGE_S

# A login storm guard. A failing backend can produce hundreds of auth errors in a row -- one
# per granule -- and each one asks for a refresh. Without this, that is hundreds of logins.
MIN_REFRESH_INTERVAL_S = 60.0

# Per backend, per process. A run that genuinely needs more than this is not experiencing
# expiry, and silently re-logging-in forever would hide whatever it IS experiencing.
_DEFAULT_MAX_REFRESHES = 20
MAX_REFRESHES = _DEFAULT_MAX_REFRESHES

# Granules between proactive freshness checks inside a long per-item loop. The check itself is
# a clock read, but there is nothing to gain from asking twice a second, and a round number
# keeps the log legible.
CHECK_EVERY = 50

# Injectable so tests can advance time without sleeping (see tests/test_auth.py).
_clock = time.monotonic


class CredentialRefreshError(RuntimeError):
    """A credential could not be replaced -- refused, rate-limited out, or over budget.

    Distinct from a login failure so a caller can tell "these credentials are wrong" (raised
    by the handlers) from "these credentials may well be fine but I am not allowed to mint
    another right now".
    """


@dataclass
class _Session:
    """What we know about one backend's current credential. The whole point is `at`."""
    backend: str
    at: float                        # when it was last (re)authenticated, _clock()
    strategy: str | None = None      # so a deep call site can refresh without its `eff` bag
    refreshes: int = 0
    last_refresh_at: float | None = None


# backend -> _Session. Process-global because the credential is: every one of these libraries
# keeps its session in ITS OWN module globals, so there is no handle to pass around and
# nothing would be gained by scoping this any tighter.
_SESSIONS: dict[str, _Session] = {}

# `earthaccess.open()` fans out over a thread pool, so several workers can hit the same dead
# credential at the same instant and all ask to refresh it. The lock makes the rate-limit
# check-and-act atomic, so the first wins and the rest coalesce onto its result.
_LOCK = threading.RLock()


def _attr(settings, name: str, default=None):
    """One setting, from a pydantic model OR the flat dict the `eff` bags carry.

    `verify` holds an `EarthdataAuth`; a product's `eff["earthdata"]` is
    `{"auth_strategy": "netrc"}`. Both reach here, and neither should have to convert.
    """
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _strategy(backend: str, settings) -> str | None:
    """The auth strategy in force, falling back to what the recorded session logged in with.

    The fallback is what lets a deep call site build a refresher without threading its `eff`
    bag down to it -- the strategy was already captured when the stage logged in.
    """
    got = _attr(settings, "auth_strategy")
    if got is not None:
        return got
    s = _SESSIONS.get(backend)
    return None if s is None else s.strategy


# --------------------------------------------------------------------------- #
# Backend login handlers. Each does a REAL round-trip so verify() confirms the
# credentials are valid, not just present. Heavy clients are imported lazily so
# importing this module (e.g. from the pipeline) is cheap.
# --------------------------------------------------------------------------- #
def _login_earthdata(settings) -> None:
    import earthaccess
    strategy = _attr(settings, "auth_strategy", "netrc")
    auth = earthaccess.login(strategy=strategy)
    if not getattr(auth, "authenticated", False):
        raise RuntimeError(
            f"Earthdata login (strategy={strategy!r}) did not "
            "authenticate; check ~/.netrc or EARTHDATA_USERNAME/PASSWORD.")


def _login_gee(settings) -> None:
    import ee
    account, key_file = _attr(settings, "service_account"), _attr(settings, "key_file")
    project = _attr(settings, "project")
    if account and key_file:
        creds = ee.ServiceAccountCredentials(account, str(key_file))
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)
    ee.Number(1).getInfo()   # round-trip confirms the backend actually responds


def _login_copernicus(settings, *, force: bool = False) -> None:
    """Copernicus Marine (CMEMS). The toolbox reads ~/.netrc natively, so `netrc` needs
    no credentials file of its own -- the secret stays where every other one does.

    `force` re-fetches rather than accepting the cached ~/.copernicusmarine token, which is
    the whole difference between a login and a refresh.
    """
    import copernicusmarine
    from pathlib import Path

    kw = {"check_credentials_valid": True}
    if force:
        kw["force_overwrite"] = True
    if _attr(settings, "auth_strategy") == "netrc":
        kw["credentials_file"] = Path.home() / ".netrc"
    # environment -> COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD, read by the toolbox.
    # interactive -> its own ~/.copernicusmarine file, or a prompt.
    if not copernicusmarine.login(**kw):
        raise RuntimeError(
            f"Copernicus Marine login (strategy={_attr(settings, 'auth_strategy')!r}) did not "
            "authenticate; check ~/.netrc (machine auth.marine.copernicus.eu) or "
            "COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD.")


# backend name -> login handler. Add a new service with one entry here.
AUTH_HANDLERS = {
    "earthdata": _login_earthdata,
    "gee": _login_gee,
    "copernicus": _login_copernicus,
}


# --------------------------------------------------------------------------- #
# Forcing a FRESH login. Deliberately a second registry: for two of the three backends,
# re-calling the login handler does not re-authenticate, so "log in" and "log in again"
# are different operations and conflating them produces a refresh that silently no-ops.
# --------------------------------------------------------------------------- #
def _refresh_earthdata(settings) -> None:
    """Re-authenticate to Earthdata, defeating earthaccess's already-logged-in short-circuit.

    `earthaccess.auth.Auth.login` opens with

        if self.authenticated and (system == self.system):
            return self

    and `earthaccess.login()` passes the system it is already on -- so on a second call it
    returns the same stale `Auth` WITHOUT contacting EDL. Clearing the flag is what makes the
    handler actually re-run `_netrc()` / `_environment()`; `earthaccess.login` then rebuilds
    `earthaccess.__store__` from the new Auth, which is what `open()`/`download()` read.
    """
    import earthaccess

    auth_obj = getattr(earthaccess, "__auth__", None)
    if auth_obj is not None:
        auth_obj.authenticated = False
    _login_earthdata(settings)


def _refresh_copernicus(settings) -> None:
    """Re-validate Copernicus Marine credentials.

    HONEST CAVEAT: this is the weakest of the three, because copernicusmarine caches no token
    in-process -- credentials are re-read from env/file on every call, and the ARCO data path
    is `botocore.UNSIGNED` with the username attached only for usage accounting. So a CMEMS
    401/403 mid-run is usually NOT an expiry that re-logging-in can fix. What DOES go stale is
    the lazy dataset HANDLE, which carries the client it was opened with, and the only refresh
    that helps there is re-opening it -- see `processes.cmems`, which does exactly that.

    Kept anyway: it re-checks validity, so a genuinely lapsed subscription becomes a named
    error at the point of failure instead of a silently empty channel.
    """
    _login_copernicus(settings, force=True)


def _refresh_gee(settings) -> None:
    """Re-initialise Earth Engine. `ee.Initialize` on an initialised client is a no-op, so the
    reset is what forces new credentials to be picked up."""
    import ee

    ee.Reset()
    _login_gee(settings)


def _refresh_pc(settings=None) -> None:
    """Evict Planetary Computer's SAS token cache so the next `sign()` mints a NEW token.

    `planetary_computer.sas.get_token` re-fetches only when the cached token has under a
    minute left:

        if not token or token.ttl() < 60:

    A token Azure has just REJECTED can still look valid by that clock -- clock skew, an
    Azure-side revocation, a short grace window -- and `sign()` then hands back the very token
    that produced the 403, making the re-sign a silent no-op. 1.0.0 has no force flag;
    clearing the dict IS the force flag.

    PC is anonymous, so this backend deliberately has no AUTH_HANDLERS entry and no
    `AuthConfig.pc` block -- there is nothing to log in to, only a credential to evict.
    """
    import planetary_computer.sas as pc_sas

    pc_sas.TOKEN_CACHE.clear()


# backend name -> FORCED re-login handler. A backend absent here simply cannot be refreshed;
# `refresher()` returns None for it and the call site keeps today's fail-fast behaviour.
#
# INVARIANT, and it is load-bearing: a name in AUTH_HANDLERS must match an `AuthConfig`
# attribute, because `required_backends` reaches settings via `getattr(project.auth, backend)`.
# A name that appears ONLY here need not -- and `pc` must not, since there is no such block.
REFRESH_HANDLERS = {
    "earthdata": _refresh_earthdata,
    "gee": _refresh_gee,
    "copernicus": _refresh_copernicus,
    "pc": _refresh_pc,
}


def authenticate(backend: str, settings) -> None:
    """Authenticate to one backend (raises on failure)."""
    handler = AUTH_HANDLERS.get(backend)
    if handler is None:
        raise ValueError(f"no auth handler registered for backend {backend!r}")
    handler(settings)


# --------------------------------------------------------------------------- #
# Credential lifetime: when it was minted, how old it is, and replacing it
# --------------------------------------------------------------------------- #
def configure(auth_cfg) -> None:
    """Apply a project's `auth.max_age_s` / `auth.max_refreshes` to this process.

    Module globals rather than threaded arguments, matching how every other tuning constant in
    the tree works (net.MAX_RETRY, landsat_pc.READ_ATTEMPTS): the refresh closures handed to
    `net.retry` are zero-argument by contract, so there is nowhere to thread a policy TO.
    Called from each stage's `acquire()`, which is the one entry point every invocation path
    -- pipeline, CLI subcommand and `python -m ...` -- goes through.

    `None` restores the defaults, which is what a project with no auth block means -- and
    what a test needs between cases.
    """
    global MAX_AGE_S, MAX_REFRESHES
    MAX_AGE_S = float(getattr(auth_cfg, "max_age_s", _DEFAULT_MAX_AGE_S))
    MAX_REFRESHES = int(getattr(auth_cfg, "max_refreshes", _DEFAULT_MAX_REFRESHES))


def login(backend: str, settings, *, force: bool = False) -> bool:
    """Authenticate to `backend` and RECORD WHEN, so its age is knowable later.

    Every stage should reach a backend through here rather than calling the client library
    directly -- not because the login differs, but because a login nobody timestamped cannot
    be proactively replaced, and that is the difference between a run that finishes and one
    that stops partway with no explanation.

    A LOGIN THAT ALREADY HAPPENED IS NOT REPEATED. Returns True if this call actually
    authenticated, False if it reused a credential younger than `MAX_AGE_S` -- which is the
    same freshness `ensure_fresh` enforces, so reusing one is never a weaker guarantee than
    minting one.

    That matters because a stage's `run()` calls this unconditionally at its top, and the
    number of times `run()` is entered is an ORCHESTRATION detail, not a property of the
    credential. The preflight (`verify`) already logs in once before any stage starts; every
    stage login after it was a second round-trip to EDL for a token we were already holding.
    Dispatching one AoI at a time -- which is what makes AoIs parallelisable at all, since
    `acquire()` already takes an AoI list -- turns that waste into a login storm: N AoIs times
    M products against one account, all inside a few seconds, which is exactly what a service
    is entitled to treat as abuse.

    `force=True` skips the freshness check. `refresh()` does NOT route through here (it has
    its own budget, rate limit and forced-relogin handlers); this is for a caller that knows
    the credential is stale for a reason the clock cannot see.

    THE LOCK IS HELD ACROSS THE LOGIN ITSELF, not just the bookkeeping. Checking freshness
    and then authenticating as two steps leaves the window this method exists to close: on a
    cold start several workers all read "no session", all decide to log in, and the account
    sees the storm anyway. Serialising them means the first authenticates while the rest
    wait, and each then finds a credential seconds old and reuses it -- one round-trip, N
    callers. The wait is real but it is once per backend per run, and it is strictly less
    than the N logins it replaces.
    """
    with _LOCK:
        if not force:
            a = age(backend)
            if a is not None and a < MAX_AGE_S:
                log.debug("  %s: reusing the credential minted %.1f min ago",
                          backend, a / 60.0)
                return False
        log.info("Authenticating with %s (strategy=%s)", backend,
                 _attr(settings, "auth_strategy", "n/a"))
        authenticate(backend, settings)
        _SESSIONS[backend] = _Session(backend=backend, at=_clock(),
                                      strategy=_attr(settings, "auth_strategy"))
    return True


def age(backend: str) -> float | None:
    """Seconds since `backend` was last (re)authenticated, or None if it never was."""
    s = _SESSIONS.get(backend)
    return None if s is None else _clock() - s.at


def reset() -> None:
    """Forget every recorded session. For tests; the credentials themselves are untouched."""
    _SESSIONS.clear()


def refresh(backend: str, settings=None) -> bool:
    """Force a fresh login for `backend`. True if one actually happened.

    False means "not now, and that is fine" -- another call refreshed this backend moments ago
    (see MIN_REFRESH_INTERVAL_S), so the credential in play is already new and the caller's
    retry should go ahead against it.

    Raises CredentialRefreshError when a refresh is impossible rather than merely unnecessary:
    an interactive strategy (which would block an unattended run on a password prompt), a
    backend with no refresh handler, or a budget that has run out.

    `settings` may be omitted where the stage already logged in -- the strategy was recorded
    then, so a deep call site does not have to thread its `eff` bag down to build a refresher.
    """
    handler = REFRESH_HANDLERS.get(backend)
    if handler is None:
        raise CredentialRefreshError(
            f"no refresh handler registered for backend {backend!r}; it cannot be "
            "re-authenticated mid-run.")

    # Checked FIRST, and before the rate limit, so an unattended run fails fast with an
    # actionable message instead of stalling on a prompt nobody is there to answer.
    if _strategy(backend, settings) == "interactive":
        raise CredentialRefreshError(
            f"{backend} credentials expired mid-run and auth_strategy='interactive' cannot "
            f"re-authenticate without a prompt. Set auth.{backend}.auth_strategy to 'netrc' "
            "or 'environment' for long unattended runs.")

    # Held across the check AND the act: `earthaccess.open` fans out over a thread pool, so
    # without this every worker that hit the same dead credential would pass the rate-limit
    # check together and then all log in.
    with _LOCK:
        s = _SESSIONS.get(backend)
        now = _clock()
        if s is not None and s.last_refresh_at is not None:
            since = now - s.last_refresh_at
            if since < MIN_REFRESH_INTERVAL_S:
                log.debug("  %s was refreshed %.0fs ago; reusing that credential rather than "
                          "logging in again", backend, since)
                return False
        if s is not None and s.refreshes >= MAX_REFRESHES:
            raise CredentialRefreshError(
                f"{backend} refresh budget exhausted ({s.refreshes} refreshes); refusing to "
                "re-login again -- the credentials are being rejected for a reason a refresh "
                "cannot fix.")

        was = age(backend)
        strategy = _strategy(backend, settings)
        handler(settings)
        n = (s.refreshes + 1) if s is not None else 1
        _SESSIONS[backend] = _Session(backend=backend, at=now, strategy=strategy,
                                      refreshes=n, last_refresh_at=now)
    log.info("  RE-AUTHENTICATED %s (age was %s, refresh %d of %d this run)", backend,
             "unknown" if was is None else f"{was / 60.0:.1f} min", n, MAX_REFRESHES)
    return True


def refresher(backend: str, settings=None):
    """A zero-argument callable that replaces `backend`'s credential, or None if it cannot be.

    This is the shape `net.retry(refresh=...)` expects:

        net.retry(lambda: earthaccess.open([g])[0], what="MUR granule",
                  refresh=auth.refresher("earthdata", settings))

    None when the backend has no refresh handler, which makes the `refresh=` argument inert
    and leaves that call site behaving exactly as it did before -- a product on a public
    endpoint does not need to know whether it has credentials.
    """
    if backend is None or backend not in REFRESH_HANDLERS:
        return None
    return lambda: refresh(backend, settings)


def ensure_fresh(backend: str, settings, *, max_age_s: float | None = None) -> bool:
    """Replace `backend`'s credential if it is older than `max_age_s`. True if it did.

    The proactive half. Reacting to a 403 works, but it costs a failed request every time and
    -- for the products that read outside their retry -- risks the failure being mistaken for
    missing data. Call this at LOOP BOUNDARIES (per AoI, or every N granules), never per item:
    the check is cheap but the refresh is not, and there is nothing to gain from asking twice
    a second.

    Never raises: a proactive refresh is an optimisation, and a run must not die because an
    optimisation failed. A genuine problem will resurface on the next real request, where the
    reactive path reports it with the context of what was being read.
    """
    # Resolved here, not as a default argument: a default is bound at import, so it would
    # ignore whatever `configure()` later read out of the project config.
    max_age_s = MAX_AGE_S if max_age_s is None else max_age_s
    a = age(backend)
    if a is None or a < max_age_s:
        return False
    log.info("  %s credential is %.1f min old (max_age_s=%.0f); refreshing before the next "
             "batch", backend, a / 60.0, max_age_s)
    try:
        return refresh(backend, settings)
    except CredentialRefreshError as exc:
        log.warning("  %s proactive refresh skipped: %s", backend, exc)
        return False


def required_backends(project: Project, products=None) -> dict[str, object]:
    """{backend: settings} for the backends the selected products require.

    `products` optionally restricts to a subset (e.g. what the pipeline is about
    to run). Config validation guarantees each required settings block exists, so
    the values are never None.
    """
    items = project.products.items()
    if products is not None:
        want = set(products)
        items = [(p, o) for p, o in items if p in want]
    out: dict[str, object] = {}
    for product, opts in items:
        backend = required_backend(product, opts)
        if backend:
            out[backend] = getattr(project.auth, backend)
    return out


def verify(project: Project, products=None) -> dict[str, str]:
    """Connect to every required backend; raise if any credentials fail.

    Returns ``{backend: 'ok'}``. On failure raises RuntimeError listing ALL
    failed backends (so the user can fix everything at once), not just the first.

    Also SEEDS the session registry, so a credential's age is counted from the preflight that
    actually minted it rather than from whenever a stage first happened to ask -- and every
    stage login afterwards reuses what this minted (see `login`).

    Logs in with `force=True`: the whole promise of a preflight is that it CONNECTS. Reusing a
    recorded session here would turn "your credentials work" into "your credentials worked at
    some point earlier in this process", which is not the question the user asked.
    """
    backends = required_backends(project, products)
    if not backends:
        log.info("No credentialed backends required (all products use public sources).")
        return {}

    results: dict[str, str] = {}
    for backend, settings in backends.items():
        log.info("Verifying %s credentials...", backend)
        try:
            login(backend, settings, force=True)
            results[backend] = "ok"
            log.info("  %s: OK", backend)
        except Exception as exc:
            results[backend] = f"FAILED: {exc}"
            log.error("  %s: %s", backend, exc)

    failed = {b: r for b, r in results.items() if r != "ok"}
    if failed:
        raise RuntimeError(
            "credential verification failed for: " + ", ".join(sorted(failed)) + "\n"
            + "\n".join(f"  {b}: {r}" for b, r in failed.items()))
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Verify that the configured credentials connect to each service.")
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logctx.configure(verbose=args.verbose)

    project = load_config(args.config)
    try:
        results = verify(project)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
    if results:
        print("All credentials verified: " + ", ".join(f"{b}=ok" for b in results))
    else:
        print("No credentialed backends required.")


if __name__ == "__main__":
    main()
