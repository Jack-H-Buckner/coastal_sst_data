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

Usage (preflight only -- verify credentials and exit):
    python -m coastal_sst_data.auth --config config.yaml
"""

from __future__ import annotations

import argparse
import logging

from .config import Project, load_config, required_backend

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backend login handlers. Each does a REAL round-trip so verify() confirms the
# credentials are valid, not just present. Heavy clients are imported lazily so
# importing this module (e.g. from the pipeline) is cheap.
# --------------------------------------------------------------------------- #
def _login_earthdata(settings) -> None:
    import earthaccess
    auth = earthaccess.login(strategy=settings.auth_strategy)
    if not getattr(auth, "authenticated", False):
        raise RuntimeError(
            f"Earthdata login (strategy={settings.auth_strategy!r}) did not "
            "authenticate; check ~/.netrc or EARTHDATA_USERNAME/PASSWORD.")


def _login_gee(settings) -> None:
    import ee
    if settings.service_account and settings.key_file:
        creds = ee.ServiceAccountCredentials(settings.service_account, str(settings.key_file))
        ee.Initialize(creds, project=settings.project)
    else:
        ee.Initialize(project=settings.project)
    ee.Number(1).getInfo()   # round-trip confirms the backend actually responds


def _login_copernicus(settings) -> None:
    """Copernicus Marine (CMEMS). The toolbox reads ~/.netrc natively, so `netrc` needs
    no credentials file of its own -- the secret stays where every other one does."""
    import copernicusmarine
    from pathlib import Path

    kw = {"check_credentials_valid": True}
    if settings.auth_strategy == "netrc":
        kw["credentials_file"] = Path.home() / ".netrc"
    # environment -> COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD, read by the toolbox.
    # interactive -> its own ~/.copernicusmarine file, or a prompt.
    if not copernicusmarine.login(**kw):
        raise RuntimeError(
            f"Copernicus Marine login (strategy={settings.auth_strategy!r}) did not "
            "authenticate; check ~/.netrc (machine auth.marine.copernicus.eu) or "
            "COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD.")


# backend name -> login handler. Add a new service with one entry here.
AUTH_HANDLERS = {
    "earthdata": _login_earthdata,
    "gee": _login_gee,
    "copernicus": _login_copernicus,
}


def authenticate(backend: str, settings) -> None:
    """Authenticate to one backend (raises on failure)."""
    handler = AUTH_HANDLERS.get(backend)
    if handler is None:
        raise ValueError(f"no auth handler registered for backend {backend!r}")
    handler(settings)


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
    """
    backends = required_backends(project, products)
    if not backends:
        log.info("No credentialed backends required (all products use public sources).")
        return {}

    results: dict[str, str] = {}
    for backend, settings in backends.items():
        log.info("Verifying %s credentials...", backend)
        try:
            authenticate(backend, settings)
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

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
