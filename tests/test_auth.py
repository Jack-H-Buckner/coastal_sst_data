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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "-o", "log_cli=true"])