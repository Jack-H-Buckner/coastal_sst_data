#!/usr/bin/env python3
"""
coastal_sst_data -- the standalone `python -m ...processes.<product>` entry point.

Every process module can be run on its own, and every one of them grew the same ~20 lines
to do it: an ArgumentParser with --config/--aoi/--overwrite/--dry-run/-v, a
logging.basicConfig, load_config, and a call to that module's `acquire`. Twelve copies,
identical but for the description string and the odd extra flag -- and they had already
drifted (bathymetry's --dry-run had no help text; mur's parser called itself the "MUR L4
SST acquisition" while the module had grown past that).

This is that boilerplate, once. A process module ends with:

    def main():
        entry.process_main(acquire, "coastal_sst_data MUR L4 SST acquisition.")

and a module with an extra flag declares it rather than rewriting the parser:

    entry.process_main(acquire, "...MODIS...", extra=[
        entry.Flag("--full-series", "ignore Landsat coincidence; load the full series"),
    ])

Kept OUT of cli.py deliberately: cli imports pipeline, which imports every process module,
so a process module importing cli would close an import cycle. This module imports only
argparse/logging/config, so it cannot.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from .config import load_config


@dataclass(frozen=True)
class Flag:
    """An extra `--flag` one process accepts, forwarded to its `acquire` as a kwarg.

    `dest` defaults to the flag name with the leading dashes stripped and hyphens turned
    into underscores (--full-series -> full_series), which is what argparse does anyway.
    """
    name: str
    help: str
    dest: str | None = None

    @property
    def kwarg(self) -> str:
        return self.dest or self.name.lstrip("-").replace("-", "_")


def build_parser(description: str, extra=()) -> argparse.ArgumentParser:
    """The parser every process module shares."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="Path to a project config YAML.")
    ap.add_argument("--aoi", nargs="+", help="Process only these AoI name(s).")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess items even if the aligned output exists")
    ap.add_argument("--dry-run", action="store_true", help="Report only; download nothing.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    for f in extra:
        ap.add_argument(f.name, action="store_true", help=f.help,
                        **({"dest": f.dest} if f.dest else {}))
    return ap


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def process_main(acquire, description: str, extra=(), argv=None) -> None:
    """Parse argv, load the config, and hand it to one process module's `acquire`.

    `argv` is threaded through (rather than always read from sys.argv) so a test can drive
    a module's CLI in-process, the way tests/test_cli.py already drives `cli.main([...])`.
    """
    args = build_parser(description, extra).parse_args(argv)
    setup_logging(args.verbose)
    project = load_config(args.config)
    kwargs = {f.kwarg: getattr(args, f.kwarg) for f in extra}
    return acquire(project, aois=args.aoi, dry_run=args.dry_run,
                   overwrite=args.overwrite, **kwargs)
