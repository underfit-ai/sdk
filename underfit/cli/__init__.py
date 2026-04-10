"""Command-line interface for Underfit."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _view(args: argparse.Namespace) -> int:
    logdir = Path(args.logdir).expanduser()
    if not logdir.is_dir():
        sys.stderr.write(f"error: logdir does not exist: {logdir.resolve()}\n")
        return 1
    previous = os.environ.get("UNDERFIT_CONFIG")
    with TemporaryDirectory(prefix="underfit-view-") as tmp:
        config = Path(tmp) / "underfit.toml"
        config.write_text(
            f'auth_enabled = false\n\n[database]\npath = {json.dumps(str(Path(tmp) / "db.sqlite"))}\n\n'
            f'[storage]\nbase = {json.dumps(str(logdir.resolve()))}\n\n[backfill]\nenabled = true\n',
            encoding="utf-8",
        )
        os.environ["UNDERFIT_CONFIG"] = str(config)
        try:
            uvicorn = importlib.import_module("uvicorn")
            uvicorn.run(importlib.import_module("underfit_api.main").app)
        except ImportError:
            sys.stderr.write("error: `underfit view` requires `underfit[view]`\n")
            return 1
        finally:
            if previous is None:
                os.environ.pop("UNDERFIT_CONFIG", None)
            else:
                os.environ["UNDERFIT_CONFIG"] = previous
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Underfit CLI."""
    parser = argparse.ArgumentParser(prog="underfit")
    subparsers = parser.add_subparsers(required=True)
    view = subparsers.add_parser("view")
    view.add_argument("--logdir", default="./underfit")
    view.set_defaults(func=_view)
    args = parser.parse_args(argv)
    return args.func(args)
