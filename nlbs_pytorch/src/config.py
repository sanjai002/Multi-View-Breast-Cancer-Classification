"""Config loading with dotted-key CLI overrides and attribute access.

Nested dicts are converted to ``Cfg`` at load time (recursively), so both CLI
overrides (``a.b=c``) and runtime attribute assignment (``cfg.a.b = c``) persist.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """Dict with attribute access. Nested dicts are already Cfg (see ``to_cfg``)."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def to_cfg(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_cfg(v) for v in obj]
    return obj


def _coerce(v: str) -> Any:
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return to_cfg(raw)
