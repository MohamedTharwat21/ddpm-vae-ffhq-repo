"""Minimal YAML configuration loading.

Every experiment (VAE v1/v2/v3, LDM linear/cosine, pixel DDPM linear-logit
/ +REPA / cosine+EMA-65 / cosine+EMA-100 / linear+EMA) is described by one
YAML file under ``configs/``. Parameters were transcribed directly from the
actual ``CONFIG`` dicts supplied for each trained run - nothing here was
invented; see each YAML file's ``# source:`` comment for where its values
came from, and the README's experiment sections for anything that had to
be derived rather than copied verbatim (called out explicitly there).

``Config`` is intentionally a thin dict wrapper (attribute access as a
convenience over ``dict.__getitem__``) rather than a rigid dataclass
schema - the point is to keep configuration data-driven and easy to diff
against the original notebooks' ``CONFIG`` dicts, not to add a validation
framework this project does not need.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

__all__ = ["Config", "load_config"]


class Config(dict):
    """A dict that also supports attribute access and recursive wrapping."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaf
    conflicts; nested dicts are merged key-by-key rather than replaced
    wholesale, so an override file only needs to state what actually
    differs)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Config:
    """Load a YAML config file. Supports a single-level ``extends: <path>``
    key (relative to the file itself) so an experiment config can inherit
    from a base config and only override what differs - used sparingly,
    only where the source material showed one experiment was a direct
    variant of another (e.g. the cosine+EMA 65- and 100-epoch checkpoints
    of the same training run). Overrides are deep-merged, so an override
    file only needs to list the specific keys that change, not every key
    in a nested section."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if "extends" in raw:
        base_path = os.path.join(os.path.dirname(path), raw.pop("extends"))
        base = load_config(base_path)
        merged = _deep_merge(dict(base), raw)
        return _wrap(merged)

    return _wrap(raw)
