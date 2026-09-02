"""Experiment-directory management and device selection.

No path is hardcoded to ``/kaggle/...`` anywhere in this module - every
root (``output_root``, ``data_root``, etc.) is a parameter sourced from the
active YAML config. Point those config values at Kaggle paths, a local
directory, or anything else; the code itself is environment-agnostic.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import torch

__all__ = ["make_experiment_dir", "resolve_device"]


def make_experiment_dir(output_root: str, experiment_name: str, resume_dir: str | None = None) -> str:
    """Create (or reuse, if resuming) a unique experiment directory:

        output_root/experiment_name_<YYYYmmdd-HHMMSS>_<8-char-id>/

    Pass ``resume_dir`` (an existing experiment directory path) to continue
    training in-place rather than starting a new directory.
    """
    if resume_dir is not None:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"resume_dir does not exist: {resume_dir}")
        return resume_dir

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    exp_dir = os.path.join(output_root, f"{experiment_name}_{stamp}_{short_id}")
    os.makedirs(exp_dir, exist_ok=False)
    os.makedirs(os.path.join(exp_dir, "generated_samples"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "denoising"), exist_ok=True)
    return exp_dir


def resolve_device(preferred: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA if available, else CPU. Otherwise honors the
    explicit request (``"cuda"``, ``"cpu"``, ``"mps"``, ...)."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)
