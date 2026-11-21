"""Load the cloned MiniT2I code and, when the snapshot is present, the B/16 weights."""

from __future__ import annotations

import sys
from pathlib import Path

from src.paths import THIRD_PARTY

MINIT2I_ROOT = THIRD_PARTY / "minit2i-torch"
DIFFUSERS_DIR = MINIT2I_ROOT / "diffusers"
HF_REPO = "MiniT2I/MiniT2I"
HF_SUBFOLDER = "minit2i-b-16"


def ensure_minit2i_path() -> Path:
    if not DIFFUSERS_DIR.exists():
        raise FileNotFoundError(f"missing clone: {DIFFUSERS_DIR}")
    text = str(DIFFUSERS_DIR)
    if text not in sys.path:
        sys.path.insert(0, text)
    return DIFFUSERS_DIR


def import_official_modules():
    ensure_minit2i_path()
    import mmdit  # type: ignore
    import pipeline  # type: ignore

    return mmdit, pipeline


def load_b16_transformer(snapshot: Path, dtype=None):
    """Load MiniT2I-B/16 from a local snapshot of the HF subfolder."""
    mmdit, pipeline = import_official_modules()
    transformer_dir = snapshot / "transformer"
    if not transformer_dir.exists():
        transformer_dir = snapshot
    model = pipeline.MiniT2IMMJiTModel.from_pretrained(transformer_dir, torch_dtype=dtype)
    return model, mmdit
