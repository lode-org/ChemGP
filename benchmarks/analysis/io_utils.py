"""I/O helpers for benchmark case ingestion."""

from __future__ import annotations

from pathlib import Path


def load_con_frames(path: str | Path):
    """Load an eOn-style .con trajectory via readcon-core Python bindings."""
    import readcon

    return readcon.read_con(str(path))


def frame_count(path: str | Path) -> int:
    """Return the number of frames in a .con file."""
    return len(load_con_frames(path))
