"""Shared mutable state for the BERTimbau fine-tuning process.

Stored in a dedicated module so it survives Streamlit's per-rerun page
re-execution while remaining accessible from background training threads.
Python's module cache ensures this dict is initialized exactly once.
"""
from __future__ import annotations

from typing import Any, Dict

state: Dict[str, Any] = {
    "running": False,
    "done": False,
    "error": None,
    "current_fold": 0,
    "current_epoch": {},  # fold (int) → (epoch_num, total_epochs)
    "fold_logs": {},      # fold (int) → List[Dict] with per-epoch metrics
    "fold_results": [],   # List[{fold, val_f1_macro}]
}


def reset() -> None:
    state.update({
        "running": False,
        "done": False,
        "error": None,
        "current_fold": 0,
        "current_epoch": {},
        "fold_logs": {},
        "fold_results": [],
    })
