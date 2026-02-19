"""
Sanity check: list distinct machine_ids per machine_type in the DCASE dataset.
Usage: python -m src.utils.list_machine_ids --data_root /path/to/64mel-spectr-dcase2020-task2-dev-dataset [--split train]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset import get_machine_id_summary


def main() -> None:
    p = argparse.ArgumentParser(description="List machine_ids per machine_type (sanity check)")
    p.add_argument("--data_root", type=str, required=True, help="Root path to DCASE mel-spectrogram dataset")
    p.add_argument("--split", type=str, default="train", choices=("train", "test"), help="Split to scan")
    args = p.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"Data root not found: {root}")

    summary = get_machine_id_summary(root, split=args.split)
    if not summary:
        print(f"No .npy files found under {root}/<machine_type>/{args.split}")
        return

    print(f"machine_ids per machine_type (split={args.split}, data_root={root}):\n")
    for type_name in sorted(summary.keys()):
        counts = summary[type_name]
        ids = sorted(counts.keys())
        total = sum(counts.values())
        print(f"  {type_name}:")
        print(f"    machine_ids: {ids}  (num_classes = {len(ids)}, max_id = {max(ids)})")
        print(f"    counts:      {counts}")
        print(f"    total files: {total}")
        print()
    print("Dataset exposes contiguous machine_id 0..K-1 per type; num_machine_ids = K (inferred in trainer).")


if __name__ == "__main__":
    main()
