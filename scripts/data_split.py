#!/usr/bin/env python3
"""Create a deterministic subject-level image split without altering source files."""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_SUFFIXES = ("_mask", "_tumor", "_lesion_mask")


def subject_id(path):
    stem = path.stem
    for suffix in MASK_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    match = re.match(r"(case\d+|[^_]+)(?:_view\d+)?$", stem, re.I)
    return (match.group(1) if match else stem).lower()


def allocate(subjects, rng, train_ratio, val_ratio):
    values = sorted(subjects)
    rng.shuffle(values)
    train_end = round(len(values) * train_ratio)
    val_end = train_end + round(len(values) * val_ratio)
    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing one subfolder per class")
    parser.add_argument("destination", type=Path, help="New output folder for train/val/test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Ratios must leave a positive test partition.")
    if not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}")
    if args.destination.exists() and any(args.destination.rglob("*")):
        raise SystemExit(f"Destination must be empty: {args.destination}")

    grouped = defaultdict(lambda: defaultdict(list))
    for class_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                grouped[class_dir.name][subject_id(path)].append(path)

    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "source": str(args.source.resolve()), "splits": {}}
    for class_name, subjects in grouped.items():
        split_subjects = allocate(subjects, rng, args.train_ratio, args.val_ratio)
        for split, identifiers in split_subjects.items():
            for identifier in identifiers:
                manifest["splits"].setdefault(split, {}).setdefault(class_name, []).append(identifier)
                target_dir = args.destination / split / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                for source_path in subjects[identifier]:
                    shutil.copy2(source_path, target_dir / source_path.name)

    manifest_path = args.destination / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created subject-level split at {args.destination} with manifest {manifest_path}")


if __name__ == "__main__":
    main()
