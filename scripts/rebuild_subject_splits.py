#!/usr/bin/env python3
"""Rebuild a dataset into deterministic subject-level partitions.

The source may already contain train/val/test folders. Eligible images are
merged by subject identifier, then copied once into a fresh destination. Masks
and auxiliary files remain with the image because they share its subject group.
"""
import argparse
import hashlib
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
    case_match = re.match(r"case\d+", stem, re.I)
    if case_match:
        return case_match.group(0).lower()
    return re.sub(r"_view\d+$", "", stem, flags=re.I).lower()


def collect(source):
    grouped = defaultdict(list)
    for split in ("train", "val", "test"):
        for path in sorted((source / split).glob("*/*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                grouped[(path.parent.name, subject_id(path))].append(path)
    if not grouped:
        raise SystemExit(f"No eligible images found under {source}/train, val, or test")
    return grouped


def allocate(subjects, rng, train_ratio, val_ratio):
    values = sorted(subjects)
    rng.shuffle(values)
    train_end = round(len(values) * train_ratio)
    val_end = train_end + round(len(values) * val_ratio)
    return {"train": values[:train_end], "val": values[train_end:val_end], "test": values[val_end:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("breast", "oasbud"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.data_dir / args.dataset
    output = args.output or args.data_dir / f"{args.dataset}.subject_clean"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    grouped = collect(source)
    by_class = defaultdict(dict)
    for (class_name, identifier), paths in grouped.items():
        by_class[class_name][identifier] = paths
    rng = random.Random(args.seed)
    manifest = {
        "dataset": args.dataset,
        "source": str(source.resolve()),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio},
        "splits": {},
    }
    for class_name, subjects in sorted(by_class.items()):
        allocation = allocate(subjects, rng, args.train_ratio, args.val_ratio)
        for split, identifiers in allocation.items():
            manifest["splits"].setdefault(split, {})[class_name] = sorted(identifiers)
            target = output / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for identifier in identifiers:
                for path in subjects[identifier]:
                    shutil.copy2(path, target / path.name)
    manifest["file_count"] = sum(
        1 for path in output.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "split_manifest.json").write_text(manifest_text, encoding="utf-8")
    print(f"Created {output} with {manifest['file_count']} image/mask files")
    print(f"Manifest SHA-256: {hashlib.sha256(manifest_text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
