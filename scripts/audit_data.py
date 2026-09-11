#!/usr/bin/env python3
"""Audit labelled image splits without altering the source datasets."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def subject_id(dataset, filename):
    stem = Path(filename).stem
    if dataset == "oasbud":
        return stem.split("_view", 1)[0]
    if dataset == "breast":
        match = re.match(r"(case\d+)", stem, re.I)
        return match.group(1).lower() if match else stem
    return None  # BUSI renamed files cannot be checked without an original-ID manifest.


def audit_dataset(root, dataset):
    members = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(int))
    for image in (root / dataset).glob("*/*/*"):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        if image.stem.lower().endswith(("_tumor", "_mask", "_lesion_mask")):
            continue
        split, label = image.parts[-3], image.parts[-2]
        counts[split][label] += 1
        identifier = subject_id(dataset, image.name)
        if identifier:
            members[identifier].add(split)
    leakage = {key: sorted(value) for key, value in members.items() if len(value) > 1}
    return {
        "subject_ids_verifiable": dataset != "busi",
        "counts": counts,
        "cross_split_subjects": leakage,
    }


def build_report(root, datasets=None):
    """Return the canonical data-readiness report used by training and evaluation.

    With no selection, every local dataset is reported. A valid experiment may
    explicitly audit only datasets whose subject identifiers are recoverable.
    """
    selected = tuple(datasets or ("busi", "oasbud", "breast"))
    unknown = set(selected) - {"busi", "oasbud", "breast"}
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(sorted(unknown))}")
    report = {dataset: audit_dataset(root, dataset) for dataset in selected}
    failed = [
        dataset
        for dataset, result in report.items()
        if not result["subject_ids_verifiable"] or result["cross_split_subjects"]
    ]
    report["audit_passed"] = not failed
    report["audited_datasets"] = list(selected)
    report["failed_datasets"] = failed
    report["notes"] = [
        "BUSI filenames were renamed during curation, so patient-level separation cannot be verified from this copy.",
        "A non-empty cross_split_subjects section invalidates a patient-independent estimate until the cohort is rebuilt.",
        "A dataset omitted from an experiment must not be described as part of that experiment's evidence.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", choices=("busi", "oasbud", "breast"),
                        help="Datasets included in this experiment; default audits all local datasets.")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    root = Path(args.data_dir)
    report = build_report(root, args.datasets)
    text = json.dumps(report, indent=2, sort_keys=True, default=dict)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
