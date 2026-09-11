#!/usr/bin/env python3
"""Evaluate the available checkpoints on the same held-out split.

This is an internal architecture benchmark. It does not download or compare
against published scores, because those are not comparable without identical
data, subject partitions, preprocessing, and training budgets.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--models", nargs="+", default=("efficientnet_b0", "resnet50", "custom_cnn"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_benchmark.json")
    args = parser.parse_args()
    records = []
    for model_name in args.models:
        checkpoint = ROOT / "outputs" / f"{model_name}_model.pth"
        if not checkpoint.exists():
            raise SystemExit(f"Missing checkpoint: {checkpoint}. Train {model_name} first.")
        env = os.environ.copy()
        env.setdefault("TRAIN_DATASETS", "breast,oasbud")
        subprocess.run(
            [sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--split", args.split],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
        records.append({
            "model_name": metrics["model_name"],
            "checkpoint": metrics["checkpoint"],
            "split": metrics["split"],
            "samples": metrics["samples"],
            "datasets": metrics["datasets"],
            "data_audit_passed": metrics["data_audit_passed"],
            "summary": metrics["summary"],
        })
    payload = {
        "description": "Internal same-split architecture benchmark; not a clinical or published-score comparison.",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
