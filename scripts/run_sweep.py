#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-dir", default="runs/sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    for raw_config in args.configs:
        config = Path(raw_config)
        run_dir = output_dir / config.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "submission.json"
        command = [
            sys.executable,
            "-m",
            "vibiomir",
            "run",
            "--config",
            str(config),
            "--corpus",
            args.corpus,
            "--queries",
            args.queries,
            "--output",
            str(prediction_path),
        ]
        try:
            subprocess.run(command, check=True, env=environment)
            evaluate = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vibiomir",
                    "evaluate",
                    "--predictions",
                    str(prediction_path),
                    "--gold",
                    args.gold,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            metrics = json.loads(evaluate.stdout)
            rows.append({"config": str(config), "status": "ok", **metrics})
        except subprocess.CalledProcessError as exc:
            rows.append({"config": str(config), "status": f"failed:{exc.returncode}"})

        with (output_dir / "scoreboard.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

