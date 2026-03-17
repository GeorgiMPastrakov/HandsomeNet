"""Run the remaining HandsomeNet training milestones with one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PlannedRun:
    """One training run in the HandsomeNet training plan."""

    label: str
    description: str
    args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/freihand"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-overfit",
        action="store_true",
        help="Also rerun the baseline overfit milestone before the short full-split runs.",
    )
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> list[PlannedRun]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan: list[PlannedRun] = []

    if args.include_overfit:
        plan.append(
            PlannedRun(
                label="baseline_overfit",
                description="Baseline overfit proof on 32 unique IDs.",
                args=[
                    "--model",
                    "baseline",
                    "--device",
                    args.device,
                    "--epochs",
                    "50",
                    "--batch-size",
                    "8",
                    "--num-workers",
                    "0",
                    "--limit-train-unique",
                    "32",
                    "--limit-val-unique",
                    "8",
                    "--stop-train-pixel-error",
                    "5",
                    "--seed",
                    str(args.seed),
                    "--data-root",
                    str(args.data_root),
                    "--run-name",
                    f"{timestamp}_baseline_overfit",
                ],
            )
        )

    plan.extend(
        [
            PlannedRun(
                label="baseline_short",
                description="Baseline short full-split validation run.",
                args=[
                    "--model",
                    "baseline",
                    "--device",
                    args.device,
                    "--epochs",
                    "5",
                    "--batch-size",
                    "32",
                    "--num-workers",
                    "0",
                    "--val-fraction",
                    "0.1",
                    "--seed",
                    str(args.seed),
                    "--data-root",
                    str(args.data_root),
                    "--run-name",
                    f"{timestamp}_baseline_short",
                ],
            ),
            PlannedRun(
                label="handsomenet_short",
                description="HandsomeNet short controlled run.",
                args=[
                    "--model",
                    "handsomenet",
                    "--device",
                    args.device,
                    "--epochs",
                    "5",
                    "--batch-size",
                    "16",
                    "--num-workers",
                    "0",
                    "--val-fraction",
                    "0.1",
                    "--seed",
                    str(args.seed),
                    "--data-root",
                    str(args.data_root),
                    "--run-name",
                    f"{timestamp}_handsomenet_short",
                ],
            ),
        ]
    )

    return plan


def main() -> None:
    args = parse_args()
    plan = build_plan(args)

    if args.dry_run:
        print(render_plan(plan))
        return

    summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path("artifacts/runs") / f"training_plan_{summary_timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary: list[dict[str, str | int]] = []

    for planned_run in plan:
        command = [sys.executable, "scripts/train.py", *planned_run.args]
        print(f"\n== Running {planned_run.label} ==")
        print(" ".join(command))
        completed = subprocess.run(command, check=False)

        record = {
            "label": planned_run.label,
            "description": planned_run.description,
            "command": " ".join(command),
            "returncode": completed.returncode,
        }
        summary.append(record)
        summary_path.write_text(json.dumps(summary, indent=2))

        if completed.returncode != 0:
            raise SystemExit(
                "Training plan stopped because "
                f"{planned_run.label} failed with code {completed.returncode}."
            )

    print(f"\nTraining plan completed. Summary: {summary_path}")


def render_plan(plan: list[PlannedRun]) -> str:
    lines = ["Handsomenet training plan:"]
    for run in plan:
        lines.append(f"- {run.label}: {run.description}")
        lines.append(f"  {sys.executable} scripts/train.py {' '.join(run.args)}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
