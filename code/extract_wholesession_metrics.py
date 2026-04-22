import argparse
import ast
import glob
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


@dataclass
class MetricRow:
    step: int
    depth: float
    avg_reward: float
    total_reward: float
    coverage: float
    ild: float


def parse_report_line(line: str) -> MetricRow:
    # Expected format:
    # step: 1990 @ online episode: {...} @ training: {...}
    parts = line.strip().split("@")
    if len(parts) < 3:
        raise ValueError(f"Malformed report line: {line}")

    step = int(parts[0].split(":")[1].strip())
    online_raw = parts[1].strip()
    train_raw = parts[2].strip()

    if not online_raw.startswith("online episode:"):
        raise ValueError(f"Missing online episode block: {line}")
    if not train_raw.startswith("training:"):
        raise ValueError(f"Missing training block: {line}")

    online = ast.literal_eval(online_raw[len("online episode:") :].strip())
    train = ast.literal_eval(train_raw[len("training:") :].strip())

    return MetricRow(
        step=step,
        depth=float(online["step"]),
        avg_reward=float(train["avg_reward"]),
        total_reward=float(train["avg_total_reward"]),
        coverage=float(online["coverage"]),
        ild=float(online["ILD"]),
    )


def load_rows(report_path: Path) -> List[MetricRow]:
    rows: List[MetricRow] = []
    with report_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("step:"):
                continue
            rows.append(parse_report_line(line))
    if not rows:
        raise ValueError(f"No metric rows found in report: {report_path}")
    return rows


def select_row(rows: List[MetricRow], mode: str) -> MetricRow:
    if mode == "last":
        return rows[-1]
    if mode == "best_total_reward":
        return max(rows, key=lambda x: x.total_reward)
    if mode == "best_avg_reward":
        return max(rows, key=lambda x: x.avg_reward)
    raise ValueError(f"Unknown select mode: {mode}")


def row_to_dict(row: MetricRow) -> Dict[str, float]:
    return {
        "step": row.step,
        "Depth": row.depth,
        "Average reward": row.avg_reward,
        "Total reward": row.total_reward,
        "Coverage": row.coverage,
        "ILD": row.ild,
    }


def print_single(report: Path, row: MetricRow) -> None:
    metrics = row_to_dict(row)
    print(f"report: {report}")
    print(
        "step={step} | Depth={Depth:.4f} | Average reward={Average reward:.4f} | "
        "Total reward={Total reward:.4f} | Coverage={Coverage:.4f} | ILD={ILD:.4f}".format(
            **metrics
        )
    )


def print_aggregate(rows: List[MetricRow]) -> None:
    depth = [r.depth for r in rows]
    avg_reward = [r.avg_reward for r in rows]
    total_reward = [r.total_reward for r in rows]
    coverage = [r.coverage for r in rows]
    ild = [r.ild for r in rows]

    def fmt(values: List[float]) -> str:
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{mean(values):.4f} +- {stdev(values):.4f}"

    print("aggregate over selected runs:")
    print(f"Depth: {fmt(depth)}")
    print(f"Average reward: {fmt(avg_reward)}")
    print(f"Total reward: {fmt(total_reward)}")
    print(f"Coverage: {fmt(coverage)}")
    print(f"ILD: {fmt(ild)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract whole-session benchmark metrics from model.report."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="",
        help="Path to one model.report file.",
    )
    parser.add_argument(
        "--report_glob",
        type=str,
        default="",
        help="Glob for multiple report files (e.g., output/Kuairand_Pure/agents/*/model.report).",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="best_total_reward",
        choices=["last", "best_total_reward", "best_avg_reward"],
        help="How to select one row from each report.",
    )
    args = parser.parse_args()

    if not args.report_path and not args.report_glob:
        raise ValueError("Provide either --report_path or --report_glob.")

    report_files: List[Path] = []
    if args.report_path:
        report_files.append(Path(args.report_path))
    if args.report_glob:
        report_files.extend(Path(p) for p in glob.glob(args.report_glob))

    report_files = sorted(set(report_files))
    if not report_files:
        raise ValueError("No report files matched.")

    selected_rows: List[MetricRow] = []
    for report in report_files:
        rows = load_rows(report)
        row = select_row(rows, args.select)
        selected_rows.append(row)
        print_single(report, row)

    if len(selected_rows) > 1:
        print_aggregate(selected_rows)


if __name__ == "__main__":
    main()
