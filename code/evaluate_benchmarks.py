import argparse
import ast
import csv
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_REQUEST = "request"
TASK_WHOLE = "whole"
TASK_CROSS = "cross"
ALL_TASKS = [TASK_REQUEST, TASK_WHOLE, TASK_CROSS]


@dataclass
class ParsedRow:
    step: int
    online: Dict[str, Any]
    train: Dict[str, Any]


def to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def parse_report_line(line: str) -> ParsedRow:
    parts = line.strip().split("@")
    if len(parts) < 3:
        raise ValueError(f"Malformed report line: {line}")

    step = int(parts[0].split(":", 1)[1].strip())

    online_raw = parts[1].strip()
    train_raw = parts[2].strip()

    if ":" not in online_raw or ":" not in train_raw:
        raise ValueError(f"Malformed online/training blocks: {line}")

    online = ast.literal_eval(online_raw.split(":", 1)[1].strip())
    train = ast.literal_eval(train_raw.split(":", 1)[1].strip())

    if not isinstance(online, dict) or not isinstance(train, dict):
        raise ValueError(f"Online/training is not dict: {line}")

    return ParsedRow(step=step, online=online, train=train)


def load_rows(report_path: Path) -> List[ParsedRow]:
    rows: List[ParsedRow] = []
    with report_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("step:"):
                continue
            try:
                rows.append(parse_report_line(line))
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No valid metric rows found in report: {report_path}")
    return rows


def read_metric(row: ParsedRow, candidates: Iterable[str]) -> float:
    for c in candidates:
        src, key = c.split(".", 1)
        d = row.online if src == "online" else row.train
        if key in d:
            v = to_float(d[key])
            if is_finite(v):
                return v
    return float("nan")


def extract_rates(row: ParsedRow) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d in (row.online, row.train):
        for k, v in d.items():
            if k.endswith("_rate"):
                out[k] = to_float(v)
    return out


def extract_task_metrics(task: str, row: ParsedRow) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if task == TASK_REQUEST:
        metrics["Average L-reward"] = read_metric(row, ["online.avg_reward", "train.avg_reward"])
        metrics["Max L-reward"] = read_metric(row, ["online.max_reward", "train.max_reward"])
        metrics["Coverage"] = read_metric(row, ["online.coverage", "train.coverage"])
        metrics["ILD"] = read_metric(
            row,
            [
                "online.EILD",
                "online.ILD",
                "online.intra_slate_diversity",
                "train.EILD",
                "train.ILD",
                "train.intra_slate_diversity",
            ],
        )
    elif task == TASK_WHOLE:
        metrics["Depth"] = read_metric(row, ["online.step", "train.step"])
        metrics["Average reward"] = read_metric(row, ["train.avg_reward", "online.avg_reward"])
        metrics["Total reward"] = read_metric(
            row, ["train.avg_total_reward", "online.avg_total_reward"]
        )
        metrics["Coverage"] = read_metric(row, ["online.coverage", "train.coverage"])
        metrics["ILD"] = read_metric(row, ["online.EILD", "online.ILD", "train.EILD", "train.ILD"])
    elif task == TASK_CROSS:
        metrics["Return day"] = read_metric(
            row, ["train.avg_retention", "online.return_day", "train.return_day"]
        )
        metrics["User retention"] = read_metric(
            row,
            [
                "train.user_retention",
                "online.user_retention",
                "train.retention_rate",
                "online.retention_rate",
                "train.avg_user_retention",
            ],
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    metrics.update(extract_rates(row))
    return metrics


def select_row(task: str, rows: List[ParsedRow], mode: str) -> ParsedRow:
    if mode == "last":
        return rows[-1]

    def key_for_row(r: ParsedRow) -> float:
        m = extract_task_metrics(task, r)
        if mode == "best_total_reward":
            return m.get("Total reward", float("nan"))
        if mode == "best_avg_reward":
            if task == TASK_REQUEST:
                return m.get("Average L-reward", float("nan"))
            return m.get("Average reward", float("nan"))
        if mode == "best_return_day":
            v = m.get("Return day", float("nan"))
            if not is_finite(v):
                return float("-inf")
            return -v
        raise ValueError(f"Unknown select mode: {mode}")

    if mode == "auto":
        if task == TASK_REQUEST:
            mode = "best_avg_reward"
        elif task == TASK_WHOLE:
            mode = "best_total_reward"
        elif task == TASK_CROSS:
            mode = "best_return_day"
        else:
            mode = "last"
        return select_row(task, rows, mode)

    scored: List[Tuple[float, ParsedRow]] = []
    for r in rows:
        k = key_for_row(r)
        if is_finite(k):
            scored.append((k, r))
    if not scored:
        return rows[-1]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def parse_namespace_line(line: str) -> Optional[Dict[str, Any]]:
    text = line.strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        return None
    try:
        parsed = eval(text, {"__builtins__": {}}, {"Namespace": lambda **kwargs: kwargs})
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def load_run_config(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log_path = run_dir / "log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file for report run: {log_path}")

    init_cfg: Optional[Dict[str, Any]] = None
    full_cfg: Optional[Dict[str, Any]] = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            ns = parse_namespace_line(raw)
            if ns is None:
                continue
            if (
                "env_class" in ns
                and "policy_class" in ns
                and "agent_class" in ns
                and "buffer_class" in ns
            ):
                init_cfg = ns
            if "save_path" in ns:
                full_cfg = ns

    if init_cfg is None:
        raise ValueError(f"Cannot find class namespace in log: {log_path}")
    if full_cfg is None:
        raise ValueError(f"Cannot find args namespace(save_path=...) in log: {log_path}")
    return init_cfg, full_cfg


def parse_expect_arg_specs(specs: List[str]) -> Dict[str, Any]:
    expected: Dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --expect_arg: {spec}. Use KEY=VALUE.")
        key, raw_val = spec.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            raise ValueError(f"Invalid --expect_arg key in: {spec}")
        try:
            val = ast.literal_eval(raw_val)
        except Exception:
            val = raw_val
        expected[key] = val
    return expected


def values_match(actual: Any, expected: Any, tol: float = 1e-8) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            s = actual.strip().lower()
            if s in {"true", "1"}:
                actual = True
            elif s in {"false", "0"}:
                actual = False
        return bool(actual) == expected

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            a = float(actual)
            b = float(expected)
            return abs(a - b) <= tol
        except Exception:
            return False

    return str(actual) == str(expected)


def enforce_run_expectations(
    run_dir: Path,
    full_cfg: Dict[str, Any],
    expected_cfg: Dict[str, Any],
    expect_n: int,
) -> None:
    checks: Dict[str, Any] = dict(expected_cfg)
    if expect_n >= 0:
        checks["slate_size"] = expect_n

    for key, expected in checks.items():
        if key not in full_cfg:
            raise ValueError(
                f"[{run_dir}] fairness check failed: missing arg '{key}' in training config."
            )
        actual = full_cfg[key]
        if not values_match(actual, expected):
            raise ValueError(
                f"[{run_dir}] fairness check failed: '{key}' mismatch, "
                f"expected={expected}, actual={actual}."
            )

def parse_baseline_specs(task: str, specs: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Returns:
        {task: {baseline_name: report_glob}}
    """
    baseline_map: Dict[str, Dict[str, str]] = {t: {} for t in ALL_TASKS}
    if not specs:
        return baseline_map

    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --baseline spec: {spec}. Use NAME=GLOB or TASK:NAME=GLOB.")
        left, pattern = spec.split("=", 1)
        if task == "all":
            if ":" not in left:
                raise ValueError(
                    f"Invalid --baseline spec for --task all: {spec}. Use TASK:NAME=GLOB."
                )
            task_name, baseline_name = left.split(":", 1)
            task_name = task_name.strip().lower()
            if task_name not in ALL_TASKS:
                raise ValueError(f"Unknown task in baseline spec: {task_name}")
            baseline_map[task_name][baseline_name.strip()] = pattern.strip()
        else:
            baseline_map[task][left.strip()] = pattern.strip()
    return baseline_map


def aggregate_metric(values: List[float]) -> Tuple[float, float]:
    finite = [v for v in values if is_finite(v)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return mean(finite), stdev(finite)


def required_cols_for_task(task: str) -> List[str]:
    if task == TASK_REQUEST:
        return ["Average L-reward", "Max L-reward", "Coverage", "ILD"]
    if task == TASK_WHOLE:
        return ["Depth", "Average reward", "Total reward", "Coverage", "ILD"]
    if task == TASK_CROSS:
        return ["Return day", "User retention"]
    raise ValueError(task)


def fmt_mean_std(mu: float, sd: float) -> str:
    if not is_finite(mu):
        return "NA"
    if not is_finite(sd):
        return f"{mu:.4f}"
    if sd == 0:
        return f"{mu:.4f}"
    return f"{mu:.4f} +- {sd:.4f}"


def evaluate_task(
    task: str,
    baseline_to_glob: Dict[str, str],
    select_mode: str,
    extra_rate_keys: List[str],
    expected_cfg: Dict[str, Any],
    expect_n: int,
) -> List[Dict[str, Any]]:
    task_rows: List[Dict[str, Any]] = []
    need_expect_check = bool(expected_cfg) or expect_n >= 0
    for baseline, report_glob in baseline_to_glob.items():
        report_files = sorted({Path(p) for p in glob.glob(report_glob)})
        if not report_files:
            task_rows.append(
                {
                    "task": task,
                    "baseline": baseline,
                    "n_runs": 0,
                    "metrics": {},
                }
            )
            continue

        selected_metrics: List[Dict[str, float]] = []
        for rp in report_files:
            if need_expect_check:
                run_dir = rp.parent
                _, full_cfg = load_run_config(run_dir)
                enforce_run_expectations(
                    run_dir=run_dir,
                    full_cfg=full_cfg,
                    expected_cfg=expected_cfg,
                    expect_n=expect_n,
                )
            rows = load_rows(rp)
            selected = select_row(task, rows, select_mode)
            selected_metrics.append(extract_task_metrics(task, selected))

        cols = required_cols_for_task(task)
        available_rate_keys = sorted(
            {
                k
                for m in selected_metrics
                for k in m.keys()
                if k.endswith("_rate") and (not extra_rate_keys or k in extra_rate_keys)
            }
        )
        cols.extend(available_rate_keys)

        agg: Dict[str, Tuple[float, float]] = {}
        for c in cols:
            mu, sd = aggregate_metric([m.get(c, float("nan")) for m in selected_metrics])
            agg[c] = (mu, sd)

        task_rows.append(
            {
                "task": task,
                "baseline": baseline,
                "n_runs": len(selected_metrics),
                "metrics": agg,
            }
        )
    return task_rows


def print_task_table(task: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n=== {task.upper()} ===")
    if not rows:
        print("No baselines specified.")
        return

    cols = required_cols_for_task(task)
    extra = sorted(
        {
            k
            for r in rows
            for k in r.get("metrics", {}).keys()
            if k.endswith("_rate")
        }
    )
    cols.extend(extra)

    header = ["Baseline", "Runs"] + cols
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    for r in rows:
        line = [r["baseline"], str(r["n_runs"])]
        for c in cols:
            if c in r.get("metrics", {}):
                mu, sd = r["metrics"][c]
                line.append(fmt_mean_std(mu, sd))
            else:
                line.append("NA")
        print(" | ".join(line))


def save_csv(path: Path, all_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    all_metric_names = sorted(
        {
            c
            for r in all_rows
            for c in r.get("metrics", {}).keys()
        }
    )

    fields = ["task", "baseline", "n_runs"]
    for c in all_metric_names:
        fields.append(f"{c}_mean")
        fields.append(f"{c}_std")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            row: Dict[str, Any] = {
                "task": r["task"],
                "baseline": r["baseline"],
                "n_runs": r["n_runs"],
            }
            for c in all_metric_names:
                if c in r.get("metrics", {}):
                    mu, sd = r["metrics"][c]
                    row[f"{c}_mean"] = mu
                    row[f"{c}_std"] = sd
                else:
                    row[f"{c}_mean"] = ""
                    row[f"{c}_std"] = ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified benchmark evaluation entry: aligns output to paper metrics and "
            "adds extra behavior rates (e.g., is_click_rate, is_hate_rate)."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", TASK_REQUEST, TASK_WHOLE, TASK_CROSS],
        help="Evaluate one task or all tasks.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Baseline spec. For single task: NAME=GLOB. "
            "For --task all: TASK:NAME=GLOB."
        ),
    )
    parser.add_argument(
        "--select",
        type=str,
        default="auto",
        choices=["auto", "last", "best_total_reward", "best_avg_reward", "best_return_day"],
        help="Row selection mode per report.",
    )
    parser.add_argument(
        "--extra_rate_keys",
        nargs="*",
        default=[],
        help=(
            "Optional whitelist for extra *_rate metrics. "
            "Empty means include all available rates."
        ),
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional path to save aggregated metrics CSV.",
    )
    parser.add_argument(
        "--expect_N",
        type=int,
        default=-1,
        help="Fairness constraint shortcut for slate_size (N). -1 means disabled.",
    )
    parser.add_argument(
        "--expect_arg",
        action="append",
        default=[],
        help=(
            "Additional fairness constraint as KEY=VALUE. "
            "Example: --expect_arg single_response=False --expect_arg reward_func='get_immediate_reward'."
        ),
    )
    args = parser.parse_args()

    baseline_map = parse_baseline_specs(args.task, args.baseline)
    expected_cfg = parse_expect_arg_specs(args.expect_arg)

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    all_rows: List[Dict[str, Any]] = []
    for t in tasks:
        rows = evaluate_task(
            task=t,
            baseline_to_glob=baseline_map[t],
            select_mode=args.select,
            extra_rate_keys=args.extra_rate_keys,
            expected_cfg=expected_cfg,
            expect_n=args.expect_N,
        )
        print_task_table(t, rows)
        all_rows.extend(rows)

    if args.save_csv:
        save_csv(Path(args.save_csv), all_rows)
        print(f"\nSaved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
