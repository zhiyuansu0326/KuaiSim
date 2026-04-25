import argparse
import ast
import glob
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

import utils

TASK_REQUEST = "request"
TASK_WHOLE = "whole"
TASK_CROSS = "cross"
ALL_TASKS = [TASK_REQUEST, TASK_WHOLE, TASK_CROSS]

LEGACY_ARG_DEFAULTS: Dict[str, Any] = {
    "intra_slate_metric": "ILD",
}


def apply_legacy_config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in LEGACY_ARG_DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg


@dataclass
class ParsedRow:
    step: int
    online: Dict[str, Any]
    train: Dict[str, Any]


def load_class(package_name: str, class_name: str):
    module = importlib.import_module(f"{package_name}.{class_name}")
    return getattr(module, class_name)


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
    online = ast.literal_eval(online_raw.split(":", 1)[1].strip())
    train = ast.literal_eval(train_raw.split(":", 1)[1].strip())
    if not isinstance(online, dict) or not isinstance(train, dict):
        raise ValueError(f"Malformed report dicts: {line}")
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
        raise ValueError(f"No valid rows found in report: {report_path}")
    return rows


def read_metric(row: ParsedRow, keys: Iterable[str]) -> float:
    for k in keys:
        src, key = k.split(".", 1)
        d = row.online if src == "online" else row.train
        if key in d:
            v = to_float(d[key])
            if is_finite(v):
                return v
    return float("nan")


def extract_rates_from_row(row: ParsedRow) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for d in (row.online, row.train):
        for k, v in d.items():
            if k.endswith("_rate"):
                rates[k] = to_float(v)
    return rates


def select_row(task: str, rows: List[ParsedRow], mode: str) -> ParsedRow:
    if mode == "last":
        return rows[-1]

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
        if mode == "best_avg_reward":
            key = read_metric(r, ["online.avg_reward", "train.avg_reward"])
        elif mode == "best_total_reward":
            key = read_metric(r, ["train.avg_total_reward", "online.avg_total_reward"])
        elif mode == "best_return_day":
            rd = read_metric(
                r, ["train.avg_retention", "online.return_day", "train.return_day"]
            )
            key = -rd if is_finite(rd) else float("nan")
        else:
            raise ValueError(f"Unknown select mode: {mode}")
        if is_finite(key):
            scored.append((key, r))
    if not scored:
        return rows[-1]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def parse_namespace_line(line: str) -> Optional[Dict[str, Any]]:
    text = line.strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        return None
    try:
        parsed = eval(
            text, {"__builtins__": {}}, {"Namespace": lambda **kwargs: kwargs}
        )
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def load_run_config(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log_path = run_dir / "log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

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
        raise ValueError(f"Cannot find initial class Namespace in log: {log_path}")
    if full_cfg is None:
        raise ValueError(
            f"Cannot find final args Namespace(save_path=...) in log: {log_path}"
        )
    apply_legacy_config_defaults(full_cfg)
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


def apply_legacy_arg_defaults(args: argparse.Namespace) -> None:
    apply_legacy_config_defaults(vars(args))


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


def mean_of(values: Iterable[float], drop_zero_head: bool = False) -> float:
    vals = list(values)
    if drop_zero_head and vals and vals[0] == 0:
        vals = vals[1:]
    vals = [to_float(v) for v in vals if is_finite(to_float(v))]
    if not vals:
        return float("nan")
    return mean(vals)


def build_agent_from_run(
    run_dir: Path,
    device: str,
    run_cfg: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    merac_eval_gumbel: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    if run_cfg is None:
        init_cfg, full_cfg = load_run_config(run_dir)
    else:
        init_cfg, full_cfg = run_cfg

    if merac_eval_gumbel and init_cfg.get("policy_class") == "MERACompletePolicy":
        full_cfg = dict(full_cfg)
        full_cfg["merac_eval_gumbel"] = True

    env_class = load_class("env", init_cfg["env_class"])
    policy_class = load_class("model.policy", init_cfg["policy_class"])
    agent_class = load_class("model.agent", init_cfg["agent_class"])
    buffer_class = load_class("model.buffer", init_cfg["buffer_class"])
    critic_class = (
        load_class("model.critic", init_cfg["critic_class"])
        if "critic_class" in init_cfg
        else None
    )

    args = argparse.Namespace(**full_cfg)
    apply_legacy_arg_defaults(args)
    args.device = device
    if device.startswith("cuda:"):
        args.cuda = int(device.split(":")[1])
    else:
        args.cuda = -1
    utils.set_random_seed(getattr(args, "seed", 11))

    env = env_class(args)
    policy = policy_class(args, env).to(device)

    if critic_class is None:
        raise ValueError(
            "Validation runner currently supports actor-critic checkpoints only."
        )

    td3_like_agents = {"TD3", "CrossSessionTD3"}
    if init_cfg["agent_class"] in td3_like_agents:
        critic1 = critic_class(args, env, policy).to(device)
        critic2 = critic_class(args, env, policy).to(device)
        critic = [critic1, critic2]
    else:
        critic = critic_class(args, env, policy).to(device)

    buffer = buffer_class(args, env, policy, critic)
    agent = agent_class(args, env, policy, critic, buffer)
    agent.load()
    return agent, init_cfg, full_cfg


def run_validation_episode_rollout(agent: Any, eval_steps: int) -> Dict[str, float]:
    # BaseRLAgent family monitor initialization.
    if hasattr(agent, "setup_monitors"):
        agent.setup_monitors()
    else:
        raise ValueError("Agent does not support setup_monitors for validation.")

    # Ensure deterministic validation behavior when dropout/batchnorm exist.
    if hasattr(agent, "actor"):
        agent.actor.eval()
    if hasattr(agent, "critic"):
        if isinstance(agent.critic, list):
            for c in agent.critic:
                c.eval()
        else:
            agent.critic.eval()
    if hasattr(agent, "critic1"):
        agent.critic1.eval()
    if hasattr(agent, "critic2"):
        agent.critic2.eval()

    obs = agent.env.reset()
    for i in range(eval_steps):
        obs = agent.run_episode_step(i, 0.0, obs, False, False)

    env_report = agent.env.get_report(eval_steps)
    eval_hist = agent.eval_history

    out: Dict[str, float] = {}
    # Whole-session metrics
    if "coverage" in env_report and "ILD" in env_report and "step" in env_report:
        out["Depth"] = to_float(env_report["step"])
        out["Average reward"] = mean_of(eval_hist.get("avg_reward", []))
        out["Total reward"] = mean_of(
            eval_hist.get("avg_total_reward", []), drop_zero_head=True
        )
        out["Coverage"] = to_float(env_report["coverage"])
        out["ILD"] = to_float(env_report["ILD"])

    # Cross-session metrics
    if "avg_retention" in eval_hist:
        out["Return day"] = mean_of(eval_hist.get("avg_retention", []))
    elif "return_day" in env_report:
        out["Return day"] = to_float(env_report["return_day"])

    # User retention may not be logged in this codebase.
    out["User retention"] = (
        to_float(eval_hist["user_retention"])
        if isinstance(eval_hist.get("user_retention"), (int, float))
        else float("nan")
    )

    # Extra behavior rates
    for k, v in eval_hist.items():
        if k.endswith("_rate"):
            out[k] = mean_of(v)

    agent.env.stop()
    return out


def evaluate_request_from_test_report(
    run_dir: Path, select_mode: str
) -> Dict[str, float]:
    report_path = run_dir / "model_test.report"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing request test report: {report_path}")
    rows = load_rows(report_path)
    row = select_row(TASK_REQUEST, rows, select_mode)

    metrics: Dict[str, float] = {
        "Average L-reward": read_metric(row, ["online.avg_reward", "train.avg_reward"]),
        "Max L-reward": read_metric(row, ["online.max_reward", "train.max_reward"]),
        "Coverage": read_metric(row, ["online.coverage", "train.coverage"]),
        "ILD": read_metric(
            row,
            [
                "online.EILD",
                "online.ILD",
                "online.intra_slate_diversity",
                "train.EILD",
                "train.ILD",
                "train.intra_slate_diversity",
            ],
        ),
    }
    metrics.update(extract_rates_from_row(row))
    return metrics


def parse_baseline_specs(task: str, specs: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {t: {} for t in ALL_TASKS}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --baseline: {spec}")
        left, pattern = spec.split("=", 1)
        if task == "all":
            if ":" not in left:
                raise ValueError(f"For --task all, use TASK:NAME=GLOB. Got: {spec}")
            t, name = left.split(":", 1)
            t = t.strip().lower()
            if t not in ALL_TASKS:
                raise ValueError(f"Unknown task in baseline spec: {t}")
            out[t][name.strip()] = pattern.strip()
        else:
            out[task][left.strip()] = pattern.strip()
    return out


def aggregate_metric(values: List[float]) -> Tuple[float, float]:
    finite = [v for v in values if is_finite(v)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return mean(finite), stdev(finite)


def required_cols(task: str) -> List[str]:
    if task == TASK_REQUEST:
        return ["Average L-reward", "Max L-reward", "Coverage", "ILD"]
    if task == TASK_WHOLE:
        return ["Depth", "Average reward", "Total reward", "Coverage", "ILD"]
    if task == TASK_CROSS:
        return ["Return day", "User retention"]
    raise ValueError(task)


def fmt(mu: float, sd: float) -> str:
    if not is_finite(mu):
        return "NA"
    if not is_finite(sd) or sd == 0:
        return f"{mu:.4f}"
    return f"{mu:.4f} +- {sd:.4f}"


def evaluate_task(
    task: str,
    baseline_globs: Dict[str, str],
    select_mode: str,
    eval_steps: int,
    device: str,
    expected_cfg: Dict[str, Any],
    expect_n: int,
    merac_eval_gumbel: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    need_expect_check = bool(expected_cfg) or expect_n >= 0
    for baseline, run_glob in baseline_globs.items():
        run_dirs = sorted({Path(p) for p in glob.glob(run_glob) if Path(p).is_dir()})
        if not run_dirs:
            rows.append(
                {"task": task, "baseline": baseline, "n_runs": 0, "metrics": {}}
            )
            continue

        run_metrics: List[Dict[str, float]] = []
        for run_dir in run_dirs:
            run_cfg: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
            if task != TASK_REQUEST or need_expect_check:
                run_cfg = load_run_config(run_dir)
            if need_expect_check:
                if run_cfg is None:
                    run_cfg = load_run_config(run_dir)
                _, full_cfg = run_cfg
                enforce_run_expectations(
                    run_dir=run_dir,
                    full_cfg=full_cfg,
                    expected_cfg=expected_cfg,
                    expect_n=expect_n,
                )
            if task == TASK_REQUEST:
                m = evaluate_request_from_test_report(run_dir, select_mode)
            else:
                agent, _, _ = build_agent_from_run(
                    run_dir,
                    device=device,
                    run_cfg=run_cfg,
                    merac_eval_gumbel=merac_eval_gumbel,
                )
                m = run_validation_episode_rollout(agent, eval_steps=eval_steps)
            run_metrics.append(m)

        cols = required_cols(task)
        extra = sorted(
            {k for m in run_metrics for k in m.keys() if k.endswith("_rate")}
        )
        cols.extend(extra)

        agg: Dict[str, Tuple[float, float]] = {}
        for c in cols:
            mu, sd = aggregate_metric([m.get(c, float("nan")) for m in run_metrics])
            agg[c] = (mu, sd)

        rows.append(
            {
                "task": task,
                "baseline": baseline,
                "n_runs": len(run_metrics),
                "metrics": agg,
            }
        )
    return rows


def print_table(task: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n=== {task.upper()} VALIDATION ===")
    if not rows:
        print("No baseline mapping.")
        return
    cols = required_cols(task)
    extra = sorted(
        {k for r in rows for k in r.get("metrics", {}) if k.endswith("_rate")}
    )
    cols.extend(extra)
    head = ["Baseline", "Runs"] + cols
    print(" | ".join(head))
    print(" | ".join(["---"] * len(head)))
    for r in rows:
        line = [r["baseline"], str(r["n_runs"])]
        for c in cols:
            if c in r.get("metrics", {}):
                mu, sd = r["metrics"][c]
                line.append(fmt(mu, sd))
            else:
                line.append("NA")
        print(" | ".join(line))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation benchmark entry. "
            "Request uses model_test.report; whole/cross run no-explore checkpoint validation."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", TASK_REQUEST, TASK_WHOLE, TASK_CROSS],
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="NAME=RUN_DIR_GLOB, or TASK:NAME=RUN_DIR_GLOB when --task all.",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="auto",
        choices=[
            "auto",
            "last",
            "best_avg_reward",
            "best_total_reward",
            "best_return_day",
        ],
        help="Selection mode for request test report rows.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Validation rollout steps for whole/cross tasks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Validation device, e.g. cpu or cuda:0.",
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
    parser.add_argument(
        "--merac_eval_gumbel",
        action="store_true",
        help="Force MERACompletePolicy to apply Gumbel perturbation during validation rollouts.",
    )
    args = parser.parse_args()

    baseline_map = parse_baseline_specs(args.task, args.baseline)
    expected_cfg = parse_expect_arg_specs(args.expect_arg)
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    for task in tasks:
        task_rows = evaluate_task(
            task=task,
            baseline_globs=baseline_map[task],
            select_mode=args.select,
            eval_steps=args.eval_steps,
            device=args.device,
            expected_cfg=expected_cfg,
            expect_n=args.expect_N,
            merac_eval_gumbel=args.merac_eval_gumbel,
        )
        print_table(task, task_rows)


if __name__ == "__main__":
    main()
