"""Reusable episode-level diagnostics recording and aggregation for evaluation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _metric_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


class EvalDiagnosticsRecorder:
    """Persist per-episode task diagnostics and build task-agnostic summaries.

    Tasks can override ``Base_Task.get_eval_diagnostics``. The evaluator remains
    unaware of task predicates and only aggregates the common schema.
    """

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.episodes_path = self.save_dir / "episode_diagnostics.jsonl"
        self.summary_path = self.save_dir / "diagnostics_summary.json"
        self.records: list[dict[str, Any]] = []
        self.episodes_path.write_text("", encoding="utf-8")

    def record_episode(
        self,
        task_env: Any,
        episode_id: int,
        seed: int,
        success: bool,
    ) -> dict[str, Any]:
        get_diagnostics = getattr(task_env, "get_eval_diagnostics", None)
        diagnostics = get_diagnostics(success=success) if callable(get_diagnostics) else {}
        if not isinstance(diagnostics, dict):
            raise TypeError("get_eval_diagnostics() must return a dictionary")

        diagnostics = json_safe(diagnostics)
        diagnostics.setdefault("schema_version", SCHEMA_VERSION)
        diagnostics.setdefault("task_name", getattr(task_env, "task_name", None))
        diagnostics.setdefault("success", bool(success))
        diagnostics.setdefault(
            "primary_failure_reason",
            "success" if success else "unspecified_failure",
        )
        diagnostics.setdefault("conditions", {})
        diagnostics.setdefault("metrics", {})
        diagnostics.setdefault("events", [])

        record = {
            "episode_id": int(episode_id),
            "seed": int(seed),
            "result": "Success" if success else "Fail",
            "diagnostics": diagnostics,
        }
        self.records.append(record)
        with self.episodes_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def summarize(self) -> dict[str, Any]:
        episode_count = len(self.records)
        success_count = sum(
            bool(record["diagnostics"].get("success")) for record in self.records
        )
        failure_count = episode_count - success_count

        reason_counts = Counter(
            str(record["diagnostics"].get("primary_failure_reason", "unspecified_failure"))
            for record in self.records
            if not record["diagnostics"].get("success")
        )
        failure_reasons = {
            reason: {
                "count": count,
                "episode_rate": count / episode_count if episode_count else 0.0,
                "failure_rate": count / failure_count if failure_count else 0.0,
            }
            for reason, count in sorted(reason_counts.items())
        }

        condition_values: dict[str, list[bool]] = defaultdict(list)
        metric_values: dict[str, list[float]] = defaultdict(list)
        event_counts: Counter[str] = Counter()
        for record in self.records:
            diagnostics = record["diagnostics"]
            for name, value in diagnostics.get("conditions", {}).items():
                if isinstance(value, (bool, np.bool_)):
                    condition_values[str(name)].append(bool(value))
            for name, value in diagnostics.get("metrics", {}).items():
                if _finite_number(value):
                    metric_values[str(name)].append(float(value))
            for event in diagnostics.get("events", []):
                if isinstance(event, dict) and event.get("name") is not None:
                    event_counts[str(event["name"])] += 1

        conditions = {
            name: {
                "true_count": sum(values),
                "evaluated_count": len(values),
                "true_rate": sum(values) / len(values),
            }
            for name, values in sorted(condition_values.items())
        }
        metrics = {
            name: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sorted(metric_values.items())
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "episode_count": episode_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / episode_count if episode_count else 0.0,
            "failure_reasons": failure_reasons,
            "conditions": conditions,
            "metrics": metrics,
            "event_counts": dict(sorted(event_counts.items())),
        }

    def write_summary(self) -> dict[str, Any]:
        summary = self.summarize()
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        return summary

    @staticmethod
    def format_summary(summary: dict[str, Any]) -> str:
        lines = ["Diagnostics:"]
        failure_reasons = summary.get("failure_reasons", {})
        if failure_reasons:
            lines.append("  Failure reasons:")
            for reason, values in failure_reasons.items():
                lines.append(
                    f"    {reason}: {values['count']} "
                    f"({values['failure_rate']:.3f} of failures)"
                )
        else:
            lines.append("  Failure reasons: none")

        conditions = summary.get("conditions", {})
        if conditions:
            lines.append("  Condition rates:")
            for name, values in conditions.items():
                lines.append(
                    f"    {name}: {values['true_count']}/{values['evaluated_count']} "
                    f"({values['true_rate']:.3f})"
                )
        return "\n".join(lines) + "\n"

    @staticmethod
    def wandb_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for reason, values in summary.get("failure_reasons", {}).items():
            key = _metric_key(reason)
            metrics[f"eval/failure_reason/{key}_count"] = int(values["count"])
            metrics[f"eval/failure_reason/{key}_rate"] = float(values["failure_rate"])
        for name, values in summary.get("conditions", {}).items():
            metrics[f"eval/condition/{_metric_key(name)}"] = float(values["true_rate"])
        for name, values in summary.get("metrics", {}).items():
            metrics[f"eval/diagnostic/{_metric_key(name)}_mean"] = float(values["mean"])
        return metrics
