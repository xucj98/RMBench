import json
import tempfile
import unittest
from pathlib import Path

from script.eval_diagnostics import EvalDiagnosticsRecorder


class _FakeTask:
    task_name = "fake_task"

    def __init__(self, diagnostics):
        self._diagnostics = diagnostics

    def get_eval_diagnostics(self, success):
        return {**self._diagnostics, "success": success}


class EvalDiagnosticsRecorderTest(unittest.TestCase):
    def test_records_jsonl_and_aggregates_generic_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvalDiagnosticsRecorder(directory)
            recorder.record_episode(
                _FakeTask({
                    "primary_failure_reason": "button_not_pressed",
                    "conditions": {"pressed": False, "conditional_value": None},
                    "metrics": {"press_count": 0, "ignored": float("nan")},
                    "events": [],
                }),
                episode_id=0,
                seed=100000,
                success=False,
            )
            recorder.record_episode(
                _FakeTask({
                    "primary_failure_reason": "success",
                    "conditions": {"pressed": True, "conditional_value": True},
                    "metrics": {"press_count": 1},
                    "events": [{"name": "button_pressed"}],
                }),
                episode_id=1,
                seed=100001,
                success=True,
            )

            summary = recorder.write_summary()

            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failure_reasons"]["button_not_pressed"]["count"], 1)
            self.assertEqual(summary["conditions"]["pressed"]["true_rate"], 0.5)
            self.assertEqual(
                summary["conditions"]["conditional_value"]["evaluated_count"],
                1,
            )
            self.assertEqual(summary["metrics"]["press_count"]["mean"], 0.5)
            self.assertNotIn("ignored", summary["metrics"])
            self.assertEqual(summary["event_counts"]["button_pressed"], 1)

            jsonl_path = Path(directory) / "episode_diagnostics.jsonl"
            records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual([record["seed"] for record in records], [100000, 100001])
            self.assertTrue((Path(directory) / "diagnostics_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
