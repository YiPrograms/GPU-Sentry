from __future__ import annotations

import unittest
from types import SimpleNamespace

from gpu_sentry.online.policy import Decision, PolicyInput, RollingMeanMaxPolicy, load_policy
from gpu_sentry.online.processor import OnlineProcessor


class RecordingPolicy:
    def __init__(self) -> None:
        self.observation: PolicyInput | None = None

    def decide(self, observation: PolicyInput) -> Decision:
        self.observation = observation
        return Decision(False, "recorded", {"policy": type(self).__name__})


class RollingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "history_windows": 3,
            "mean_mining_probability": 0.3,
            "max_mining_probability": 0.5,
        }

    def observation(self, score: float) -> PolicyInput:
        return PolicyInput(
            session_id="session",
            window_id="w0000",
            mining_probability=score,
            prediction={"mining_probability_mean": score},
            window_features={"max_bitwise_integer_ratio": 0.8},
            kernel_launches=(
                {
                    "code_id": 42,
                    "kernel_name": "sha256_kernel",
                    "grid_dim": [128, 1, 1],
                    "block_dim": [256, 1, 1],
                },
            ),
            trigger_reason=("token_budget_would_overflow",),
            process_info={"exe_path": "/tmp/workload"},
            signature_cache_hit=False,
        )

    def test_requires_both_thresholds(self) -> None:
        policy = RollingMeanMaxPolicy(self.settings)
        result = policy.decide(self.observation(0.4))
        self.assertFalse(result.suspicious)
        self.assertEqual(result.reason, "below_rolling_thresholds")

    def test_thresholds_are_inclusive(self) -> None:
        policy = RollingMeanMaxPolicy(self.settings)
        policy.decide(self.observation(0.2))
        policy.decide(self.observation(0.2))
        result = policy.decide(self.observation(0.5))
        self.assertTrue(result.suspicious)
        self.assertAlmostEqual(result.details["mean_mining_probability"], 0.3)
        self.assertEqual(result.details["max_mining_probability"], 0.5)

    def test_policy_can_be_loaded_from_an_import_path(self) -> None:
        policy = load_policy(
            "gpu_sentry.online.policy:RollingMeanMaxPolicy",
            self.settings,
        )
        self.assertIsInstance(policy.decide(self.observation(0.1)), Decision)

    def test_processor_provides_kernel_and_process_information(self) -> None:
        policy = RecordingPolicy()
        state = SimpleNamespace(
            session_id="session",
            process_info={"exe_path": "/tmp/workload"},
            policy=policy,
        )
        verdict = {
            "window_id": "w0004",
            "prediction": {"mining_probability_mean": 0.75},
            "l0_features": {"launch_count": 2},
            "trigger_reason": ["token_budget_would_overflow"],
            "signature_cache_hit": True,
            "_policy_kernel_launches": [
                {"code_id": 42, "kernel_name": "sha256_kernel"},
            ],
        }

        OnlineProcessor._apply_online_verdict_policy(None, state, verdict)

        assert policy.observation is not None
        self.assertEqual(policy.observation.kernel_launches[0]["code_id"], 42)
        self.assertEqual(policy.observation.process_info["exe_path"], "/tmp/workload")
        self.assertNotIn("_policy_kernel_launches", verdict)


if __name__ == "__main__":
    unittest.main()
