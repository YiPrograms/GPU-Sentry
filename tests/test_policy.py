from __future__ import annotations

import unittest

from gpu_sentry.online.policy import Decision, RollingMeanMaxPolicy, load_policy


class RollingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "history_windows": 3,
            "mean_mining_probability": 0.3,
            "max_mining_probability": 0.5,
        }

    def test_requires_both_thresholds(self) -> None:
        policy = RollingMeanMaxPolicy(self.settings)
        result = policy.update(0.4)
        self.assertFalse(result.suspicious)
        self.assertEqual(result.reason, "below_rolling_thresholds")

    def test_thresholds_are_inclusive(self) -> None:
        policy = RollingMeanMaxPolicy(self.settings)
        policy.update(0.2)
        policy.update(0.2)
        result = policy.update(0.5)
        self.assertTrue(result.suspicious)
        self.assertAlmostEqual(result.details["mean_mining_probability"], 0.3)
        self.assertEqual(result.details["max_mining_probability"], 0.5)

    def test_policy_can_be_loaded_from_an_import_path(self) -> None:
        policy = load_policy(
            "gpu_sentry.online.policy:RollingMeanMaxPolicy",
            self.settings,
        )
        self.assertIsInstance(policy.update(0.1), Decision)


if __name__ == "__main__":
    unittest.main()
