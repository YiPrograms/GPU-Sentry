from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gpu_sentry.configuration import ConfigError, load_config
from gpu_sentry.online.config import load_online_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    def write_config(self, directory: Path) -> Path:
        path = directory / "config.json"
        path.write_text(json.dumps(self.raw), encoding="utf-8")
        return path

    def test_rejects_invalid_collector_size(self) -> None:
        self.raw["collector"]["launch_batch_size"] = 0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigError):
                load_config(self.write_config(Path(directory)))

    def test_online_paths_are_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_online_config(self.write_config(root), "models/detector")

        self.assertEqual(config.work_dir, root / "artifacts/online")
        self.assertEqual(config.checkpoint_path, (root / "models/detector").resolve())

    def test_custom_policy_defines_its_own_settings(self) -> None:
        self.raw["decision"] = {
            "policy": "custom_policy:ProcessAllowlistPolicy",
            "allowed_executables": ["/usr/bin/python"],
        }
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(self.write_config(Path(directory)))

        self.assertEqual(config.decision["allowed_executables"], ["/usr/bin/python"])


if __name__ == "__main__":
    unittest.main()
