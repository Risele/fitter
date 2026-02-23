import os
import unittest

from app import bootstrap
from app import runtime_config


class RuntimeBootstrapTests(unittest.TestCase):
    def test_requirements_parser_skips_pyinstaller(self):
        names = bootstrap.iter_runtime_requirements()
        self.assertIn("matplotlib", names)
        self.assertIn("scipy", names)
        self.assertNotIn("pyinstaller", names)

    def test_default_codegen_and_logging_config(self):
        cfg = runtime_config.get_config(force_reload=True)
        self.assertIn("codegen", cfg)
        self.assertEqual(int(cfg["codegen"]["max_line_length"]), 80)
        runtime_config.set_timing_env_from_config()
        self.assertEqual(os.environ.get("FITTER_TIMING_LOG"), "0")


if __name__ == "__main__":
    unittest.main()
