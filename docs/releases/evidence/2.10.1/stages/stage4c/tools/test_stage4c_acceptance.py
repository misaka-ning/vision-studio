"""Pure guard tests; fabricated rows are never accepted as execution evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage4c_acceptance", HERE / "stage4c_acceptance.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def row(success=True, skipped=False):
    return {"success": success, "exit_code": 0 if success else 8, "skipped": skipped}


class SelectionGuards(unittest.TestCase):
    def test_corrected_result_is_preferred_and_unchanged_initial_can_be_reused(self):
        result, changed = tool.choose_effective({"a": row(False), "b": row()}, {"a": row()},
                                                {"a": "old", "b": "same"}, {"a": "new", "b": "same"})
        self.assertEqual(result["a"][0], "corrected")
        self.assertEqual(result["b"][0], "initial")
        self.assertEqual(changed, {"a"})

    def test_failed_correction_cannot_fall_back_to_initial_success(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "Effective suite failed"):
            tool.choose_effective({"a": row()}, {"a": row(False)}, {"a": "same"}, {"a": "same"})

    def test_skipped_correction_is_not_success(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "failed/skipped"):
            tool.choose_effective({"a": row()}, {"a": row(True, True)}, {"a": "same"}, {"a": "same"})

    def test_changed_native_must_be_rerun(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "Changed native binary was not rerun"):
            tool.choose_effective({"a": row()}, {}, {"a": "old"}, {"a": "new"})

    def test_missing_binary_inventory_is_rejected(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "gained or lost"):
            tool.choose_effective({"a": row()}, {}, {"a": "old"}, {})

    def test_new_unregistered_corrected_suite_is_rejected(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "absent from the initial"):
            tool.choose_effective({"a": row()}, {"b": row()}, {}, {})

    def test_duplicate_runs_are_rejected(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "Duplicate"):
            tool.indexed({"runs": [{"name": "a"}, {"name": "a"}], "selected_tests": ["a", "a"]})

    def test_zero_exit_unrerun_skip_cannot_be_reused(self):
        with self.assertRaisesRegex(tool.AcceptanceError, "failed/skipped"):
            tool.choose_effective({"a": row(True, True)}, {}, {"a": "same"}, {"a": "same"})

    def test_original_parser_rejects_hidden_python_skip(self):
        helper = tool.load_helper(HERE.parent / "qa-tools/summarize_regressions.py")
        with self.assertRaisesRegex(RuntimeError, "plain OK"):
            helper.parse_actual_totals("Ran 3 tests in 0.100s\n\nOK (skipped=1)\n", "guard-fixture")


if __name__ == "__main__":
    unittest.main()
