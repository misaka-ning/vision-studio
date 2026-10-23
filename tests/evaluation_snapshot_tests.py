# SPDX-License-Identifier: AGPL-3.0-only
"""Snapshot integrity/offline rescanning regressions; no inference is mocked as real."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_tests import DatasetFixture
from evaluation_data import canonical_digest, digest
from evaluation_worker import analyze, atomic_json
from evaluation_snapshot import reanalyze


class SnapshotTests(DatasetFixture):
    def setUp(self):
        super().setUp()
        self.source = self.root / "saved-report"
        self.source.mkdir()
        (self.source / "samples").mkdir()
        self.rows = [{**sample, **row} for sample, row in zip(self.manifest["samples"], self.predictions)]
        self.cache = self.source / "predictions.jsonl"
        self.cache.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.report = analyze(self.request, self.manifest, self.cache)
        self.request = self.report["request"]
        self.report["config"] = {"backend": "test fixture; no inference", "nms_iou": .45}
        atomic_json(self.source / "request.json", self.request)
        atomic_json(self.source / "manifest.json", self.manifest)
        atomic_json(self.source / "report.json", self.report)
        atomic_json(self.source / "status.json", {"status": "complete", "completed": len(self.rows)})
        for i, row in enumerate(self.rows):
            path = self.source / "samples" / f"{i}.json"
            atomic_json(path, row)
            (path.with_suffix(".sha256")).write_text(digest(path) + "\n")
        self.destination = self.root / "new-report"
        self.options = {"confidence": .7, "match_iou": .75, "scan_start": .1,
                        "scan_end": .9, "scan_step": .1, "figure_formats": ["svg"], "figure_dpi": 300}

    @staticmethod
    def lightweight_export(path, report, **kwargs):
        # Plot export itself is tested separately; these tests exercise real AP,
        # matching, snapshot copying and publication/integrity boundaries.
        atomic_json(path, report)

    def run_analysis(self, **options):
        with patch("evaluation_snapshot.export_report", side_effect=self.lightweight_export):
            return reanalyze(self.source, self.destination, {**self.options, **options})

    def originals(self):
        return {str(p.relative_to(self.source)): digest(p) for p in self.source.rglob("*") if p.is_file()}

    def test_reanalysis_works_after_model_dataset_deleted_and_keeps_source(self):
        before = self.originals()
        self.model.unlink()
        self.data.unlink()
        shutil.rmtree(self.images)
        shutil.rmtree(self.labels)
        report = self.run_analysis()
        self.assertEqual(report["completed_images"], 3)
        self.assertEqual(report["request"]["match_iou"], .75)
        self.assertEqual(report["confidence"], .7)
        self.assertEqual(len(report["rows"]), 9)
        self.assertEqual(report["analysis_source"], "verified_snapshot")
        self.assertEqual(report["prediction_sha256"], self.report["prediction_sha256"])
        self.assertEqual(report["model_sha256"], self.report["model_sha256"])
        self.assertEqual(self.originals(), before)
        self.assertTrue((self.destination / "report.json").is_file())
        self.assertEqual(digest(self.destination / "samples/1.json"), digest(self.source / "samples/1.json"))

    def test_relocated_cache_uses_local_copy_not_original_paths(self):
        moved = self.root / "relocated"
        shutil.move(str(self.source), moved)
        self.source = moved
        report = self.run_analysis()
        self.assertEqual(report["source_report"], str(moved))

    def test_modified_predictions_rejected(self):
        with self.cache.open("a") as f:
            f.write("{}\n")
        with self.assertRaisesRegex(ValueError, "预测快照校验失败"):
            self.run_analysis()

    def test_modified_review_sample_rejected(self):
        (self.source / "samples/1.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "逐图快照校验失败"):
            self.run_analysis()

    def test_request_divergence_rejected(self):
        atomic_json(self.source / "request.json", {**self.request, "model_class_id": 0})
        with self.assertRaisesRegex(ValueError, "快照不一致"):
            self.run_analysis()

    def test_cannot_change_inference_class_or_model(self):
        for key, value in (("model_class_id", 0), ("nms_iou", .9), ("model_path", "/not-a-model")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "只允许"):
                self.run_analysis(**{key: value})

    def test_threshold_below_candidate_floor_rejected(self):
        with self.assertRaisesRegex(ValueError, "采集下限"):
            self.run_analysis(scan_start=.0001)

    def test_no_overwrite_and_no_nested_output(self):
        with self.assertRaisesRegex(ValueError, "之外"):
            reanalyze(self.source, self.source / "nested", self.options)
        self.run_analysis()
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            self.run_analysis()

    def test_partial_snapshot_stays_partial_and_does_not_publish_full_ap(self):
        self.cache.write_text(json.dumps(self.rows[0]) + "\n")
        partial_request = {**self.request, "status": "cancelled"}
        report = analyze(partial_request, self.manifest, self.cache)
        atomic_json(self.source / "request.json", report["request"])
        atomic_json(self.source / "report.json", report)
        atomic_json(self.source / "status.json", {"status": "cancelled", "completed": 1})
        result = self.run_analysis()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["completed_images"], 1)
        self.assertIsNone(result["metrics"]["mAP50"])
        self.assertIsNone(result["recommendation"])

    def test_mutation_during_render_prevents_final_report_publication(self):
        def mutate(path, report, **kwargs):
            self.lightweight_export(path, report)
            (self.source / "samples/0.sha256").write_text("changed")
        with patch("evaluation_snapshot.export_report", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "来源快照发生变化"):
                reanalyze(self.source, self.destination, self.options)
        self.assertFalse((self.destination / "report.json").exists())

    def test_preexisting_artifact_or_dangling_link_cannot_be_overwritten(self):
        self.destination.mkdir()
        artifact = self.destination / "threshold-performance.svg"
        artifact.write_text("keep this user file")
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            self.run_analysis()
        self.assertEqual(artifact.read_text(), "keep this user file")
        artifact.unlink()
        target = self.root / "missing-outside-file"
        artifact.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            self.run_analysis()
        self.assertTrue(artifact.is_symlink())
        self.assertFalse(target.exists())

    def test_qt_reserved_directory_matches_exact_options_and_state(self):
        self.destination.mkdir()
        (self.destination / "helper.log").write_bytes(b"Qt worker-owned log\n")
        atomic_json(self.destination / "status.json", {"schema_version": 1, "status": "analyzing", "completed": 0})
        atomic_json(self.destination / "analysis-options.json", self.options)
        result = self.run_analysis()
        self.assertEqual(result["analysis_source"], "verified_snapshot")
        self.assertEqual(json.loads((self.destination / "analysis-options.json").read_text()), self.options)

    def test_mismatched_reserved_directory_is_not_reused(self):
        self.destination.mkdir()
        atomic_json(self.destination / "status.json", {"status": "complete", "completed": 3})
        atomic_json(self.destination / "analysis-options.json", self.options)
        with self.assertRaisesRegex(ValueError, "预留分析任务不一致"):
            self.run_analysis()

    def test_source_symlink_file_or_sample_directory_is_rejected(self):
        original_cache = self.root / "external-predictions.jsonl"
        shutil.move(self.cache, original_cache)
        self.cache.symlink_to(original_cache)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.run_analysis()
        self.cache.unlink()
        shutil.move(original_cache, self.cache)
        original_samples = self.root / "external-samples"
        shutil.move(self.source / "samples", original_samples)
        (self.source / "samples").symlink_to(original_samples, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.run_analysis()

    def test_duplicate_json_fields_and_unknown_analysis_options_are_rejected(self):
        raw = (self.source / "request.json").read_text()
        (self.source / "request.json").write_text(raw.replace("{", '{"confidence":0.5,', 1))
        with self.assertRaisesRegex(ValueError, "重复字段"):
            self.run_analysis()
        atomic_json(self.source / "request.json", self.request)
        with self.assertRaisesRegex(ValueError, "只允许"):
            self.run_analysis(unrecognized_future_option="ignore-me")
        self.assertFalse(self.destination.exists())

    def test_unknown_source_request_metadata_retained_without_new_semantics(self):
        self.request["future_metadata"] = {"note": "retain this JSON, never execute it"}
        self.report["request"] = self.request
        atomic_json(self.source / "request.json", self.request)
        atomic_json(self.source / "report.json", self.report)
        result = self.run_analysis()
        self.assertEqual(result["request"]["future_metadata"], self.request["future_metadata"])

    def test_external_profile_restriction_and_provenance_survive_reanalysis(self):
        self.report.update({"allow_apply_profile": False, "analysis_origin": "external_import",
                            "external_provenance": {"source_sha256": "f" * 64},
                            "source_runtime": {"device": "cpu", "scope": "external recorded runtime"},
                            "source_arguments": {"rect": False}})
        atomic_json(self.source / "report.json", self.report)
        result = self.run_analysis()
        self.assertFalse(result["allow_apply_profile"])
        self.assertFalse(result["request"]["allow_apply_profile"])
        for key in ("analysis_origin", "external_provenance", "source_runtime", "source_arguments"):
            self.assertEqual(result[key], self.report[key])
        self.assertEqual(result["analysis_source"], "verified_snapshot")

    def test_future_schema_and_forged_complete_scope_are_rejected(self):
        for update in ({"schema_version": 2}, {"full_split": False}, {"scope": "subset"}):
            with self.subTest(update=update):
                atomic_json(self.source / "report.json", {**self.report, **update})
                with self.assertRaises(ValueError):
                    self.run_analysis()
        atomic_json(self.source / "report.json", self.report)
        self.assertFalse(self.destination.exists())

    def test_source_subset_with_all_candidates_stays_subset(self):
        request = {**self.request, "full_split": False}
        report = analyze(request, self.manifest, self.cache)
        atomic_json(self.source / "request.json", report["request"])
        atomic_json(self.source / "report.json", report)
        result = self.run_analysis()
        self.assertFalse(result["full_split"])
        self.assertEqual(result["scope"], "subset")
        self.assertIsNone(result["metrics"]["mAP50"])
        self.assertIsNone(result["recommendation"])
        self.assertTrue(all(group["mAP50"] is None for group in result["groups"].values()))

    def test_invalid_gt_rejected_even_with_recomputed_manifest_fingerprint(self):
        self.manifest["samples"][0]["gt"][0]["box"] = [0, 0, 101, 10]
        self.manifest["fingerprint"] = canonical_digest({k: v for k, v in self.manifest.items() if k != "fingerprint"})
        self.report["manifest"] = self.manifest
        self.report["dataset_fingerprint"] = self.manifest["fingerprint"]
        atomic_json(self.source / "manifest.json", self.manifest)
        atomic_json(self.source / "report.json", self.report)
        with self.assertRaisesRegex(ValueError, "标注坐标无效"):
            self.run_analysis()

    def test_completed_and_planned_counts_must_agree(self):
        for update in ({"completed_images": True}, {"completed_images": 2}, {"planned_images": 999},
                       {"sample_images": self.report["sample_images"][::-1]}):
            with self.subTest(update=update):
                atomic_json(self.source / "report.json", {**self.report, **update})
                with self.assertRaises(ValueError):
                    self.run_analysis()

    def test_copy_preserves_exact_source_bytes_and_detects_destination_mutation(self):
        # Noncanonical whitespace/BOM is allowed; copied manifest bytes remain
        # identical, rather than being silently reserialized under a new hash.
        manifest_path = self.source / "manifest.json"
        manifest_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(self.manifest, indent=4).encode())
        original_export = self.lightweight_export
        def mutate(path, report, **kwargs):
            original_export(path, report)
            (self.destination / "samples/1.sha256").write_text("changed")
        with patch("evaluation_snapshot.export_report", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "新报告逐图快照"):
                reanalyze(self.source, self.destination, self.options)
        self.assertFalse((self.destination / "report.json").exists())
        self.assertFalse((self.destination / ".pending-report.json").exists())
        self.assertEqual(manifest_path.read_bytes(), (self.destination / "manifest.json").read_bytes())

    def test_destination_predictions_changed_after_export_prevent_publication(self):
        def mutate(path, report, **kwargs):
            self.lightweight_export(path, report)
            (self.destination / "predictions.jsonl").write_text("{}\n")
        with patch("evaluation_snapshot.export_report", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "新报告预测、请求或数据集快照"):
                reanalyze(self.source, self.destination, self.options)
        self.assertFalse((self.destination / "report.json").exists())
        self.assertFalse((self.destination / ".pending-report.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
