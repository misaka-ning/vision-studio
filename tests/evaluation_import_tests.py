#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""External report interoperability/integrity fixtures, never accuracy claims."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from filesystem_support import symlink_or_skip
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluation_import as importer
from evaluation_snapshot import reanalyze, strict_json, validate_samples


def digest_bytes(value):
    return hashlib.sha256(value.encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()


class ExternalImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-external-import-")
        self.root = Path(self.temp.name)
        self.source, self.destination = self.root / "source", self.root / "imported"
        self.source.mkdir()
        self.environment = patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(self.root / "yolo"),
                                                  "MPLCONFIGDIR": str(self.root / "matplotlib")})
        self.environment.start()
        missing = self.root / "deliberately-absent-original-data"
        self.rows = [
            {"image": str(missing / "images/val/a.jpg"), "width": 100, "height": 80,
             "gt": [[10., 10., 30., 30.]], "boxes": [[10., 10., 30., 30.], [50., 50., 60., 60.]],
             "scores": [.9, .3], "flags": {"Small": 1, "Occluded": 0, "Blurred": 0, "No Ball": 0}},
            {"image": str(missing / "images/val/b.jpg"), "width": 100, "height": 80,
             "gt": [], "boxes": [[10., 10., 30., 30.]], "scores": [.4],
             "flags": {"Small": 0, "Occluded": 0, "Blurred": 0, "No Ball": 1}},
        ]
        self.args = {"model": "weights/model.pt", "data": str(missing / "data.yaml"),
                     "dataset_config": None, "dataset_version": None, "metadata": None,
                     "split": "val", "class_id": 32, "imgsz": 640, "batch": 4, "device": "0",
                     "threads": 1, "min_conf": .001, "conf": .5, "match_iou": .5, "nms_iou": .7,
                     "max_det": 300, "limit": None, "output": "eval/synthetic", "formats": "svg", "dpi": 300,
                     "dry_run": False}
        self.manifest = {
            "model": str(missing / "model.pt"), "model_sha256": digest_bytes("synthetic absent weight"),
            "data": str(missing / "data.yaml"), "metadata": str(missing / "metadata.csv"),
            "dataset_yaml_sha256": digest_bytes("synthetic absent yaml"),
            "metadata_sha256": digest_bytes("synthetic absent audit"), "selected": [], "skipped": [],
            "arguments": copy.deepcopy(self.args), "model_names": {str(i): "sports ball" if i == 32 else str(i) for i in range(80)},
            "selected_model_class_id": 32,
            "evaluation_code_sha256": {"evaluate.py": digest_bytes("synthetic evaluator version")},
            "metrics_method": "Ultralytics AP / highest-IoU greedy matching; fixed-threshold operating metrics"}
        for row in self.rows:
            self.manifest["selected"].append({"image": row["image"],
                "label": row["image"].replace("/images/", "/labels/").replace(".jpg", ".txt"),
                "width": row["width"], "height": row["height"], "gt_boxes": len(row["gt"]),
                "scenes": copy.deepcopy(row["flags"]), "image_sha256": digest_bytes(row["image"]),
                "label_sha256": digest_bytes(row["image"] + ":label")})
        self.summary = {"status": "complete", "model": self.manifest["model"], "model_name": "model",
            "model_class_id": 32, "split": "val", "partial_split": False, "available_before_limit": 2,
            "skipped_images": 0, "metrics": {"images": 2, "gt_boxes": 1, "predicted_boxes": 1,
                "TP": 1, "FP": 0, "FN": 0, "confidence": .5, "match_iou": .5, "no_ball_images": 1},
            "scene_metrics": {}, "runtime": {"predict_wall_seconds": 1.25,
                "predict_images_per_second_including_warmup": 1.6,
                "backend_reported_mean_ms": {"inference": 12.3}, "device": "cuda:0", "torch_threads": 1,
                "image_loader": "LoadImagesAndVideos", "loader_batch_size": 4,
                "actual_input_shape": [640, 640], "backend_fp16": False, "backend_end2end": False},
            "input_files_verified_unchanged": True,
            "versions": {"ultralytics": "synthetic-fixture", "torch": "synthetic-fixture", "numpy": "synthetic-fixture"},
            "created_at_utc": "2026-10-23T00:00:00+00:00", "arguments": copy.deepcopy(self.args)}
        self.state = {"status": "complete", "input_files_verified_unchanged": True}
        self.options = {"confidence": .5, "match_iou": .5, "scan_start": .1, "scan_end": .9,
                        "scan_step": .2, "figure_formats": ["svg"], "figure_dpi": 72}
        self.save()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def save(self):
        for name, value in (("status.json", self.state), ("metrics.json", self.summary),
                            ("inputs_manifest.json", self.manifest)):
            (self.source / name).write_bytes(encoded(value))
        (self.source / "predictions.jsonl").write_bytes(b"".join(encoded(row) for row in self.rows))

    def contents(self):
        return {name: (self.source / name).read_bytes() for name in importer.SOURCE_FILES}

    def test_full_import_without_any_original_file_has_original_id_and_verified_review_cache(self):
        before = self.contents()
        report = importer.import_report(self.source, self.destination, self.options)
        self.assertEqual(before, self.contents())
        self.assertTrue(report["full_split"])
        self.assertFalse(report["allow_apply_profile"])
        self.assertFalse(report["request"]["allow_apply_profile"])
        self.assertEqual(report["model_class_id"], 32)
        self.assertEqual(report["dataset_class_id"], 0)
        self.assertEqual(report["metrics"]["TP"], 1)
        self.assertEqual(report["metrics"]["FP"], 0)
        self.assertEqual(report["scene_review"]["reviewed_images"], 2)
        self.assertTrue(report["groups"]["Small"]["available"])
        self.assertEqual(report["source_runtime"], self.summary["runtime"])
        self.assertEqual(report["source_arguments"], self.args)
        self.assertNotIn("color_mode", report["config"])
        self.assertNotIn("scale", report["config"])
        self.assertNotIn("effective_batch_size", report["request"])
        self.assertEqual(report["config"]["nms_iou"], .7)
        self.assertEqual(report["config"]["max_detections"], 300)
        validate_samples(report["manifest"])
        for index, original in enumerate(self.manifest["selected"]):
            data = (self.destination / "samples" / f"{index}.json").read_bytes()
            cache = json.loads(data)
            expected = hashlib.sha256(data).hexdigest() + "\n"
            self.assertEqual((self.destination / "samples" / f"{index}.sha256").read_text(), expected)
            self.assertEqual(cache["image_sha256"], original["image_sha256"])
            self.assertEqual([p["class_id"] for p in cache["predictions"]], [32] * len(cache["predictions"]))
            self.assertEqual([gt["class_id"] for gt in cache["gt"]], [0] * len(cache["gt"]))
        for name, data in before.items():
            self.assertEqual((self.destination / "external-source" / name).read_bytes(), data)
        self.assertTrue((self.destination / "pr-curve.svg").is_file())
        self.assertTrue((self.destination / "report.json").is_file())

    def test_limit_subset_stays_subset_and_does_not_publish_full_ap(self):
        self.args["limit"] = 2
        self.manifest["arguments"] = copy.deepcopy(self.args)
        self.summary.update(arguments=copy.deepcopy(self.args), available_before_limit=10, partial_split=True)
        self.save()
        report = importer.import_report(self.source, self.destination, self.options)
        self.assertFalse(report["full_split"])
        self.assertEqual(report["scope"], "subset")
        self.assertIsNone(report["metrics"]["mAP50"])
        self.assertIsNone(report["recommendation"])
        self.assertEqual(report["external_provenance"]["available_before_limit"], 10)

    def test_import_then_snapshot_reanalysis_keeps_apply_block_and_provenance(self):
        imported = importer.import_report(self.source, self.destination, self.options)
        source_before = {path.relative_to(self.destination): path.read_bytes()
                         for path in self.destination.rglob("*") if path.is_file()}
        report = reanalyze(self.destination, self.root / "reanalysis", dict(self.options, confidence=.7))
        self.assertFalse(report["allow_apply_profile"])
        self.assertFalse(report["request"]["allow_apply_profile"])
        self.assertEqual(report["analysis_origin"], "external_model_evaluation")
        for key in ("external_provenance", "source_runtime", "source_arguments"):
            self.assertEqual(report[key], imported[key])
        self.assertEqual(source_before, {path.relative_to(self.destination): path.read_bytes()
                                        for path in self.destination.rglob("*") if path.is_file()})

    def test_source_runtime_and_id_malformed_cases_are_rejected_before_output(self):
        variants = [
            ("status", lambda: self.state.update(status="failed")),
            ("unverified", lambda: self.state.update(input_files_verified_unchanged=False)),
            ("args", lambda: self.summary["arguments"].update(nms_iou=.4)),
            ("id_mismatch", lambda: self.summary.update(model_class_id=0)),
            ("invalid_id", lambda: self.manifest.update(selected_model_class_id=True)),
            ("sparse_names", lambda: self.manifest.update(model_names={"32": "sports ball"})),
            ("unknown_model_hash", lambda: self.manifest.update(model_sha256="missing")),
            ("missing_runtime", lambda: self.summary.pop("runtime")),
            ("invalid_threads", lambda: self.summary["runtime"].update(torch_threads=0)),
            ("unknown_fp16", lambda: self.summary["runtime"].update(backend_fp16=None)),
            ("false_full", lambda: self.summary.update(available_before_limit=20)),
            ("bad_gt_count", lambda: self.manifest["selected"][0].update(gt_boxes=2)),
            ("image_hash", lambda: self.manifest["selected"][0].update(image_sha256="0")),
            ("scene_bool", lambda: self.manifest["selected"][0]["scenes"].update(Small=True)),
            ("wrong_scene", lambda: self.rows[0]["flags"].update(Small=0)),
            ("no_ball_disagrees", lambda: (self.rows[0]["flags"].update({"No Ball": 1}), self.manifest["selected"][0]["scenes"].update({"No Ball": 1}))),
            ("order", lambda: self.rows.reverse()),
            ("dimensions", lambda: self.rows[0].update(width=True)),
            ("box_bounds", lambda: self.rows[0]["boxes"][0].__setitem__(2, 101)),
            ("negative_score", lambda: self.rows[0]["scores"].__setitem__(0, -.1)),
            ("below_minconf", lambda: self.rows[0]["scores"].__setitem__(0, .0001)),
            ("mismatched_scores", lambda: self.rows[0]["scores"].pop()),
            ("extra_classes", lambda: self.rows[0].update(class_ids=[0, 0])),
            ("duplicate_image", lambda: self.manifest["selected"][1].update(image=self.manifest["selected"][0]["image"])),
            ("tp_fp_count", lambda: self.summary["metrics"].update(TP=0)),
            ("missing_row", lambda: self.rows.pop()),
        ]
        pristine = copy.deepcopy((self.state, self.summary, self.manifest, self.rows))
        for name, mutation in variants:
            with self.subTest(name=name):
                self.state, self.summary, self.manifest, self.rows = copy.deepcopy(pristine)
                mutation(); self.save()
                with self.assertRaises(ValueError):
                    importer.import_report(self.source, self.destination, self.options)
                self.assertFalse(self.destination.exists())

    def test_low_candidates_require_explicit_higher_analysis_and_scan(self):
        self.args["min_conf"] = .2
        self.manifest["arguments"] = copy.deepcopy(self.args)
        self.summary["arguments"] = copy.deepcopy(self.args)
        self.save()
        with self.assertRaisesRegex(ValueError, "采集下限"):
            importer.import_report(self.source, self.destination, dict(self.options, scan_start=.05))
        self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ValueError, "采集下限"):
            importer.import_report(self.source, self.destination, {})

    def test_duplicate_json_keys_nonfinite_and_links_are_rejected(self):
        original = self.contents()
        for name, replacement in (("status.json", b'{"status":"complete","status":"complete"}\n'),
                                  ("predictions.jsonl", b'{"image":"test","scores":[NaN]}\n')):
            with self.subTest(name=name):
                (self.source / name).write_bytes(replacement)
                with self.assertRaises(ValueError):
                    importer.import_report(self.source, self.destination, self.options)
                for filename, data in original.items():
                    (self.source / filename).write_bytes(data)
        link = self.source / "metrics.json"
        outside = self.root / "outside.json"; outside.write_bytes(link.read_bytes())
        link.unlink(); symlink_or_skip(self, link, outside)
        with self.assertRaisesRegex(ValueError, "普通文件"):
            importer.import_report(self.source, self.destination, self.options)

    def test_unknown_options_existing_output_and_nested_source_do_not_overwrite(self):
        with self.assertRaises(ValueError):
            importer.import_report(self.source, self.destination, {"nms_iou": .3})
        self.destination.mkdir(); marker = self.destination / "keep.txt"; marker.write_text("preserve")
        with self.assertRaises(ValueError):
            importer.import_report(self.source, self.destination, self.options)
        self.assertEqual(marker.read_text(), "preserve")
        for output in (self.source, self.source / "nested", self.root):
            with self.assertRaises(ValueError):
                importer.import_report(self.source, output, self.options)

    def test_source_mutation_during_analysis_never_publishes_report(self):
        analyze = importer.analyze
        def changing(*args, **kwargs):
            result = analyze(*args, **kwargs)
            (self.source / "status.json").write_bytes(b'{"status":"changed-during-import"}\n')
            return result
        with patch.object(importer, "analyze", side_effect=changing):
            with self.assertRaisesRegex(RuntimeError, "来源快照发生变化"):
                importer.import_report(self.source, self.destination, self.options)
        self.assertFalse((self.destination / "report.json").exists())

    def test_reserved_qt_output_and_cli_have_same_contract(self):
        self.destination.mkdir()
        (self.destination / "status.json").write_bytes(encoded({"status": "analyzing", "completed": 0}))
        (self.destination / "helper.log").write_text("Qt subprocess log, never input data.\n")
        options = self.destination / "analysis-options.json"; options.write_bytes(encoded(self.options))
        process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluation_import.py"),
                                  "--source", str(self.source), "--options", str(options),
                                  "--output", str(self.destination)], capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertTrue(json.loads(process.stdout)["success"])
        report = strict_json(self.destination / "report.json")
        self.assertFalse(report["allow_apply_profile"])
        self.assertEqual(report["completed_images"], 2)

    def test_qt_helper_log_link_is_rejected(self):
        self.destination.mkdir()
        (self.destination / "status.json").write_bytes(encoded({"status": "analyzing", "completed": 0}))
        (self.destination / "analysis-options.json").write_bytes(encoded(self.options))
        outside = self.root / "preserve.log"; outside.write_text("preserve")
        symlink_or_skip(self, self.destination / "helper.log", outside)
        with self.assertRaisesRegex(ValueError, "helper.log"):
            importer.import_report(self.source, self.destination, self.options)
        self.assertEqual(outside.read_text(), "preserve")

    def test_destination_copy_mutation_during_export_never_publishes(self):
        export = importer.export_report
        # Each attack happens after the normal report/export integrity checks,
        # exercising the final commit guard against locally modified copies.
        for index, name in enumerate(("request.json", "manifest.json", "predictions.jsonl",
                                      "external-source/status.json", "external-source/metrics.json")):
            with self.subTest(name=name):
                destination = self.root / ("mutated-target-" + str(index))
                def changing(*args, **kwargs):
                    result = export(*args, **kwargs)
                    (destination / name).write_bytes(b"changed-after-export\n")
                    return result
                with patch.object(importer, "export_report", side_effect=changing):
                    with self.assertRaisesRegex(RuntimeError, "目标快照在保存期间发生变化"):
                        importer.import_report(self.source, destination, self.options)
                self.assertFalse((destination / "report.json").exists())

    def test_all_zero_predictions_are_kept_and_model_names_list_is_supported(self):
        self.manifest["model_names"] = list(self.manifest["model_names"].values())
        for row in self.rows:
            row["boxes"], row["scores"] = [], []
        self.summary["metrics"].update(TP=0, FP=0, FN=1, predicted_boxes=0)
        self.save()
        manifest, request, rows, _, _ = importer.validate_source(self.contents())
        self.assertEqual([row["predictions"] for row in rows], [[], []])
        self.assertEqual(request["model_class_id"], 32)
        self.assertEqual(len(manifest["samples"]), 2)

    def test_actual_runtime_values_are_not_replaced_by_requested_values(self):
        self.summary["runtime"].update(torch_threads=4, loader_batch_size=1, actual_input_shape=[672, 672])
        self.save()
        _, request, _, summary, _ = importer.validate_source(self.contents())
        self.assertEqual(request["config"]["cpu_threads"], 1)
        self.assertEqual(request["config"]["resolved_cpu_threads"], 4)
        self.assertEqual(request["config"]["input_size"], 672)
        self.assertEqual(summary["runtime"]["loader_batch_size"], 1)
        self.assertEqual(request["requested_batch_size"], 4)
        self.assertNotIn("effective_batch_size", request)


if __name__ == "__main__":
    unittest.main(verbosity=2)
