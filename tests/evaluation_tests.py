# SPDX-License-Identifier: AGPL-3.0-only
"""Hand-calculated metric, dataset-integrity and actual CLI export regressions."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
import yaml

from evaluation_data import (canonical_digest, digest, prepare_dataset, validate_manifest)
import evaluation_data
from evaluation_metrics import (average_precision, box_iou, matching_pairs,
                                operating, prepare_records, recommend, threshold_grid)
from evaluation_worker import analyze, export_report


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-evaluation-tests-")
        self.root = Path(self.temp.name)
        self.environment_patch = patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(self.root / "ultralytics"),
                                                        "MPLCONFIGDIR": str(self.root / "matplotlib")})
        self.environment_patch.start()
        self.images = self.root / "images" / "val"
        self.labels = self.root / "labels" / "val"
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)
        self.data = self.root / "dataset.yaml"
        self.write_yaml()
        self.a = self.add_image("a", "0 0.5 0.5 0.5 0.5\n")
        self.b = self.add_image("b", "0 0.5 0.5 0.5 0.5\n")
        self.c = self.add_image("c", "")
        self.model = self.root / "test-model.onnx"
        self.model.write_bytes(b"only a hashing fixture, never claimed to be an inference model")
        self.manifest = self.prepare()
        self.prediction_file = self.root / "predictions.jsonl"
        self.predictions = [
            {"image": str(self.a), "width": 100, "height": 100, "predictions": [
                {"class_id": 0, "confidence": .95, "box": [25., 25., 75., 75.]},
                {"class_id": 32, "confidence": .9, "box": [25., 25., 75., 75.]},
                {"class_id": 32, "confidence": .6, "box": [25., 25., 75., 75.]}]},
            {"image": str(self.b), "width": 100, "height": 100, "predictions": [
                {"class_id": 32, "confidence": .8, "box": [0., 0., 10., 10.]},
                {"class_id": 32, "confidence": .3, "box": [25., 25., 75., 75.]}]},
            {"image": str(self.c), "width": 100, "height": 100, "predictions": [
                {"class_id": 32, "confidence": .4, "box": [25., 25., 75., 75.]}]},
        ]
        self.write_predictions()
        self.request = {"model_path": str(self.model), "model_sha256": digest(self.model),
                        "model_class_id": 32, "model_class_count": 80, "dataset_class_id": 0,
                        "confidence": .5, "match_iou": .5, "score_floor": .001,
                        "sample_images": [str(self.a), str(self.b), str(self.c)],
                        "status": "complete", "full_split": True}

    def tearDown(self):
        self.environment_patch.stop()
        self.temp.cleanup()

    def write_yaml(self, **updates):
        config = {"path": ".", "val": "images/val", "names": ["football", "robot"]}
        config.update(updates)
        self.data.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    def add_image(self, name, label, directory=None):
        path = (directory or self.images) / (name + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "#203040").save(path)
        if label is not None:
            target = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(label, encoding="utf-8")
        return path.resolve()

    def prepare(self, **updates):
        return prepare_dataset({"dataset_yaml": str(self.data), "split": "val", **updates})

    def write_predictions(self, rows=None):
        rows = self.predictions if rows is None else rows
        self.prediction_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def analyze(self, **updates):
        return analyze({**self.request, **updates}, self.manifest, self.prediction_file)


class DatasetTests(DatasetFixture):
    def test_standard_dataset_pixel_boxes_and_negative_counts(self):
        self.assertEqual(self.manifest["names"], ["football", "robot"])
        self.assertEqual(self.manifest["samples"][0]["gt"], [{"class_id": 0, "box": [25., 25., 75., 75.]}])
        self.assertEqual(self.manifest["validation"], {"positive_images": 2, "negative_images": 1,
                                                       "missing_labels": 0, "excluded": 0})
        self.assertFalse(self.manifest["requires_attention"])
        validate_manifest(self.manifest)

    def test_more_than_fifty_images_no_limit(self):
        for index in range(58):
            self.add_image(f"extra-{index:03}", "")
        manifest = self.prepare()
        self.assertEqual(len(manifest["samples"]), 61)
        self.assertEqual(manifest["available_images"], 61)

    def test_yaml_names_mapping_and_root_relative_to_yaml(self):
        self.write_yaml(names={0: "football", 1: "robot"}, nc=2)
        manifest = self.prepare()
        self.assertEqual(manifest["root"], str(self.root))
        self.assertEqual(manifest["names"], ["football", "robot"])

    def test_manifest_survives_qt_equivalent_integral_float_serialization(self):
        def qt_number(value):
            number = float(value)
            return int(number) if number.is_integer() else number
        qt_roundtrip = json.loads(json.dumps(self.manifest), parse_float=qt_number)
        self.assertIsInstance(self.manifest["samples"][0]["gt"][0]["box"][0], float)
        self.assertIsInstance(qt_roundtrip["samples"][0]["gt"][0]["box"][0], int)
        validate_manifest(qt_roundtrip)
        self.assertEqual(canonical_digest([22., 0., -0., .125, True]),
                         canonical_digest([22, 0, 0, .125, True]))
        self.assertNotEqual(canonical_digest([.1]), canonical_digest([.10000000000000002]))
        self.assertNotEqual(canonical_digest([True]), canonical_digest([1]))

    def test_txt_preserves_explicit_order_and_is_hashed(self):
        listing = self.root / "selected.txt"
        listing.write_text("images/val/c.png\nimages/val/a.png\n", encoding="utf-8")
        self.write_yaml(val="selected.txt")
        manifest = self.prepare()
        self.assertEqual([r["image"] for r in manifest["samples"]], [str(self.c), str(self.a)])
        self.assertEqual(manifest["source_hashes"][str(listing)], digest(listing))

    def test_list_of_splits_and_duplicate_rejected(self):
        other = self.add_image("other", "", self.root / "images" / "other")
        self.write_yaml(val=["images/val", "images/other"])
        self.assertEqual(len(self.prepare()["samples"]), 4)
        self.write_yaml(val=["images/val", str(other), "images/val"])
        with self.assertRaisesRegex(ValueError, "重复"):
            self.prepare()

    def test_missing_label_is_attention_not_negative(self):
        self.add_image("missing", None)
        manifest = self.prepare()
        self.assertTrue(manifest["requires_attention"])
        self.assertEqual(manifest["validation"]["missing_labels"], 1)
        self.assertEqual(manifest["validation"]["negative_images"], 1)
        self.assertEqual(len(manifest["samples"]), 3)
        with self.assertRaisesRegex(ValueError, "缺失标签"):
            analyze(self.request, manifest, self.prediction_file)

    def test_invalid_label_formats(self):
        for content in ("-1 .5 .5 .5 .5", "0.0 .5 .5 .5 .5", "2 .5 .5 .5 .5",
                        "0 nan .5 .5 .5", "0 .5 .5 0 .5", "0 .1 .1 .8 .8",
                        "0 .5 .5 .5 .5 .1 .2"):
            with self.subTest(content=content):
                (self.labels / "a.txt").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_names_must_be_contiguous_and_nc_consistent(self):
        for names, nc in (({1: "football"}, 1), ({0: "football", "0": "dup"}, 2), (["football"], 2), ([], 0)):
            with self.subTest(names=names):
                self.write_yaml(names=names, nc=nc)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_nonstandard_exif_rejected_before_coordinate_rotation(self):
        image = Image.new("RGB", (100, 100), "red")
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.images / "rotated.jpg", exif=exif)
        (self.labels / "rotated.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EXIF"):
            self.prepare()

    def test_optional_review_metadata_strict_and_exclusions_recorded(self):
        self.write_yaml(names=["football"])
        metadata = self.root / "metadata.csv"
        fields = ["图片", "split", "annotation_status", "label_review_required", "scene_review_required",
                  "Small", "Occluded", "Blurred", "No Ball"]
        rows = []
        for filename, status in (("a.png", "reviewed_annotation"), ("b.png", "reviewed_excluded"),
                                 ("c.png", "negative_confirmed")):
            row = {k: "0" for k in fields}
            row.update({"图片": filename, "split": "val", "annotation_status": status,
                        "No Ball": "1" if filename == "c.png" else "0"})
            rows.append(row)
        with metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = self.prepare(metadata_path=str(metadata))
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual(manifest["skipped"][0]["reason"], "reviewed_excluded")
        self.assertFalse(manifest["requires_attention"])
        rows[2]["No Ball"] = "0"
        with metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "No Ball"):
            self.prepare(metadata_path=str(metadata))


class MetricsTests(DatasetFixture):
    def test_hand_calculated_operating_point_and_raw_indices(self):
        report = self.analyze()
        metrics = report["metrics"]
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (1, 2, 1))
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["F1"], .4)
        self.assertEqual(metrics["no_ball_false_positive_rate"], 0.)
        self.assertEqual(report["per_image"][0]["matched_indices"], [1])
        self.assertEqual(report["per_image"][0]["fp_indices"], [2])
        self.assertEqual(report["request"]["model_class_id"], 32)
        self.assertEqual(report["request"]["dataset_class_id"], 0)

    def test_low_threshold_rematches_and_negative_image_rate(self):
        metrics = self.analyze(confidence=.25)["metrics"]
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (2, 3, 0))
        self.assertEqual(metrics["precision"], .4)
        self.assertEqual(metrics["recall"], 1.)
        self.assertEqual(metrics["no_ball_false_positive_rate"], 1.)
        self.assertEqual(metrics["no_ball_false_positive_images"], 1)
        self.assertEqual(metrics["no_ball_false_positive_boxes"], 1)

    def test_zero_predictions_with_gt_is_zero_precision_recall_f1(self):
        metrics = self.analyze(confidence=.99)["metrics"]
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["F1"]), (0., 0., 0.))
        self.assertEqual(metrics["FN"], 2)

    def test_only_target_gt_defines_negative_image(self):
        report = self.analyze(dataset_class_id=1)
        metrics = report["metrics"]
        self.assertEqual(metrics["no_ball_images"], 3)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["mAP50"])
        self.assertEqual(metrics["FP"], 3)
        self.assertAlmostEqual(metrics["no_ball_false_positive_rate"], 2 / 3)

    def test_empty_cancelled_result_unknown_model_count(self):
        self.write_predictions([])
        report = self.analyze(status="cancelled", model_class_count=0)
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["completed_images"], 0)
        self.assertEqual(report["metrics"]["TP"], 0)
        self.assertIsNone(report["metrics"]["precision"])
        self.assertIsNone(report["metrics"]["recall"])
        self.assertIsNone(report["metrics"]["no_ball_false_positive_rate"])
        self.assertIsNone(report["metrics"]["mAP50"])
        self.assertFalse(report["full_split"])

    def test_subset_single_and_cancelled_never_present_full_map(self):
        self.write_predictions(self.predictions[:1])
        report = self.analyze(sample_images=[str(self.a)], full_split=False)
        self.assertEqual(report["scope"], "single_image")
        self.assertIsNone(report["metrics"]["mAP50"])
        self.assertIsNotNone(report["partial_metrics"]["mAP50"])
        self.assertIsNone(report["recommendation"])
        report = self.analyze(status="cancelled")
        self.assertFalse(report["full_split"])
        self.assertIsNone(report["metrics"]["mAP50"])

    def test_ap_independent_of_working_threshold(self):
        a = self.analyze(confidence=.25)
        b = self.analyze(confidence=.95)
        self.assertEqual(a["metrics"]["mAP50"], b["metrics"]["mAP50"])
        self.assertGreater(a["metrics"]["mAP50"], 0)
        self.assertLessEqual(a["metrics"]["mAP50_95"], 1)
        self.assertIsNotNone(a["recommendation"])

    def test_perfect_ap_uses_actual_ultralytics_integral(self):
        samples = self.manifest["samples"][:2]
        predictions = [{"predictions": [{"class_id": 32, "confidence": .9, "box": [25, 25, 75, 75]}]} for _ in samples]
        records = prepare_records(samples, predictions, 32, 0)
        ap, _ = average_precision(records)
        self.assertAlmostEqual(ap["mAP50"], .995)
        self.assertAlmostEqual(ap["mAP50_95"], .995)

    def test_threshold_grid_inclusive_and_bounded(self):
        grid = threshold_grid()
        self.assertEqual(len(grid), 91)
        self.assertEqual((grid[0], grid[-1]), (.05, .95))
        self.assertEqual(threshold_grid(.1, .35, .1), [.1, .2, .3, .35])
        for arguments in ((.95, .05, .01), (.05, .95, 0), (0, 1, .00001), (float("nan"), .95, .01)):
            with self.assertRaises(ValueError):
                threshold_grid(*arguments)

    def test_highest_iou_one_to_one_and_stable_ties(self):
        self.assertEqual(matching_pairs(np.array([[.9, .8], [.85, .1]]), .5).tolist(), [[0, 0]])
        self.assertEqual(matching_pairs(np.ones((2, 2)), .5).tolist(), [[0, 0]])
        self.assertEqual(matching_pairs(np.array([[.7, .9], [.95, .1]]), .5).tolist(), [[0, 1], [1, 0]])

    def test_recommendation_constraints_and_no_feasible_point(self):
        rows = self.analyze()["rows"]
        best = recommend(rows)
        self.assertIsNotNone(best)
        self.assertIsNone(recommend(rows, minimum_precision=1., minimum_recall=1.))
        self.assertEqual(recommend(rows, maximum_fpr=0.)["no_ball_false_positive_rate"], 0.)


class CacheIntegrityTests(DatasetFixture):
    def test_candidate_floor_cannot_recover_discarded_predictions(self):
        with self.assertRaisesRegex(ValueError, "采集下限"):
            self.analyze(score_floor=.1)

    def test_bad_prediction_caches_rejected(self):
        for change in ("order", "missing", "duplicate", "nan", "score", "box", "dimensions", "class"):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.predictions)
                if change == "order":
                    rows.reverse()
                elif change == "missing":
                    rows.pop()
                elif change == "duplicate":
                    rows[1] = rows[0]
                elif change == "nan":
                    rows[0]["predictions"][0]["confidence"] = float("nan")
                elif change == "score":
                    rows[0]["predictions"][0]["confidence"] = 1.1
                elif change == "box":
                    rows[0]["predictions"][0]["box"] = [0, 0, 101, 100]
                elif change == "dimensions":
                    rows[0]["width"] = 50
                elif change == "class":
                    rows[0]["predictions"][0]["class_id"] = -1
                self.write_predictions(rows)
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_mapping_ids_validate_actual_counts(self):
        for changes in ({"model_class_id": 80}, {"model_class_id": -1}, {"model_class_id": 1.5},
                        {"dataset_class_id": 2}, {"dataset_class_id": True}, {"model_class_count": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.analyze(**changes)

    def test_manifest_modification_rejected(self):
        self.manifest["samples"][0]["gt"] = []
        with self.assertRaisesRegex(ValueError, "指纹"):
            self.analyze()

    def test_label_changes_rejected(self):
        (self.labels / "a.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "原图或标注"):
            self.analyze()

    def test_label_changed_during_preparation_is_rejected(self):
        original = evaluation_data.read_boxes
        changed = False
        def mutate_after_read(path, width, height, count):
            nonlocal changed
            result = original(path, width, height, count)
            if not changed:
                changed = True
                Path(path).write_text("", encoding="utf-8")
            return result
        with patch("evaluation_data.read_boxes", side_effect=mutate_after_read):
            with self.assertRaisesRegex(RuntimeError, "原图或标注"):
                self.prepare()

    def test_image_changes_rejected(self):
        Image.new("RGB", (100, 100), "red").save(self.a)
        with self.assertRaisesRegex(RuntimeError, "原图或标注"):
            self.analyze()

    def test_yaml_changes_rejected(self):
        self.write_yaml(names=["changed", "robot"])
        with self.assertRaisesRegex(RuntimeError, "配置"):
            self.analyze()

    def test_model_changes_rejected(self):
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "模型内容"):
            self.analyze()

    def test_selection_cannot_duplicate_or_add_unknown_images(self):
        for images in ([str(self.a), str(self.a)], ["/unknown.png"]):
            with self.subTest(images=images), self.assertRaisesRegex(ValueError, "范围"):
                self.analyze(sample_images=images)


class ExportTests(DatasetFixture):
    def test_source_changed_while_exporting_no_complete_report_published(self):
        import evaluation_worker
        report = self.analyze()
        target = self.root / "mutating-export" / "report.json"
        original = evaluation_worker.write_csv
        changed = False
        def mutate_during_export(path, rows):
            nonlocal changed
            original(path, rows)
            if not changed:
                changed = True
                self.model.write_bytes(b"model replaced while rendering plots")
        with patch("evaluation_worker.write_csv", side_effect=mutate_during_export):
            with self.assertRaisesRegex(RuntimeError, "模型内容"):
                export_report(target, report)
        self.assertFalse(target.exists())

    def test_actual_cli_preparation_analysis_and_decodable_exports(self):
        prepare_request = self.root / "prepare-request.json"
        prepare_request.write_text(json.dumps({"dataset_yaml": str(self.data), "split": "val"}), encoding="utf-8")
        manifest_path = self.root / "manifest.json"
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(self.request), encoding="utf-8")
        output = self.root / "report" / "report.json"
        command = [sys.executable, str(ROOT / "scripts" / "evaluation_worker.py")]
        environment = dict(os.environ)
        environment["YOLO_CONFIG_DIR"] = str(self.root / "ultralytics")
        environment["MPLCONFIGDIR"] = str(self.root / "matplotlib")
        prepared = subprocess.run([*command, "prepare", "--request", str(prepare_request), "--output", str(manifest_path)],
                                  env=environment, capture_output=True, text=True, timeout=40)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        process = subprocess.run([*command, "analyze", "--request", str(request_path), "--manifest", str(manifest_path),
                                  "--predictions", str(self.prediction_file), "--output", str(output)],
                                 env=environment, capture_output=True, text=True, timeout=80)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["rows"]), 91)
        self.assertEqual(report["metrics"]["TP"], 1)
        self.assertEqual(report["algorithm_version"], "vision-eval-1/highest-iou-stable-ties")
        for name in report["artifacts"]:
            artifact = output.parent / name
            self.assertTrue(artifact.is_file(), name)
            self.assertGreater(artifact.stat().st_size, 30, name)
            if artifact.suffix == ".png":
                with Image.open(artifact) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(image.height, 600)
                    image.verify()
        with (output.parent / "thresholds.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 91)
        # Saved JSON/GT remains usable even if the original dataset later disappears.
        self.a.unlink()
        self.assertEqual(json.loads(output.read_text())["manifest"]["samples"][0]["gt"][0]["class_id"], 0)


if __name__ == "__main__":
    unittest.main()
