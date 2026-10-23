# SPDX-License-Identifier: AGPL-3.0-only
"""Hand-calculated scene, report, figure and publication-scope regressions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from PIL import Image

from evaluation_metrics import GROUPS, prepare_records, summarize_groups
from evaluation_reporting import (FIGURES, csv_tables, figure_options, markdown_report,
                                  save_report_artifacts, tex_escape, write_csv)
from evaluation_tests import DatasetFixture
from evaluation_worker import export_report


class SceneMetricsTests(DatasetFixture):
    def review_metadata(self):
        self.write_yaml(names=["football"])
        metadata = self.root / "metadata.csv"
        fields = ["图片", "split", "annotation_status", "label_review_required", "scene_review_required",
                  "Small", "Occluded", "Blurred", "No Ball"]
        with metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for image in (self.a, self.b, self.c):
                writer.writerow({"图片": image.name, "split": "val", "label_review_required": 0,
                                 "scene_review_required": 0, "annotation_status": "negative_confirmed" if image == self.c else "reviewed_annotation",
                                 "Small": int(image != self.c), "Occluded": int(image == self.a), "Blurred": 0,
                                 "No Ball": int(image == self.c)})
        self.manifest = self.prepare(metadata_path=str(metadata))

    def test_presence_detects_existence_independently_of_localization(self):
        report = self.analyze()
        self.assertEqual((report["metrics"]["TP"], report["metrics"]["FN"]), (1, 1))
        self.assertEqual(report["metrics"]["presence_confusion"], {"TP": 2, "FP": 0, "FN": 0, "TN": 1})
        self.assertEqual(self.analyze(confidence=.99)["metrics"]["presence_confusion"], {"TP": 0, "FP": 0, "FN": 2, "TN": 1})
        self.assertEqual(self.analyze(confidence=.25)["metrics"]["presence_confusion"], {"TP": 2, "FP": 1, "FN": 0, "TN": 0})

    def test_no_metadata_is_unknown_instead_of_no_difficulty(self):
        report = self.analyze()
        self.assertEqual(tuple(report["groups"]), GROUPS)
        self.assertEqual(report["scene_review"], {"reviewed_images": 0, "unreviewed_images": 3, "source": "unreviewed"})
        for name in ("Small", "Occluded", "Blurred", "No difficulty"):
            group = report["groups"][name]
            self.assertFalse(group["available"])
            self.assertEqual(group["review_status"], "unreviewed")
            for field in ("images", "TP", "recall", "mAP50"):
                self.assertIsNone(group[field])
            self.assertEqual(group["sample_images"], [])
        self.assertEqual(report["groups"]["No Ball"]["sample_images"], [str(self.c)])
        self.assertEqual(report["groups"]["All"]["sample_images"], self.request["sample_images"])
        self.assertTrue(all(row["Small"] is None and row["Occluded"] is None and not row["scene_reviewed"]
                            for row in report["per_image"]))

    def test_reviewed_groups_publish_ap_but_single_image_is_descriptive(self):
        self.review_metadata()
        report = self.analyze()
        groups = report["groups"]
        self.assertEqual(report["scene_review"]["reviewed_images"], 3)
        self.assertEqual(groups["Small"]["images"], 2)
        self.assertGreater(groups["Small"]["mAP50"], 0)
        self.assertTrue(groups["Small"]["ap_published"])
        self.assertEqual((groups["Small"]["TP"], groups["Small"]["FP"], groups["Small"]["FN"]), (1, 2, 1))
        self.assertEqual(groups["Small"]["sample_images"], [str(self.a), str(self.b)])
        self.assertEqual(groups["Occluded"]["images"], 1)
        self.assertIsNone(groups["Occluded"]["mAP50"])
        self.assertFalse(groups["Occluded"]["ap_published"])
        self.assertGreater(groups["Occluded"]["partial_metrics"]["mAP50"], 0)
        self.assertTrue(groups["Blurred"]["available"])
        self.assertEqual(groups["Blurred"]["images"], 0)
        self.assertIsNone(groups["Blurred"]["recall"])

    def test_cancelled_and_subset_do_not_publish_group_ap(self):
        self.review_metadata()
        self.write_predictions(self.predictions[:2])
        for changes in ({"status": "cancelled"}, {"sample_images": [str(self.a), str(self.b)], "full_split": False}):
            report = self.analyze(**changes)
            for group in report["groups"].values():
                self.assertFalse(group["ap_published"])
                self.assertIsNone(group["mAP50"])
            self.assertGreater(report["groups"]["Small"]["partial_metrics"]["mAP50"], 0)
            self.assertEqual(report["groups"]["All"]["sample_images"], [str(self.a), str(self.b)])

    def test_no_difficulty_requires_review_and_excludes_target_negatives(self):
        records = prepare_records(self.manifest["samples"], self.predictions, 32, 0, scene_reviewed=True)
        groups = summarize_groups(records, .5, .5, True)
        self.assertEqual(groups["No difficulty"]["sample_images"], [str(self.a), str(self.b)])
        self.assertEqual(groups["No Ball"]["sample_images"], [str(self.c)])
        self.assertTrue(groups["No difficulty"]["ap_published"])

    def test_target_negative_group_follows_mapping_not_metadata_flag(self):
        report = self.analyze(dataset_class_id=1)
        group = report["groups"]["No Ball"]
        self.assertEqual(group["images"], 3)
        self.assertEqual(group["sample_images"], self.request["sample_images"])
        self.assertEqual(group["review_status"], "derived_from_gt")
        self.assertIsNone(group["mAP50"])
        self.assertIsNone(group["recall"])


class ReportOutputTests(DatasetFixture):
    def test_default_and_explicit_formats_dpi_validation(self):
        self.assertEqual(figure_options({}), (["png", "svg", "pdf"], 300))
        self.assertEqual(figure_options({"figure_formats": ["svg", "png", "svg"], "figure_dpi": 144}), (["svg", "png"], 144))
        for changes in ({"figure_formats": []}, {"figure_formats": ["eps"]}, {"figure_formats": True},
                        {"figure_dpi": True}, {"figure_dpi": "300"}, {"figure_dpi": 71}, {"figure_dpi": 1201}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.analyze(**changes)

    def test_all_nine_figures_and_three_formats_are_decodable(self):
        report = self.analyze()
        output = self.root / "complete-paper-report" / "report.json"
        export_report(output, report)
        artifact_names = set(report["artifacts"])
        for name in FIGURES:
            for fmt in ("png", "svg", "pdf"):
                self.assertIn(f"{name}.{fmt}", artifact_names)
            with Image.open(output.parent / f"{name}.png") as picture:
                self.assertGreater(picture.width, 1000)
                self.assertGreater(picture.height, 600)
                self.assertAlmostEqual(picture.info["dpi"][0], 300, places=1)
                picture.verify()
            self.assertTrue((output.parent / f"{name}.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertEqual(ET.parse(output.parent / f"{name}.svg").getroot().tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(report["export_options"], {"formats": ["png", "svg", "pdf"], "dpi": 300})
        self.assertEqual(json.loads(output.read_text())["artifacts"], report["artifacts"])
        self.assertEqual(len([name for name in artifact_names if Path(name).suffix in (".png", ".svg", ".pdf")]), 27)
        with (output.parent / "false-positive-no-ball.csv").open(encoding="utf-8-sig", newline="") as stream:
            table = csv.DictReader(stream)
            self.assertIn("image", table.fieldnames)
            self.assertEqual(list(table), [])
        with (output.parent / "ap-by-iou.csv").open(encoding="utf-8-sig", newline="") as stream:
            ap_rows = list(csv.DictReader(stream))
        self.assertEqual(len(ap_rows), 10)
        self.assertAlmostEqual(float(ap_rows[0]["AP"]), report["metrics"]["mAP50"])
        self.assertAlmostEqual(float(ap_rows[5]["AP"]), report["metrics"]["mAP75"])
        with (output.parent / "scene-metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            groups = {row["scene"]: row for row in csv.DictReader(stream)}
        self.assertEqual(groups["Small"]["images"], "")
        self.assertEqual(groups["No Ball"]["images"], "1")
        text = (output.parent / "REPORT.md").read_text()
        for marker in ("Wilson", "FPPI", "阳性图片漏检率", "未审核 3 张", "背景 TN 没有定义", "预测存在不表示定位正确"):
            self.assertIn(marker, text)

    def test_tables_latex_agree_with_operating_point_and_withhold_partial_ap(self):
        self.write_predictions(self.predictions[:2])
        report = self.analyze(status="cancelled")
        report["request"]["model_path"] = "/model/test_&{1}.onnx"
        output = self.root / "tables-only"
        output.mkdir()
        csv_tables(output, report, write_csv)
        with (output / "ap-by-iou.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertTrue(all(row["AP"] == "" and row["ap_published"] == "False" for row in csv.DictReader(stream)))
        with (output / "per-image.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(int(row["TP"]) for row in rows), report["metrics"]["TP"])
        self.assertEqual(sum(int(row["FP"]) for row in rows), report["metrics"]["FP"])
        self.assertEqual(sum(int(row["FN"]) for row in rows), report["metrics"]["FN"])
        latex = (output / "paper-row.tex").read_text()
        self.assertIn(r"test\_\&\{1\}.onnx", latex)
        self.assertIn("subset & 2 & N/A & N/A", latex)
        self.assertIn("confidence=0.5", latex)
        self.assertTrue(latex.endswith(" \\\\\n"))

    def test_svg_selection_does_not_generate_unrequested_formats_and_labels_fit(self):
        import matplotlib.figure
        report = self.analyze(figure_formats=["svg"], figure_dpi=144)
        output = self.root / "svg-only"
        original = matplotlib.figure.Figure.savefig
        checked = []
        def check_layout(figure, path, **kwargs):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            bounds = figure.bbox
            for axes in figure.axes:
                for text in axes.texts + [axes.title, axes.xaxis.label, axes.yaxis.label]:
                    if not text.get_visible() or not text.get_text():
                        continue
                    box = text.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, bounds.x0 - 2, (Path(path).name, text.get_text()))
                    self.assertGreaterEqual(box.y0, bounds.y0 - 2, (Path(path).name, text.get_text()))
                    self.assertLessEqual(box.x1, bounds.x1 + 2, (Path(path).name, text.get_text()))
                    self.assertLessEqual(box.y1, bounds.y1 + 2, (Path(path).name, text.get_text()))
            checked.append(Path(path).stem)
            return original(figure, path, **kwargs)
        with patch.object(matplotlib.figure.Figure, "savefig", check_layout):
            artifacts = save_report_artifacts(output, report)
        self.assertEqual(set(checked), set(FIGURES))
        self.assertFalse(list(output.glob("*.png")))
        self.assertFalse(list(output.glob("*.pdf")))
        self.assertEqual(len([name for name in artifacts if name.endswith(".svg")]), 9)

    def test_legacy_optional_fields_render_as_unknown_without_fabricating_data(self):
        report = self.analyze()
        report.pop("scene_review")
        report["metrics"].pop("presence_confusion")
        report["groups"] = {}
        text = markdown_report(report, ["svg"], 300)
        self.assertIn("N/A", text)
        self.assertNotIn("| 图片数 | 0 | 0 | 0 | 0 |", text)
        self.assertEqual(tex_escape("a_b&c%"), r"a\_b\&c\%")


if __name__ == "__main__":
    unittest.main()
