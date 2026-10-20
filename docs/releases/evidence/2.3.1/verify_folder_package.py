#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Audit migrated folder/list import through the final read-only DEB application.

Reuses the successful public COCO8 package-matrix extraction. No extraction,
download, private dataset, source modification or Python model inference occurs.
The only application invocation uses its packaged launcher, Python and Qt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "packaging"))
import verify_evaluation_deb as qa


def hashes(paths):
    return {str(path): qa.digest(path) for path in paths}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deb", type=Path, required=True)
    parser.add_argument("--matrix-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {"schema_version": 1, "version": "2.3.1", "success": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": {}, "errors": [],
              "scope": "Final DEB read-only folder migration with stale absolute image list; ONNX CPU",
              "dataset_kind": "public_COCO8_real_ground_truth_copied_unchanged",
              "limitations": ["Four public COCO8 validation images verify import/evaluation plumbing only.",
                              "This is not a deployment benchmark, football accuracy claim or CUDA comparison."]}
    try:
        matrix = qa.read_json(args.matrix_report)
        require(matrix.get("success") is True and matrix.get("eligible_for_release") is True,
                "Wait for the complete successful final package matrix before reusing its extraction")
        require(matrix.get("version") == "2.3.1", "Matrix version is not 2.3.1")
        app = Path(matrix["workdir"]) / "extracted/opt/VisionStudio"
        require(app.is_dir(), "Final read-only package extraction is absent")
        binding = qa.bind_package(args.deb.resolve(), app)
        require(binding["version"] == "2.3.1" and binding["deb_sha256"] == matrix["deb_sha256"],
                "Reused extraction is not bound to the same final 2.3.1 DEB")
        qa.require_readonly(app)
        result.update(binding)
        result.update(package_bound=True, read_only_application=True, app_root=str(app),
                      binary_sha256=qa.digest(app / "build/bin/vision-studio"))
        result["checks"]["final_deb_binding_and_readonly"] = True

        public_yaml, provenance_path = args.dataset.resolve(), args.provenance.resolve()
        require(matrix["dataset"]["yaml"] == str(public_yaml), "Matrix uses another input dataset")
        require(matrix["dataset"]["yaml_sha256"] == qa.digest(public_yaml), "Public YAML changed since matrix")
        public_report = qa.read_json(Path(matrix["cases"]["onnx_cpu"]["report_directory"]) / "report.json")
        manifest = public_report["manifest"]
        require(manifest["data_yaml"] == str(public_yaml) and manifest["split"] == "val",
                "Expected the matrix's actual public Val dataset manifest")
        samples = manifest["samples"]
        require(len(samples) == 4, "This regression specifically expects the four public COCO8 Val images")
        public_root = Path(manifest["root"]).resolve()
        require(public_root == public_yaml.parent / "coco8", "Public dataset root differs from the supplied COCO8 folder")
        source_paths = [public_yaml, provenance_path]
        for sample in samples:
            for key in ("image", "label"):
                path = Path(sample[key]).resolve()
                require(public_root in path.parents, "Public input escaped the known COCO8 root")
                require(qa.digest(path) == sample[key + "_sha256"], "Public source differs from the matrix snapshot")
                source_paths.append(path)
        source_before = hashes(source_paths)
        provenance = qa.read_json(provenance_path)
        require(provenance.get("dataset") == "COCO8", "Expected documented public COCO8 provenance")
        result["dataset_provenance"] = provenance
        result["public_input_sha256"] = source_before

        workspace = Path(tempfile.mkdtemp(prefix="vision-studio-folder-package-qa-"))
        result["workdir"] = str(workspace)
        fixture = workspace / "migrated-dataset"
        state = workspace / "application"
        fixture.mkdir(); state.mkdir()
        for relative in ("images/val", "labels/val", "lists"):
            (fixture / relative).mkdir(parents=True)
        absent_old_root = workspace / "absent-original-export"
        expected_images, stale_images = [], []
        copies = []
        for sample in samples:
            image = Path(sample["image"])
            label = Path(sample["label"])
            target_image = fixture / "images/val" / image.name
            target_label = fixture / "labels/val" / label.name
            shutil.copyfile(image, target_image); shutil.copyfile(label, target_label)
            require(qa.digest(target_image) == qa.digest(image) and qa.digest(target_label) == qa.digest(label),
                    "Copied public image or original GT label changed")
            expected_images.append(str(target_image))
            stale_images.append(str(absent_old_root / "images/val" / image.name))
            copies.append({"image_sha256": qa.digest(image), "label_sha256": qa.digest(label),
                           "source_image": str(image), "source_label": str(label),
                           "fixture_image": str(target_image), "fixture_label": str(target_label)})
        (fixture / "lists/val.txt").write_text("\n".join(stale_images) + "\n", encoding="utf-8")
        config = fixture / "data.yaml"
        config.write_text("path: " + json.dumps(str(absent_old_root)) + "\nval: lists/val.txt\nnames: " +
                          json.dumps(manifest["names"], ensure_ascii=False) + "\n", encoding="utf-8")
        (fixture / "train_from_scratch.example.yaml").write_text("epochs: 100\nmodel: yolo.pt\n", encoding="utf-8")
        fixture_before = hashes(sorted(path for path in fixture.rglob("*") if path.is_file()))
        for path in fixture.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        fixture.chmod(0o555)
        qa.require_readonly(fixture)
        result.update(fixture=str(fixture), absent_original_root=str(absent_old_root),
                      fixture_input_sha256=fixture_before, public_copies=copies)
        result["checks"]["unchanged_public_val_fixture_and_readonly"] = True

        model = app / "models/yolov5n.onnx"
        request = {"dataset_yaml": str(fixture), "split": "val", "model_path": str(model),
                   "task": "yolov5", "device": "cpu", "device_index": 0, "input_size": 640,
                   "model_class_id": 0, "dataset_class_id": 0, "confidence": .25, "nms_iou": .45,
                   "color_mode": "color", "scan_start": .05, "scan_end": .95, "scan_step": .01,
                   "match_iou": .5}
        request_file, output = state / "request.json", state / "ui-smoke"
        qa.atomic_json(request_file, request)
        env = qa.isolated_environment(state, app, app / "runtime/bin/python", None, None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result["command"] = qa.run([app / "run-installed.sh", "--smoke-evaluation", output,
                                    "--evaluation-request", request_file], env, state, "application")
        smoke = qa.read_json(output / "evaluation-smoke-report.json")
        require(smoke.get("version") == "2.3.1", "Final launcher reported another application version")
        audited = qa.audit_case(smoke, state, model, "cpu", 0)
        actual = qa.read_json(Path(audited["report_directory"]) / "report.json")
        imported = actual["manifest"]
        require(actual["completed_images"] == 4 and actual["planned_images"] == 4,
                "Actual folder task did not process all four images")
        require(imported["dataset_source"] == str(fixture), "Directory import origin was lost")
        require(imported["data_yaml"] == str(config), "Unrelated training YAML was selected")
        require(imported["root_relocated"] is True and imported["root"] == str(fixture),
                "Stale dataset root was not relocated to the explicitly selected directory")
        require(imported["configured_root"] == str(absent_old_root), "Original configured root is not retained")
        require(bool(imported.get("relocation_notice")), "Relocation notice is absent")
        require(actual["request"]["sample_images"] == expected_images, "Absolute image-list remapping/order is wrong")
        require(not absent_old_root.exists(), "Regression unexpectedly created the old dataset location")
        result["case"] = audited
        result["smoke_report"] = str(output / "evaluation-smoke-report.json")
        result["checks"].update(real_onnx_cpu_four_images=True, directory_origin_preserved=True,
                                 unique_conventional_yaml_selected=True, stale_root_and_absolute_list_relocated=True,
                                 all_91_threshold_points=True, selected_png_json_csv_exports=True)

        require(hashes(source_paths) == source_before, "Public source files changed during the folder regression")
        require(hashes(sorted(path for path in fixture.rglob("*") if path.is_file())) == fixture_before,
                "Read-only fixture file content/list changed during import/evaluation")
        qa.require_readonly(app); qa.require_readonly(fixture)
        require({name: qa.digest(app / name) for name in binding["code_sha256"]} == binding["code_sha256"],
                "Final application code changed during the folder regression")
        result["checks"].update(public_and_fixture_inputs_unchanged=True, packaged_application_unchanged=True)
    except Exception as error:
        result["errors"].append(str(error))
    result["success"] = bool(result["checks"]) and all(result["checks"].values()) and not result["errors"]
    qa.atomic_json(args.report, result)
    print(json.dumps({"success": result["success"], "report": str(args.report.resolve()),
                      "workdir": result.get("workdir"), "checks": result["checks"], "errors": result["errors"]},
                     ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
