#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Preview or publish strict local 2.10.1 acceptance projections.

No QA inference, installation, Git mutation, networking or cleanup is executed.
--execute requires actual 17 final reports, finalizer and all 44 native suites.
Unknown strings, file names, model/data identities and metrics are not projected.
Raw originals stay local; optional raw logs must pass a conservative privacy scan.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
from public_evidence_schema import (VERSION, ASSETS, PUBLIC_REPORT_NAMES, SUITE_NAMES,
    REFACTOR_CHECKS, REBUILD_CHECKS, report_projection, strict_json, safe_relative)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "packaging/verify_deb.py").is_file())
AREA = ROOT / ("output/v" + VERSION)
RAW = AREA / "qa-final"
RELEASE = ROOT / "output/releases" / VERSION
TOOLS = AREA / "qa-tools"
OUTPUT = ROOT / "docs/releases/evidence" / VERSION / "final"
GATES = ("qa-report.json", "install-qa.json", "source-bundle-qa.json", "packaged-gpu-qa.json",
    "packaged-evaluation-qa.json", "folder-package-qa.json", "packaged-private-precheck.json",
    "packaged-dataset-session-qa.json", "packaged-model-yaml-qa.json", "packaged-preview-drop-qa.json",
    "packaged-evaluation-features-qa.json", "packaged-external-import-qa.json", "packaged-startup-preference-qa.json",
    "packaged-model-library-drop-qa.json", "packaged-decision-layout-qa.json",
    "refactor-architecture-qa.json", "source-rebuild-qa.json")
DERIVED = ("gpu-qa.json", "conversion-qa.json", "evaluation-qa.json", "ui-queue-qa.json",
    "dataset-session-qa.json", "preview-input-qa.json", "model-yaml-qa.json", "evaluation-features-qa.json",
    "model-library-drop-qa.json")
SHA = re.compile(r"[0-9a-f]{64}\Z")
HOST_OR_SECRET = re.compile(
    r"(?:/(?:home|tmp|root|Users|proc|sys|run|var|opt)/|[A-Za-z]:[\\/]|file://|"
    r"(?:^|[\"'\s])/[A-Za-z0-9_.-]+(?:/|\b)|\\\\[A-Za-z0-9_.-]+[\\/]|"
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\b(?:ghp_|github_pat_|sk_live_)[A-Za-z0-9_]+|"
    r"Authorization\s*:|Bearer\s+[A-Za-z0-9]|api[_ -]?key\s*[=:])", re.I)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Required regular original is absent")
    return path


def sha(path):
    digest = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = strict_json(regular(path).read_bytes())
    require(isinstance(value, dict), "Original JSON must be an object")
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def commitment(path):
    return {"bytes": regular(path).stat().st_size, "sha256": sha(path)}


def assert_public(value):
    """Defense in depth after allowlist projection; no sanitizing raw strings."""
    require(not HOST_OR_SECRET.search(json.dumps(value, ensure_ascii=False)), "Refuse private path/contact/secret in public projection")


def write_new(path, value):
    assert_public(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def tree_snapshot(root):
    require(root.is_dir() and not root.is_symlink(), "Final raw original directory is absent")
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Raw evidence contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            require(".." not in Path(relative).parts, "Raw original escapes evidence tree")
            files[relative] = commitment(path)
    require(files, "Final raw original tree is empty")
    return files


def tree_commitment(files):
    # File names may themselves identify private samples: publish only an
    # aggregate commitment, not an individual private-name or model hash map.
    return {"file_count": len(files), "total_bytes": sum(row["bytes"] for row in files.values()),
            "canonical_relative_file_map_sha256": canonical(files), "individual_private_names_published": False}


def safe_original(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Unsafe original reference")
    path = regular(root / relative)
    require(path.resolve().is_relative_to(root.resolve()), "Original reference escapes final evidence")
    return path


def reopen():
    """Reopen real completed originals without calling mutating aggregate()."""
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(ROOT / "packaging"))
    final = importlib.import_module("finalize_reports")
    regression = importlib.import_module("summarize_regressions")
    require(final.VERSION == VERSION and set(regression.SUITES) == SUITE_NAMES and len(SUITE_NAMES) == 44,
            "Reviewed finalizer/suite contract changed")
    assets = read(RAW / "final-assets.json")
    require(assets.get("schema_version") == 1 and assets.get("version") == VERSION and set(assets.get("assets", {})) == ASSETS,
            "All three actual immutable core assets are required")
    for name, row in assets["assets"].items():
        require(set(row) == {"bytes", "sha256"} and type(row["bytes"]) is int and 0 < row["bytes"] < 2 * 1024**3 and
                row == commitment(RELEASE / name), "Final immutable asset differs")
    expected_sums = "".join(assets["assets"][name]["sha256"] + "  " + name + "\n" for name in
        (f"vision-studio_{VERSION}-1_amd64.deb", f"vision-studio-{VERSION}-complete-source.tar.xz")).encode()
    require((RELEASE / "SHA256SUMS").read_bytes() == (RAW / "SHA256SUMS").read_bytes() == expected_sums,
            "Final asset checksum bytes differ")
    deb_sha = assets["assets"][f"vision-studio_{VERSION}-1_amd64.deb"]["sha256"]
    done = read(RAW / "finalize-reports.json")
    require(regular(AREA / "final-delivery-logs/finalize-reports.log").read_bytes() == (RAW / "finalize-reports.json").read_bytes() and
            done.get("success") is True and not done.get("errors") and done.get("deb_sha256") == deb_sha and
            done.get("manual_assets") == assets["assets"], "Actual finalizer original did not complete for these assets")
    reports = {}
    for name in (*GATES, *DERIVED):
        require(regular(RAW / name).read_bytes() == regular(RELEASE / name).read_bytes(), "Retained final report differs from release original")
        report = read(RAW / name)
        final.passed(report, name, deb_sha)
        reports[name] = report
    ctest = read(RAW / "ctest-qa.json")
    require(ctest.get("version") == VERSION and ctest.get("success") is True and ctest.get("errors") == [] and
            all(type(ctest.get(key)) is int and ctest[key] == 0 for key in ("failed", "skipped", "blacklisted")),
            "Full current CTest acceptance has not passed")
    suites = ctest.get("suites", [])
    require(len(suites) == ctest.get("effective_suite_count") == 44 and {row.get("name") for row in suites} == SUITE_NAMES,
            "Exactly all 44 suites must actually execute")
    suite_originals, totals = {}, {"qt_tests_passed": 0, "python_tests_passed": 0}
    for row in suites:
        name = row["name"]
        require(row.get("skipped") is False and type(row.get("exit_code")) is int and row["exit_code"] == 0,
                "A native suite was skipped or failed")
        final.suite(ctest, RAW, name)
        log, xml = safe_original(RAW, row["log"]), safe_original(RAW, row["junit"])
        regression.validate_junit(xml, name)
        require(sha(xml) == row["junit_sha256"], "Original JUnit changed")
        parsed = regression.parse_actual_totals(log.read_text(encoding="utf-8"), name)
        totals["qt_tests_passed"] += sum(item["passed"] for item in parsed["qt_totals"])
        totals["python_tests_passed"] += parsed["python_tests"]
        suite_originals[name] = {"log": log, "junit": xml}
    require(all(ctest.get(key) == count for key, count in totals.items()), "Actual complete-suite counts disagree")
    current = regression.code_hashes(ROOT)
    require(current == ctest.get("source_code_input_sha256") == reports["source-bundle-qa.json"].get("production_worktree_input_sha256") and
            reports["source-bundle-qa.json"].get("production_worktree_matches_frozen_source") is True,
            "Current production source does not match tested complete source")
    architecture = importlib.import_module("verify_refactor_architecture").audit_completed(ROOT, RELEASE, RAW, reports["refactor-architecture-qa.json"])
    rebuilt = importlib.import_module("verify_source_rebuild").audit_completed(ROOT, RELEASE, RAW, reports["source-rebuild-qa.json"])
    require(set(architecture["checks"]) == REFACTOR_CHECKS and set(rebuilt["checks"]) == REBUILD_CHECKS and
            done.get("refactor_architecture_checks") == architecture["checks"] and done.get("source_rebuild_checks") == rebuilt["checks"],
            "New finalizer gate results differ from reopened actual originals")
    final.audit_startup_preferences(reports["packaged-startup-preference-qa.json"], deb_sha)
    decision = importlib.import_module("decision_layout_gate").audit(reports["packaged-decision-layout-qa.json"], ROOT)
    require(done.get("decision_layout_originals_reopened") == decision, "Decision-layout originals no longer match completed aggregate")
    drop = importlib.import_module("model_library_drop_gate").audit_model_library_drop(ROOT, RELEASE, RAW)
    require(done.get("model_library_drop_checks") == drop["checks"], "Model-library gate changed after finalization")
    freeze = read(RAW / "source-freeze.json")
    commit = freeze.get("archive_input_commit", freeze.get("source_commit"))
    require(freeze.get("version") == VERSION and isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
            "Final original source-freeze receipt is absent")
    reports.update({"ctest-qa.json": ctest, "finalize-reports.json": done})
    return {"reports": reports, "assets": assets, "ctest": ctest, "suite_originals": suite_originals,
        "source_inputs": current, "archive_input_commit": commit, "rebuild": rebuilt}


def prepare_public_logs(context, requested):
    logs = {}
    for name in sorted(set(requested)):
        require(name in SUITE_NAMES, "Raw public log must name one reviewed suite")
        path = context["suite_originals"][name]["log"]
        text = path.read_text(encoding="utf-8")
        require(len(text) < 16 * 1024**2 and not HOST_OR_SECRET.search(text),
                "Requested actual original log contains a host path/contact/secret; refusing public bytes")
        logs["logs/" + name + ".log"] = path
    return logs


def dependency_hashes():
    result = {}
    for directory, label in ((TOOLS, "qa-tools"), (ROOT / "packaging", "packaging")):
        for path in sorted(directory.glob("*.py")):
            require(re.fullmatch(r"[a-z][a-z0-9_]*\.py", path.name), "Unexpected verifier program name")
            result[label + "/" + path.name] = sha(path)
    return result


def local_archive_members(context):
    """Fixed QA reports/logs only; never recurse into artifacts or model inputs."""
    members = {}
    for name in (*GATES, *DERIVED, "ctest-qa.json", "source-freeze.json", "production-inputs-before.json", "final-assets.json", "finalize-reports.json"):
        members["qa-final/" + name] = regular(RAW / name)
    for originals in context["suite_originals"].values():
        for path in originals.values():
            members["qa-final/" + path.relative_to(RAW).as_posix()] = path
    for path in sorted((AREA / "final-delivery-logs").glob("*.log")):
        members["final-delivery-logs/" + path.name] = regular(path)
    receipt_path = regular(context["rebuild"]["receipt_file"])
    receipt = read(receipt_path)
    members["source-rebuild/execution-receipt.json"] = receipt_path
    for row in receipt["commands"]:
        require(row["name"] in {"configure", "build", "version", "cpu_smoke"}, "Unknown rebuild command role")
        for stream in ("stdout", "stderr"):
            path = regular(row[stream])
            require(sha(path) == row[stream + "_sha256"], "Rebuild original log changed")
            members["source-rebuild/" + row["name"] + "." + stream + ".log"] = path
    require(all(path.suffix in {".json", ".log", ".xml"} for path in members.values()),
            "Local archive must contain QA documents/logs only")
    return members


def archive_local_originals(members, destination):
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", str(destination.relative_to(ROOT))],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(ignored.returncode == 0, "Local original archive must be ignored by Git before creation")
    require(not destination.exists() and not destination.is_symlink(), "Refuse replacing original local archive")
    with tarfile.open(destination, "x:xz") as archive:
        for name, path in sorted(members.items()):
            safe_relative(name)
            regular(path)
            archive.add(path, arcname=name, recursive=False)
    return {"file": destination.name, **commitment(destination), "local_only": True,
        "git_ignored": True, "qa_document_count": len(members), "user_images_models_and_artifact_directories_included": False}


def evidence_only_transition(project, source_commit, evidence_commit, allowed_paths):
    """Optional read-only A→B check, called by root only after B is committed."""
    require(all(re.fullmatch(r"[0-9a-f]{40}", commit or "") for commit in (source_commit, evidence_commit)), "Use real full commit identities")
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=project).decode().strip()
    require(git("merge-base", "--is-ancestor", source_commit, evidence_commit) == "", "Source commit is not an ancestor")
    # Disabling rename detection exposes both the old and new paths, so moving
    # production source into an allowed docs path cannot evade this whitelist.
    changed = git("diff", "--name-only", "--no-renames", source_commit, evidence_commit).splitlines()
    allowed = set(allowed_paths)
    require(changed and set(changed) <= allowed, "A→B contains a file outside the explicit evidence whitelist")
    for name in changed:
        safe_relative(name)
        require(name in {"README.md", "CHANGELOG.md", f"docs/releases/{VERSION}.md", "docs/architecture/refactor-progress.md"} or
                name.startswith(f"docs/releases/evidence/{VERSION}/"), "A→B whitelist includes a non-evidence production file")

    # Existing evidence is immutable. Disabling rename/copy detection exposes
    # a moved original as a deletion plus an addition, so its deletion is still
    # rejected. NUL-delimited status records keep literal paths unambiguous.
    status_bytes = subprocess.check_output(["git", "diff", "--name-status", "--no-renames", "-z",
                                            source_commit, evidence_commit], cwd=project)
    status_parts = status_bytes.split(b"\0")
    require(status_parts[-1] == b"" and len(status_parts) % 2 == 1, "Malformed A→B Git status records")
    statuses = {}
    for index in range(0, len(status_parts) - 1, 2):
        status, name = status_parts[index].decode("ascii"), status_parts[index + 1].decode("utf-8")
        require(name not in statuses, "Duplicate A→B Git status path")
        statuses[name] = status
    require(set(statuses) == set(changed), "A→B Git name/status identities disagree")
    for name in changed:
        if name.startswith(f"docs/releases/evidence/{VERSION}/"):
            require(statuses[name] == "A", "A→B evidence files must be newly added; existing originals cannot be modified, deleted or moved")

    def blob_sha256(commit, name):
        # -z preserves literal names; an absent tree entry means the file did
        # not exist in this commit, not that a failed git-show was ignored.
        entry = subprocess.check_output(["git", "ls-tree", "-z", commit, "--", name], cwd=project)
        if not entry:
            return None
        entries = entry.rstrip(b"\0").split(b"\0")
        require(len(entries) == 1, "A→B identity must resolve to exactly one Git file")
        header, actual_name = entries[0].split(b"\t", 1)
        mode, kind, object_id = header.split()
        require(kind == b"blob" and mode in {b"100644", b"100755"} and actual_name.decode("utf-8") == name,
                "A→B evidence entry is not an ordinary Git file")
        data = subprocess.check_output(["git", "cat-file", "blob", object_id.decode("ascii")], cwd=project)
        return hashlib.sha256(data).hexdigest()

    files = []
    for name in sorted(changed):
        before, after = blob_sha256(source_commit, name), blob_sha256(evidence_commit, name)
        require(before is not None or after is not None, "Changed A→B file is absent in both Git trees")
        files.append({"file": name, "status": "added" if before is None else "deleted" if after is None else "modified",
                      "before_sha256": before, "after_sha256": after})
    return {"schema_version": 1, "version": VERSION, "source_commit": source_commit, "evidence_commit": evidence_commit,
        "only_explicit_documentation_and_evidence_files_changed": True, "changed_files": sorted(changed),
        "files": files, "hash_scope": "SHA256 of literal Git blob bytes; absent side is null",
        "production_files_changed": False, "git_tree_comparison_executed": True}


def write_public_checksums(directory):
    """Write checksums for the files actually shipped in the public projection."""
    with (directory / "SHA256SUMS").open("x") as output:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name not in {"SHA256SUMS", "local-originals.tar.xz"}:
                output.write(sha(path) + "  " + path.relative_to(directory).as_posix() + "\n")


def collect(args):
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Final public projection destination must be new")
    context = reopen()
    raw_before = tree_snapshot(RAW)
    dependencies_before = dependency_hashes()
    logs = prepare_public_logs(context, args.public_log_suite)
    members = local_archive_members(context) if args.archive_local_originals else {}
    payloads = {}
    for name, report in context["reports"].items():
        require(name in PUBLIC_REPORT_NAMES, "Report role is not publicly allowlisted")
        projected = report_projection(report)
        require(projected is not None, "Actual report has no approved projection")
        projected.update(version=VERSION, original_report=commitment(RAW / name))
        payloads["reports/" + name] = projected
    payloads["final-assets.json"] = context["assets"]
    originals = {name: commitment(RAW / name) for name in (*GATES, *DERIVED,
        "ctest-qa.json", "source-freeze.json", "production-inputs-before.json", "final-assets.json", "finalize-reports.json")}
    for name, files in sorted(context["suite_originals"].items()):
        for stream, path in files.items():
            originals["suites/" + name + "." + stream] = commitment(path)
    rebuild = context["rebuild"]
    originals["rebuild/execution-receipt.json"] = commitment(rebuild["receipt_file"])
    for row in rebuild["commands"]:
        for stream in ("stdout", "stderr"):
            originals["rebuild/" + row["name"] + "." + stream + ".log"] = commitment(row[stream])
    payloads["original-commitments.json"] = {"schema_version": 1, "version": VERSION, "not_a_finalizer_input": True,
        "raw_originals_retained_local_only": True, "complete_final_raw_tree": tree_commitment(raw_before),
        "fixed_role_original_files": originals, "archive_input_commit": context["archive_input_commit"],
        "source_fingerprint_canonical_sha256": canonical(context["source_inputs"]),
        "private_individual_file_identities_published": False}
    payloads["tools/dependency-hashes.json"] = {"schema_version": 1, "version": VERSION,
        "local_audit_dependencies_sha256": dependencies_before,
        "copied_public_tool_sha256": {"public_evidence.py": sha(Path(__file__)),
            "public_evidence_schema.py": sha(Path(__file__).with_name("public_evidence_schema.py"))},
        "local_originals_required_to_reaudit": True}
    for value in payloads.values():
        assert_public(value)
    if args.archive_local_originals:
        ignored = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", str((OUTPUT / "local-originals.tar.xz").relative_to(ROOT))], cwd=ROOT)
        require(ignored.returncode == 0, "Local archive is not Git-ignored")
    OUTPUT.mkdir(parents=True)
    for name, value in sorted(payloads.items()):
        write_new(OUTPUT / name, value)
    for name, source in logs.items():
        destination = OUTPUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as original, destination.open("xb") as output:
            shutil.copyfileobj(original, output)
        require(sha(source) == sha(destination), "Approved original log changed during exact copy")
    for source in (Path(__file__), Path(__file__).with_name("public_evidence_schema.py")):
        with regular(source).open("rb") as original, (OUTPUT / "tools" / source.name).open("xb") as saved:
            shutil.copyfileobj(original, saved)
    if members:
        write_new(OUTPUT / "local-archive-commitment.json", archive_local_originals(members, OUTPUT / "local-originals.tar.xz"))
    require(tree_snapshot(RAW) == raw_before and dependency_hashes() == dependencies_before,
            "Raw originals or audit tools changed during projection")
    regression = importlib.import_module("summarize_regressions")
    require(regression.code_hashes(ROOT) == context["source_inputs"], "Production source changed during projection")
    manifest = {"schema_version": 1, "version": VERSION, "public_projection": True, "not_a_finalizer_input": True,
        "actual_gate_report_count": len(GATES), "actual_suite_count": len(context["ctest"]["suites"]),
        "qa_inference_rerun": False, "originals_modified": False, "raw_logs_published": sorted(logs),
        "no_private_images_models_metrics_or_argv_projected": True,
        "unexecuted_platforms": ["Windows", "physical camera"],
        "files": [{"file": path.relative_to(OUTPUT).as_posix(), **commitment(path)}
                  for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "local-originals.tar.xz"]}
    write_new(OUTPUT / "public-projection-manifest.json", manifest)
    write_public_checksums(OUTPUT)
    print(json.dumps({"state": "actual_originals_projected", "version": VERSION, "output": str(OUTPUT),
        "gate_count": len(GATES), "suite_count": 44, "raw_logs_copied": len(logs), "qa_inference_rerun": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--public-log-suite", choices=sorted(SUITE_NAMES), action="append", default=[],
                        help="Optional exact original CTest log; any detected private path/secret causes refusal")
    parser.add_argument("--archive-local-originals", action="store_true",
                        help="Retain QA reports/logs/JUnit only in Git-ignored local-originals.tar.xz; no images/models/artifact recursion")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"version": VERSION, "state": "preview_only", "required_final_reports": len(GATES),
            "required_suites": len(SUITE_NAMES), "output": str(OUTPUT), "output_exists": OUTPUT.exists(),
            "raw_log_suites_requested": args.public_log_suite, "local_archive_requested": args.archive_local_originals,
            "note": "No final reports projected, no files created and no presumed pass. --execute reopens actual final acceptance originals."}, indent=2))
        return 0
    collect(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError, tarfile.TarError) as error:
        print("Public projection refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
