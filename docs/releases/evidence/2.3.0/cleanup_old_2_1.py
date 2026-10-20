#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Preview two audited 2.1.0 source copies; execute only after release receipts.

This script cannot prune a release directory, run Git, touch runtime or user
data, or select arbitrary paths. It removes only the two literal candidates
below after checking their complete audited inventories. All siblings remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys

BASE = Path("/home/misaka/文档/ChatGPT/yolo/output/v2.1.0")
CANDIDATES = (BASE / "clean-source-verified", BASE / "clean-source-verified.tar")
REPOSITORY = "misaka-ning/vision-studio"
PROTECTED_NAMES = {".git", "runtime", "gpu-runtime", "recordings", "preferences.ini", "history.json"}


class CleanupError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise CleanupError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def safe_ancestors(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path)), "Path must be absolute and canonical")
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"Symlink path refused: {component}")


def identity(path):
    safe_ancestors(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), f"Not a regular file/directory: {path}")
    return {"type": "directory" if stat.S_ISDIR(info.st_mode) else "file", "device": info.st_dev,
            "inode": info.st_ino, "mode": stat.S_IMODE(info.st_mode), "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "links": info.st_nlink,
            "allocated_bytes": info.st_blocks * 512}


def inventory(path):
    root = identity(path)
    entries = {}
    if root["type"] == "file":
        root["sha256"] = sha(path)
    else:
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in sorted(dirs + files):
                item = Path(directory) / name
                rel = item.relative_to(path).as_posix()
                # Chromium's preserved @babel/runtime license directory is
                # source-notice material, not an installed runtime environment.
                notice = rel.startswith("packaging/licenses/qt/source-notices/")
                require(name != ".git" and (name not in PROTECTED_NAMES or notice),
                        f"Protected name inside candidate: {item}")
                row = identity(item)
                if row["type"] == "file": row["sha256"] = sha(item)
                entries[item.relative_to(path).as_posix()] = row
    result = {"root": root, "entries": entries}
    result["fingerprint"] = canonical(result)
    return result


def read_json(path):
    safe_ancestors(path)
    require(path.is_file(), f"Missing regular JSON receipt/audit: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assets_for(version, values):
    expected = {f"vision-studio_{version}-1_amd64.deb", f"vision-studio-{version}-complete-source.tar.xz", "SHA256SUMS"}
    require(isinstance(values, list) and len(values) == 3, f"{version}: require three officially verified core assets")
    assets = {row.get("name"): row for row in values}
    require(set(assets) == expected, f"{version}: wrong verified asset names")
    for name, row in assets.items():
        require(type(row.get("bytes")) is int and 0 < row["bytes"] < 2 * 1024**3 and
                re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) and
                type(row.get("asset_id")) is int and row["asset_id"] > 0 and row.get("verification"),
                f"{version}: incomplete official asset verification: {name}")
    return assets


def official_verify(receipt, version, release_dir):
    data = read_json(receipt)
    require(data.get("success") is True and data.get("repository") == REPOSITORY and
            data.get("tag") == "v" + version and data.get("prerelease") is False and
            data.get("url") == f"https://github.com/{REPOSITORY}/releases/tag/v{version}" and
            re.fullmatch(r"[0-9a-f]{40}", data.get("commit", "")),
            f"{version}: not a successful matching official github_release.py verify receipt")
    assets = assets_for(version, data.get("verified_assets"))
    safe_ancestors(release_dir)
    require(release_dir.is_dir() and release_dir.name == version, f"{version}: retained release directory is missing")
    for name, row in assets.items():
        path = release_dir / name
        safe_ancestors(path)
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"{version}: retained local asset differs from remote verification: {name}")
    return {"receipt": str(receipt), "receipt_sha256": sha(receipt), "tag": data["tag"],
            "commit": data["commit"], "assets": list(assets.values()), "release_dir": str(release_dir)}


def verify_old_backup(audit, old_verify, prune_receipt):
    if old_verify:
        data = read_json(old_verify)
        require(data.get("success") is True and data.get("repository") == REPOSITORY and
                data.get("tag") == "v2.1.0" and data.get("prerelease") is False,
                "Old 2.1.0 verification receipt does not match")
        assets = assets_for("2.1.0", data.get("verified_assets"))
        proof = {"receipt": str(old_verify), "sha256": sha(old_verify), "kind": "official verify"}
    else:
        require(prune_receipt is not None, "Provide the official prune receipt or a fresh official 2.1.0 verify receipt")
        data = read_json(prune_receipt)
        require(data.get("success") is True, "Official prune receipt did not succeed")
        values = data.get("verified_candidates", []) + data.get("verified_kept", [])
        rows = [r for r in values if r.get("url") == f"https://github.com/{REPOSITORY}/releases/tag/v2.1.0"]
        require(len(rows) == 1 and re.fullmatch(r"[0-9a-f]{40}", rows[0].get("commit", "")),
                "Official prune receipt has no unique remotely verified 2.1.0 backup")
        assets = assets_for("2.1.0", rows[0].get("assets"))
        proof = {"receipt": str(prune_receipt), "sha256": sha(prune_receipt), "kind": "official prune remote preflight"}
    complete = assets["vision-studio-2.1.0-complete-source.tar.xz"]
    require(complete["sha256"] == audit["backup"]["complete_source_sha256"] and
            complete["bytes"] == audit["backup"]["complete_source_bytes"],
            "Old corresponding-source backup differs from the audited archive")
    proof["complete_source"] = complete
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path(__file__).with_name("old-2.1-audit.json"))
    parser.add_argument("--verify-2-3", type=Path)
    parser.add_argument("--verify-2-2", type=Path)
    parser.add_argument("--release-2-3", type=Path)
    parser.add_argument("--release-2-2", type=Path)
    parser.add_argument("--old-verify", type=Path)
    parser.add_argument("--prune-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = {"schema_version": 1, "version": "2.3.0", "success": False, "dry_run": not args.execute,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "removed": [], "errors": []}
    try:
        if args.report:
            report_path = args.report.absolute()
            safe_ancestors(report_path)
            require(not report_path.exists() and not report_path.is_relative_to(BASE),
                    "Cleanup report must be new and outside the old candidate/state tree")
        audit_path = args.audit.absolute()
        audit = read_json(audit_path)
        require(audit.get("schema_version") == 1 and audit.get("success") is True and
                audit.get("read_only") is True and audit.get("version") == "2.3.0", "Unsupported or failed audit")
        candidates = audit.get("candidates", [])
        require(len(candidates) == 2 and [r.get("path") for r in candidates] == [str(p) for p in CANDIDATES],
                "Audit must name exactly the two literal source-copy candidates")
        before = {}
        for row, path in zip(candidates, CANDIDATES):
            require(row.get("safe_to_delete_after_remote_verification") is True, f"Candidate not approved by audit: {path}")
            require(path.parent == BASE and path not in {Path.cwd(), Path(__file__).resolve().parents[4]}, "Protected working tree")
            current = inventory(path)
            require(current == row.get("inventory"), f"Candidate changed since read-only audit: {path}")
            before[str(path)] = current
        result.update({"audit": str(audit_path), "audit_sha256": sha(audit_path), "candidates": [str(p) for p in CANDIDATES],
                       "logical_bytes": sum(r["logical_bytes"] for r in candidates),
                       "preservation_policy": "All other old QA tools, logs, models, states, user data and release folders remain untouched"})
        blocked, receipts = [], []
        for version, receipt, folder in (("2.3.0", args.verify_2_3, args.release_2_3), ("2.2.0", args.verify_2_2, args.release_2_2)):
            try:
                require(receipt is not None and folder is not None, f"{version}: official verify receipt and retained release directory required")
                receipts.append(official_verify(receipt.absolute(), version, folder.absolute()))
            except (CleanupError, OSError, ValueError, KeyError) as exc:
                blocked.append(str(exc))
        try:
            result["old_backup_receipt"] = verify_old_backup(audit, args.old_verify.absolute() if args.old_verify else None,
                                                            args.prune_receipt.absolute() if args.prune_receipt else None)
        except (CleanupError, OSError, ValueError, KeyError) as exc:
            blocked.append(str(exc))
        result["verified_retained"] = receipts
        result["blocked_proofs"] = blocked
        result["ready_to_execute"] = not blocked
        if args.execute:
            require(not blocked, "Cleanup blocked: " + "; ".join(blocked))
            require(getattr(shutil.rmtree, "avoids_symlink_attacks", False), "Safe descriptor-based rmtree is unavailable")
            for path in CANDIDATES:
                require(inventory(path) == before[str(path)], f"Candidate changed during receipt checks: {path}")
            require(sha(audit_path) == result["audit_sha256"], "Audit changed during verification")
            for proof in receipts:
                require(sha(Path(proof["receipt"])) == proof["receipt_sha256"], "Retained verification receipt changed")
            proof = result["old_backup_receipt"]
            require(sha(Path(proof["receipt"])) == proof["sha256"], "Old backup verification receipt changed")
            for path in CANDIDATES:
                if before[str(path)]["root"]["type"] == "directory": shutil.rmtree(path)
                else: path.unlink()
                result["removed"].append(str(path))
        result["success"] = True
    except (CleanupError, OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
