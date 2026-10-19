#!/usr/bin/env python3
"""Destructive cleanup tests are strictly confined to fresh /tmp test fixtures."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('old_cleanup', Path(__file__).with_name('cleanup_old_2_0.py'))
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class ExactCleanupTests(unittest.TestCase):
    def test_dotted_paths_and_symlinks_are_removed_without_following(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            base = Path(tmp)
            root = base / 'old-build'
            outside = base / 'outside'
            outside.mkdir()
            protected = outside / 'keep.bin'
            protected.write_bytes(b'outside target must survive')
            for relative in ('runtime/lib/python3.10/pkg/file.py', '.qt/config.cmake', 'plain.txt'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
            (root / 'file-link').symlink_to(protected)
            (root / 'directory-link').symlink_to(outside, target_is_directory=True)
            snapshot = cleanup.scan(root)
            self.assertEqual(snapshot['files']['directory-link']['kind'], 'symlink')
            cleanup.erase_exact(root, snapshot)
            self.assertFalse(root.exists())
            self.assertEqual(protected.read_bytes(), b'outside target must survive')

    def test_extra_file_blocks_before_any_deletion(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            root = Path(tmp) / 'old'
            root.mkdir()
            (root / 'original').write_text('original')
            snapshot = cleanup.scan(root)
            (root / 'extra').write_text('unexpected')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected entry'):
                cleanup.erase_exact(root, snapshot)
            self.assertEqual((root / 'original').read_text(), 'original')
            self.assertEqual((root / 'extra').read_text(), 'unexpected')

    def test_root_inode_replacement_is_blocked(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            base = Path(tmp)
            root = base / 'old'
            root.mkdir()
            (root / 'original').write_text('original')
            snapshot = cleanup.scan(root)
            root.rename(base / 'stashed')
            root.mkdir()
            (root / 'new-user-file').write_text('must remain')
            with self.assertRaisesRegex(RuntimeError, 'Root directory identity changed'):
                cleanup.erase_exact(root, snapshot)
            self.assertEqual((root / 'new-user-file').read_text(), 'must remain')
            self.assertEqual((base / 'stashed/original').read_text(), 'original')

    def test_child_inode_replacement_is_blocked(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            base = Path(tmp)
            root = base / 'old'
            child = root / 'python3.10'
            child.mkdir(parents=True)
            (child / 'old-file').write_text('old')
            snapshot = cleanup.scan(root)
            child.rename(base / 'stashed')
            child.mkdir()
            (child / 'new-file').write_text('new')
            with self.assertRaisesRegex(RuntimeError, 'Directory identity changed'):
                cleanup.erase_exact(root, snapshot)
            self.assertEqual((child / 'new-file').read_text(), 'new')
            self.assertEqual((base / 'stashed/old-file').read_text(), 'old')

    def test_directory_swap_to_symlink_is_blocked(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            base = Path(tmp)
            root = base / 'old'
            child = root / '.qt'
            child.mkdir(parents=True)
            (child / 'original').write_text('original')
            outside = base / 'outside'
            outside.mkdir()
            (outside / 'user-file').write_text('must remain')
            snapshot = cleanup.scan(root)
            child.rename(base / 'stashed')
            child.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'Directory identity changed'):
                cleanup.erase_exact(root, snapshot)
            self.assertTrue(child.is_symlink())
            self.assertEqual((outside / 'user-file').read_text(), 'must remain')

    def test_stable_identity_allows_only_external_hardlink_count_changes(self):
        with tempfile.TemporaryDirectory(prefix='vision220-cleanup-selftest-', dir='/tmp') as tmp:
            root = Path(tmp) / 'old'
            root.mkdir()
            (root / 'file').write_text('stable')
            original = cleanup.scan(root)
            changed = copy.deepcopy(original)
            changed['files']['file']['nlink'] += 1
            changed['files']['file']['blocks'] += 8
            self.assertEqual(cleanup.stable(original), cleanup.stable(changed))
            changed['files']['file']['sha256'] = 'f' * 64
            self.assertNotEqual(cleanup.stable(original), cleanup.stable(changed))


if __name__ == '__main__':
    unittest.main(verbosity=2)
