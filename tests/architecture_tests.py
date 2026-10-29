#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only architecture boundary guards; no Qt/model/process/network execution.

These checks reject execution handles, mutable result maps and test friends in
MainWindow, without freezing line count or every private presentation field.
The CLI check is lexical inventory only, not executable smoke acceptance.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.MULTILINE)
CLI_BASELINE = frozenset({
    'smoke', 'smoke-input-root', 'smoke-cpu-runtime', 'smoke-pt',
    'smoke-model', 'smoke-conversion', 'smoke-evaluation',
    'smoke-dataset-session', 'smoke-preview-drop', 'smoke-model-library-drop',
    'preview-drop-request', 'evaluation-request', 'conversion-source',
    'conversion-format', 'display-model', 'device', 'gpu-index', 'screenshot'})


def sources(directory=SOURCE):
    return sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.cpp', '.h', '.hpp', '.cc'})


class ArchitectureTests(unittest.TestCase):
    def test_core_does_not_depend_on_ui_widgets_or_webengine(self):
        forbidden = {'QWidget', 'QMainWindow', 'QApplication', 'QDialog', 'QFileDialog'}
        violations = []
        for path in sources(SOURCE / 'core'):
            for include in INCLUDE.findall(path.read_text(encoding='utf-8')):
                name = include.replace('\\', '/')
                resolved = (path.parent / name).resolve()
                if (resolved.is_relative_to(SOURCE / 'ui') or name.startswith(('ui/', 'QtWidgets/', 'QtWebEngine')) or
                        name.rsplit('/', 1)[-1] in forbidden):
                    violations.append((path.relative_to(ROOT).as_posix(), name))
        self.assertEqual(violations, [], 'Core must not acquire a UI dependency')

    def test_application_layer_does_not_depend_on_widgets_or_pages(self):
        forbidden = {'QWidget', 'QMainWindow', 'QApplication', 'QDialog', 'QFileDialog', 'QMessageBox'}
        violations = []
        for path in sources(SOURCE / 'app'):
            for include in INCLUDE.findall(path.read_text(encoding='utf-8')):
                name = include.replace('\\', '/')
                if ((path.parent / name).resolve().is_relative_to(SOURCE / 'ui') or
                    name.startswith(('ui/', 'QtWidgets/', 'QtWebEngine')) or
                    name.rsplit('/', 1)[-1] in forbidden):
                    violations.append((path.relative_to(ROOT).as_posix(), name))
        self.assertEqual(violations, [], 'Application/services must remain compilable without the UI layer')

    def test_window_does_not_own_execution_resources_or_test_friends(self):
        header = (SOURCE / 'ui' / 'mainwindow.h').read_text(encoding='utf-8')
        # Services/controllers may be composed by the window. Execution handles
        # and independently mutable cache maps belong to those owners.
        declarations = re.findall(r'^\s*([^;{}]+);', header, re.MULTILINE)
        forbidden = re.compile(r'\b(?:InferenceWorker|EvaluationWorker|QProcess|QThread|QThreadPool)\s*[*&]|QHash<[^;]*(?:Cached|InferenceResult)')
        self.assertFalse([d for d in declarations if forbidden.search(d)])
        self.assertNotRegex(header, r'friend\s+class\s+\w*Tests\b')

    def test_production_does_not_import_test_only_dependencies(self):
        violations = []
        for path in sources():
            for include in INCLUDE.findall(path.read_text(encoding='utf-8')):
                name = include.replace('\\', '/')
                if ((path.parent / name).resolve().is_relative_to(ROOT / 'tests') or
                        name.startswith(('tests/', 'QtTest/')) or name.rsplit('/', 1)[-1] in {'QtTest', 'QTest', 'QSignalSpy'}):
                    violations.append((path.relative_to(ROOT).as_posix(), name))
        self.assertEqual(violations, [], 'Production may expose real smoke adapters without linking QtTest')

    def test_cpp_implementations_are_not_textually_included(self):
        violations = [(path.relative_to(ROOT).as_posix(), include)
            for path in sources() for include in INCLUDE.findall(path.read_text(encoding='utf-8'))
            if Path(include).suffix.lower() in {'.cpp', '.cc', '.cxx'}]
        self.assertEqual(violations, [], 'Compile implementation units through CMake rather than including .cpp files')

    def test_existing_cli_name_inventory_is_preserved_across_source_moves(self):
        # No file or class is prescribed: bootstrap/smoke code may move. A
        # lexical match only catches accidental removal; actual CLI/report
        # contracts still require the real application/package smoke gates.
        literals = set()
        for path in sources():
            literals.update(re.findall(r'"([A-Za-z][A-Za-z0-9-]*)"', path.read_text(encoding='utf-8')))
        self.assertFalse(CLI_BASELINE - literals, 'Missing established CLI names: ' + ', '.join(sorted(CLI_BASELINE - literals)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
