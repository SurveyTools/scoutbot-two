# -*- coding: utf-8 -*-
"""Regression tests for import-time behaviour."""

import os
import subprocess
import sys


def test_import_scoutbot_does_not_require_numpy_at_module_import(tmp_path):
    stub_dir = tmp_path / 'stubs'
    stub_dir.mkdir()
    (stub_dir / 'numpy.py').write_text(
        "raise RuntimeError('numpy should not be imported during scoutbot import')\n"
    )

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(stub_dir), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, '-c', 'import scoutbot; print("ok")'],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'ok'
