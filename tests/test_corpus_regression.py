"""Replay frozen CPU baselines from a scoutbot test corpus and assert inference has not drifted.

Slow + opt-in: skipped unless SCOUTBOT_TEST_CORPUS points at a corpus directory, so public CI
(which has no corpus) is unaffected. See docs/plans/2026-07-31-scoutbot-test-corpus-builder.md.
"""

import json
import os
from pathlib import Path

import build_test_corpus as btc
import pytest

import scoutbot
from scoutbot import model_store
from scoutbot.registry.paths import get_models_dir

_CORPUS = os.getenv('SCOUTBOT_TEST_CORPUS')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _CORPUS, reason='SCOUTBOT_TEST_CORPUS not set'),
]


def _load(corpus: Path):
    manifest = json.loads((corpus / 'manifest.json').read_text())
    baselines = {
        b['name']: b
        for b in (
            json.loads(line)
            for line in (corpus / 'baselines.jsonl').read_text().splitlines()
            if line.strip()
        )
    }
    return manifest, baselines


def test_corpus_baselines_do_not_drift():
    corpus = Path(_CORPUS)
    manifest, baselines = _load(corpus)
    config = manifest['model']['config']

    model_path = get_models_dir() / manifest['model']['name']
    if not model_path.exists():
        pytest.skip(
            f'model {model_path} absent; run: uv run python -m scoutbot fetch --config {config}'
        )
    current_md5 = model_store._md5(model_path)
    current_runtime = btc.collect_runtime(Path(__file__).resolve().parents[1] / 'uv.lock')

    ok, reason = btc.stack_matches(manifest, current_runtime, current_md5)
    if not ok:
        pytest.skip(
            f'baseline frozen under a different stack ({reason}); re-run the builder to re-freeze'
        )

    # run_config must still match what was frozen (surfaces a CONFIGS change directly).
    assert btc.run_config_for(config) == manifest['run_config']

    rc = manifest['run_config']
    for name, expected in baselines.items():
        wic, detections = scoutbot.pipeline(
            str(corpus / 'images' / name),
            config=config,
            backend_device='cpu',
            loc_thresh=rc['loc_thresh'],
            slice_height=rc['slice_height'],
            slice_width=rc['slice_width'],
            overlap_height_ratio=rc['overlap_height_ratio'],
            overlap_width_ratio=rc['overlap_width_ratio'],
            perform_standard_pred=rc['perform_standard_pred'],
            postprocess_class_agnostic=rc['postprocess_class_agnostic'],
        )
        assert wic == expected['wic'], f'{name}: wic drifted'
        assert detections == expected['detections'], f'{name}: detections drifted'
