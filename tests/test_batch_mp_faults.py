"""Characterization spike (DL-5): does a forkserver Pool.imap_unordered raise when a
worker dies hard (os._exit) mid-map? The resume-safety design assumes it raises.
Each probe runs in a fresh subprocess with a hard timeout so a hang is bounded by
code, and so the forkserver singleton is never shared across probes."""

import os
import signal
import subprocess
import sys

import pytest

from scoutbot._batch_mp import BatchWork, run_batch

SENTINEL = '__DIE__'
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _maybe_die(item):
    # Module-level so forkserver workers import it by reference (tests.test_batch_mp_faults).
    if item == SENTINEL:
        os._exit(1)  # hard death: no exception, no cleanup
    return item


def _torch_in_worker(_):
    import sys

    return 'torch' in sys.modules


def _run_probe(code, timeout=30):
    """Run a forkserver probe in a fresh subprocess.

    Returns (completed, stdout). ``completed`` is False if the probe hung past
    ``timeout``; in that case the whole process group (forkserver + workers) is
    SIGKILLed so nothing is left running.
    """
    env = {
        **os.environ,
        'PYTHONPATH': _REPO_ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''),
    }
    proc = subprocess.Popen(
        [sys.executable, '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        out, _err = proc.communicate(timeout=timeout)
        return True, out.strip()
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return False, ''


_DEATH_PROBE = """
import multiprocessing
from tests.test_batch_mp_faults import _maybe_die, SENTINEL
ctx = multiprocessing.get_context('forkserver')
items = ['a', 'b', SENTINEL, 'c', 'd', 'e', 'f', 'g']
try:
    with ctx.Pool(processes=2) as pool:
        for _ in pool.imap_unordered(_maybe_die, items, chunksize=1):
            pass
    print('NO_RAISE')
except Exception as exc:
    print('RAISED:' + type(exc).__name__)
"""


@pytest.mark.slow
def test_imap_unordered_behavior_on_hard_worker_death():
    """Characterization result (darwin, forkserver): a hard worker death via
    ``os._exit(1)`` is NOT surfaced by ``imap_unordered``. The dead worker's
    task result never arrives, and the main process blocks forever waiting
    for it -- the probe subprocess only terminates because ``_run_probe``'s
    30s timeout kills the whole process group. Reproduced across repeated
    runs (not a one-off flake).

    Finding: HANG/SILENT -> Task 6 must implement the bounded-wait fallback
    (plain ``imap_unordered`` cannot be trusted to raise or return on a hard
    worker death on this platform).
    """
    completed, out = _run_probe(_DEATH_PROBE, timeout=30)
    assert not completed, (
        f'Expected the probe to hang past the timeout (previously-observed '
        f'HANG/SILENT finding), but it completed with stdout={out!r}. If '
        f'this platform/Python version now surfaces hard worker death '
        f'cleanly, update this test and the recorded finding accordingly.'
    )
    assert out == '', f'Hung probe unexpectedly produced output: {out!r}'


def _die_submit(remaining):
    # First image completes, then simulate the pool surfacing a hard death.
    yield ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])
    raise RuntimeError('pool worker died')


@pytest.mark.slow
def test_run_batch_fail_fast_flushes_partial(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    meta = {
        'mode': 'multiprocess',
        'start_method': 'forkserver',
        'processes': 2,
        'default_pid': 1,
        'base': {'name': 'batch_resumable'},
    }
    with pytest.raises(RuntimeError):
        run_batch(work, out, _die_submit, summary_meta=meta)
    assert out.read_text().strip() != ''  # completed result flushed before abort


_PRELOAD_PROBE = """
import multiprocessing
from tests.test_batch_mp_faults import _torch_in_worker
ctx = multiprocessing.get_context('forkserver')
ctx.set_forkserver_preload(['torch'])
with ctx.Pool(processes=1) as pool:
    print('TORCH_PRELOADED:' + str(pool.apply(_torch_in_worker, (None,))))
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='forkserver path is Linux/CUDA-first')
def test_forkserver_preload_puts_torch_in_workers():
    completed, out = _run_probe(_PRELOAD_PROBE, timeout=60)
    assert completed, 'preload probe hung'
    # With preload, torch is inherited from the server -> already in sys.modules at
    # worker start, not re-imported per worker.
    assert out == 'TORCH_PRELOADED:True', out
