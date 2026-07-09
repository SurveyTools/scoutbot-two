"""Characterization spike (DL-5): does a forkserver Pool.imap_unordered raise when a
worker dies hard (os._exit) mid-map? The resume-safety design assumes it raises.
Each probe runs in a fresh subprocess with a hard timeout so a hang is bounded by
code, and so the forkserver singleton is never shared across probes."""

import os
import signal
import subprocess
import sys

import pytest

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
