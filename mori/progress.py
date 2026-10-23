"""Thread-safe, source-free workflow snapshots for optional progress observers."""
from copy import deepcopy
from datetime import datetime, timezone
import threading


QUERY_STEPS = (
    ('prepare', 'Prepare sources and providers'),
    ('embed', 'Understand the question'),
    ('retrieve', 'Find matching passages'),
    ('rate_passages', 'Rate passage relevance'),
    ('rate_summaries', 'Rate source summaries'),
    ('rerank', 'Refine passage ranking'),
    ('assemble', 'Build the evidence packet'),
    ('answer', 'Write the answer'),
    ('validate', 'Check answer citations'),
    ('save', 'Save the inquiry'),
)
UPLOAD_STEPS = (
    ('prepare', 'Prepare the import'),
    ('read', 'Read source units'),
    ('import', 'Summarize and index source units'),
    ('save', 'Finish the import'),
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Progress:
    """Report completed work, never a predicted duration or provider payload.

    Each stage contributes equally; a counted running stage contributes its
    completed fraction. Only finish() can publish 100%. Observers receive an
    independent snapshot in update order and must not perform slow work.
    """
    def __init__(self, callback=None, steps=QUERY_STEPS):
        self._lock = threading.RLock()
        self._callback = callback
        started = timestamp()
        self._state = {
            'steps': [{'id': key, 'label': label, 'status': 'pending',
                       'completed': None, 'total': None, 'cached': 0, 'message': ''}
                      for key, label in steps],
            'percent': 0, 'current_stage': None, 'started_at': started,
            'updated_at': started, 'finished_at': None,
        }

    def snapshot(self):
        with self._lock:
            return deepcopy(self._state)

    def _step(self, stage):
        return next(row for row in self._state['steps'] if row['id'] == stage)

    def _publish(self):
        self._state['updated_at'] = timestamp()
        if not self._state['finished_at']:
            fraction = sum(1 if row['status'] in ('complete', 'skipped') else
                           row['completed'] / row['total'] if row['total'] else 0
                           for row in self._state['steps'])
            self._state['percent'] = min(99, int(100 * fraction / len(self._state['steps'])))
        if self._callback:
            self._callback(deepcopy(self._state))

    def start(self, stage, *, total=None, message=''):
        with self._lock:
            if self._state['finished_at']:
                return
            row = self._step(stage)
            row.update(status='running', completed=0 if total is not None else None,
                       total=total, cached=0, message=message)
            self._state['current_stage'] = stage
            self._publish()

    def advance(self, stage, *, cached=False):
        with self._lock:
            if self._state['finished_at']:
                return
            row = self._step(stage)
            if row['status'] != 'running' or row['total'] is None or row['completed'] >= row['total']:
                raise ValueError('Progress requires a running stage with unfinished counted work')
            row['completed'] += 1
            row['cached'] += int(bool(cached))
            self._publish()

    def complete(self, stage, *, message=''):
        with self._lock:
            if self._state['finished_at']:
                return
            row = self._step(stage)
            row.update(status='complete', completed=row['total'] if row['total'] is not None else 1,
                       total=row['total'] if row['total'] is not None else 1, message=message)
            self._state['current_stage'] = None
            self._publish()

    def skip(self, stage, message):
        with self._lock:
            if self._state['finished_at']:
                return
            self._step(stage).update(status='skipped', completed=0, total=0, message=message)
            self._publish()

    def fail(self):
        with self._lock:
            if self._state['finished_at']:
                return
            stage = self._state['current_stage']
            if stage is None:
                stage = next((row['id'] for row in self._state['steps'] if row['status'] == 'pending'), None)
            if stage:
                self._step(stage).update(status='failed', message='This step could not finish. Completed work is preserved.')
            self._state.update(current_stage=stage, finished_at=timestamp())
            self._publish()

    def finish(self):
        with self._lock:
            if self._state['finished_at']:
                return
            if any(row['status'] not in ('complete', 'skipped') for row in self._state['steps']):
                raise ValueError('Cannot finish a workflow with unfinished steps')
            self._state.update(percent=100, current_stage=None, finished_at=timestamp())
            self._publish()
