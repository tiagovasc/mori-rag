"""In-memory view of an inquiry while Jev rates it, for the local progress screen.

Progress snapshots stay source-free. This board is separate because it holds short
passage excerpts: it lives only in the server's memory for the current job and is
never written to a run folder, a log or a progress observer.
"""
import math
import re
import threading

STOPWORDS = frozenset(
    'a an and are as at be been but by can could did do does for from had has have how if in into is it its '
    'not no of on or over so than that the their them then there these they this those to was were what when '
    'where which while who why will with would about also more most other such'.split())
MAX_LEVELS = 10
RUN_ID = re.compile(r'[0-9]{8}T[0-9]{6}-[a-f0-9]{10}')


def short_label(text):
    """'Very strong: closely answers...' -> 'Very strong'."""
    return str(text).split(':', 1)[0].strip() or str(text).strip()


def _words(text):
    return {word for word in re.findall(r"[^\W\d_]{3,}", text.lower()) if word not in STOPWORDS}


def excerpt(text, question, limit=260):
    """The passage sentence sharing most words with the question, trimmed at a word boundary."""
    if not isinstance(text, str):
        return ''
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', ' '.join(text.split())) if part.strip()]
    if not sentences:
        return ''
    wanted = _words(question or '')
    best = max(range(len(sentences)), key=lambda i: (len(_words(sentences[i]) & wanted), -i))
    chosen = sentences[best]
    if len(chosen) < 120 and best + 1 < len(sentences) and len(chosen) + 1 + len(sentences[best + 1]) <= limit:
        chosen += ' ' + sentences[best + 1]
    if len(chosen) > limit:
        chosen = chosen[:limit].rsplit(' ', 1)[0].rstrip(',;:') + '…'
    return chosen


def level_of(rating, levels):
    """Map a rating to its rubric level, using the same expected score Mori ranks by."""
    if not isinstance(rating, dict):
        return None
    raw = rating.get('raw_score')
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
        value = raw
    else:
        score = rating.get('score')
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            return None
        value = score / 100 * (levels - 1)
    return max(0, min(levels - 1, int(math.floor(value + 0.5))))


class LiveBoard:
    """Thread-safe; every method is cheap and never raises into the pipeline."""

    def __init__(self, question, rubric, top=4):
        self._lock = threading.Lock()
        self._question = question if isinstance(question, str) else ''
        labels = [short_label(item) for item in (rubric or [])][:MAX_LEVELS]
        self._labels = labels if len(labels) >= 2 else ['Unrelated', 'Relevant']
        self._top = top
        self._passages = None
        self._summaries = None
        self._strongest = []
        self._notes = {}
        self._run = None

    def begin(self, stage, total):
        try:
            with self._lock:
                if stage == 'rate_passages':
                    self._passages = {'total': int(total), 'levels': [None] * int(total), 'rated': 0}
                elif stage == 'rate_summaries':
                    self._summaries = {'total': int(total), 'rated': 0, 'counts': [0] * len(self._labels)}
        except Exception:
            pass

    def rated(self, stage, index, row, unit=None):
        try:
            level = level_of(row.get('rating'), len(self._labels))
            with self._lock:
                if stage == 'rate_passages' and self._passages and 0 <= index < self._passages['total']:
                    if self._passages['levels'][index] is None:
                        self._passages['rated'] += 1
                    self._passages['levels'][index] = level
                    self._consider(row, unit, level)
                elif stage == 'rate_summaries' and self._summaries:
                    self._summaries['rated'] += 1
                    if level is not None:
                        self._summaries['counts'][level] += 1
        except Exception:
            pass

    def _consider(self, row, unit, level):
        rating = row.get('rating') or {}
        score = rating.get('score')
        if level is None or not isinstance(score, (int, float)):
            return
        key = (-float(score), -float(row.get('similarity') or 0), str(row.get('id')))
        if len(self._strongest) >= self._top and key >= self._strongest[-1][0]:
            return
        unit = unit if isinstance(unit, dict) else {}
        item = {'id': str(row.get('id')), 'level': level, 'label': self._labels[level],
                'score': round(float(score), 1), 'title': str(unit.get('title', '')),
                'locator': str(unit.get('locator', '')), 'source_id': str(unit.get('source_id', '')),
                'excerpt': excerpt(row.get('text', ''), self._question)}
        self._strongest = sorted([entry for entry in self._strongest if entry[1]['id'] != item['id']]
                                 + [(key, item)], key=lambda entry: entry[0])[:self._top]

    def note(self, key, value):
        try:
            if key in ('shortlist', 'evidence'):
                with self._lock:
                    self._notes[key] = value
            elif key == 'run' and isinstance(value, str) and RUN_ID.fullmatch(value):
                with self._lock:
                    self._run = value
        except Exception:
            pass

    def run_id(self):
        """The saved-inquiry folder this board belongs to, once the pipeline has created it."""
        with self._lock:
            return self._run

    def snapshot(self):
        with self._lock:
            passages = dict(self._passages, levels=list(self._passages['levels'])) if self._passages else None
            summaries = dict(self._summaries, counts=list(self._summaries['counts'])) if self._summaries else None
            return {'schema': 'mori.live.v1', 'labels': list(self._labels), 'passages': passages,
                    'summaries': summaries, 'strongest': [dict(entry[1]) for entry in self._strongest],
                    **{key: value for key, value in self._notes.items()}}
