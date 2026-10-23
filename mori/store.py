"""SQLite reference store; immutable unit versions and model-bound embeddings."""
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def vector(values):
    if not isinstance(values, (list, tuple)) or not values or any(type(x) not in (float, int) for x in values):
        raise ValueError("Invalid embedding values")
    try:
        values = [float(x) for x in values]
    except (OverflowError, ValueError):
        raise ValueError("Invalid embedding values") from None
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Invalid embedding values")
    scale = max(abs(x) for x in values)
    if not scale:
        raise ValueError("Zero embedding")
    # Scale first: squaring finite values can overflow or underflow to zero.
    scaled = [x / scale for x in values]
    norm = math.hypot(*scaled)
    return [x / norm for x in scaled]


class Store:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "corpus.sqlite"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,source_id TEXT NOT NULL,title TEXT NOT NULL,
              kind TEXT NOT NULL,locator TEXT NOT NULL,text TEXT NOT NULL,summary TEXT NOT NULL,
              metadata TEXT NOT NULL,hash TEXT NOT NULL,created TEXT NOT NULL,summary_vector TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,unit_id TEXT NOT NULL REFERENCES units(id),
              seq INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,text TEXT NOT NULL,vector TEXT NOT NULL,
              UNIQUE(unit_id,seq));
            CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT NOT NULL,created TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def fingerprint(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            return json.loads(row[0]) if row else None

    def add(self, units, chunks, fingerprint):
        """A whole import commits together; changed IDs require a new explicit version."""
        encoded = json.dumps(fingerprint, sort_keys=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            previous = json.loads(old[0]) if old else None
            if previous and {**previous, 'requested_dimensions': previous.get('requested_dimensions', 0)} != {**fingerprint, 'requested_dimensions': fingerprint.get('requested_dimensions', 0)}:
                raise ValueError("Embedding fingerprint mismatch; create a separate corpus and re-embed explicitly")
            db.execute("INSERT OR IGNORE INTO meta VALUES('embedding',?)", (encoded,))
            for unit in units:
                prior = db.execute("SELECT hash FROM units WHERE id=?", (unit['id'],)).fetchone()
                if prior:
                    if prior[0] != unit['hash']:
                        raise ValueError("Changed unit ID; preserve it and import a new version: " + unit['id'])
                    continue
                db.execute("INSERT INTO units VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                    unit['id'], unit['source_id'], unit['title'], unit['kind'], unit['locator'], unit['text'], unit['summary'],
                    json.dumps(unit['metadata'], ensure_ascii=False), unit['hash'], now(), json.dumps(unit['summary_vector'])))
                for row in chunks[unit['id']]:
                    db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", (row['id'], unit['id'], row['seq'], row['start'], row['end'], row['text'], json.dumps(row['vector'])))

    def source_catalog(self):
        """Count source groups without loading original text or embedding vectors."""
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT u.source_id AS id,
                  CASE WHEN count(DISTINCT u.title)=1 THEN min(u.title) ELSE u.source_id END AS label,
                  count(*) AS units, coalesce(sum(c.n),0) AS chunks
                FROM units AS u
                LEFT JOIN (SELECT unit_id,count(*) AS n FROM chunks GROUP BY unit_id) AS c ON c.unit_id=u.id
                GROUP BY u.source_id ORDER BY label,u.source_id
            """)]

    def select_sources(self, identifiers):
        return _SourceSelection(self, _resolve_sources(self.source_catalog(), identifiers))

    def run_roots(self):
        return [self.root]

    def _units(self, source_ids=None):
        where, values = _source_filter(source_ids, 'units')
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM units" + where + " ORDER BY id", values)]
        for row in rows:
            row['metadata'] = json.loads(row['metadata'])
            row['summary_vector'] = json.loads(row['summary_vector'])
        return rows

    def units(self):
        return self._units()

    def _chunks(self, source_ids=None):
        where, values = _source_filter(source_ids, 'u')
        with self.connect() as db:
            rows = [dict(r) for r in db.execute(
                "SELECT c.* FROM chunks AS c JOIN units AS u ON u.id=c.unit_id" + where + " ORDER BY c.unit_id,c.seq", values)]
        for row in rows:
            row['vector'] = json.loads(row['vector'])
        return rows

    def chunks(self):
        return self._chunks()

    def cache_get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def cache_put(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO cache VALUES(?,?,?)", (key, json.dumps(value, allow_nan=False), now()))

    def stats(self):
        return self._stats()

    def _stats(self, source_ids=None):
        where, values = _source_filter(source_ids, 'u')
        with self.connect() as db:
            return {"units": db.execute("SELECT count(*) FROM units AS u" + where, values).fetchone()[0],
                    "chunks": db.execute("SELECT count(*) FROM chunks AS c JOIN units AS u ON u.id=c.unit_id" + where, values).fetchone()[0],
                    "embedding": self.fingerprint()}


def _source_filter(identifiers, alias):
    if identifiers is None:
        return '', ()
    return f" WHERE {alias}.source_id IN ({','.join('?' for _ in identifiers)})", tuple(identifiers)


def _resolve_sources(catalog, identifiers):
    if (not isinstance(identifiers, list) or not identifiers
            or any(not isinstance(value, str) or not value.strip() for value in identifiers)):
        raise ValueError('Select at least one valid source ID')
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate source selection')
    lookup = {row['id']: row for row in catalog}
    if any(value not in lookup for value in identifiers):
        raise ValueError('Unknown source selection')
    return [dict(lookup[value]) for value in identifiers]


class _SourceSelection:
    """Read-only corpus view; successful provider ratings still share the cache."""
    def __init__(self, store, selection):
        self._store = store
        self.root = store.root
        self.selection = [dict(row) for row in selection]
        self._source_ids = tuple(row['id'] for row in selection)

    def units(self):
        return self._store._units(self._source_ids)

    def chunks(self):
        return self._store._chunks(self._source_ids)

    def stats(self):
        return self._store._stats(self._source_ids)

    def source_catalog(self):
        return [row for row in self._store.source_catalog() if row['id'] in self._source_ids]

    def select_sources(self, identifiers):
        return _SourceSelection(self._store, _resolve_sources(self.source_catalog(), identifiers))

    def fingerprint(self):
        return self._store.fingerprint()

    def cache_get(self, key):
        return self._store.cache_get(key)

    def cache_put(self, key, value):
        return self._store.cache_put(key, value)

    def run_roots(self):
        return self._store.run_roots()

    def add(self, *_args, **_kwargs):
        raise ValueError('Selected source views are read-only')


def validate_unit(row, limit):
    for key in ("id", "source_id", "title", "text", "summary"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Unit requires nonempty {key}")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", row['id']):
        raise ValueError("Unit IDs must be stable ASCII identifiers, at most 160 characters")
    if len(row['text']) > limit:
        raise ValueError("Unit too large; define an explicit chapter/section unit rather than silently truncating")
    result = {key: row[key] for key in ("id", "source_id", "title", "text", "summary")}
    result.update(kind=row.get('kind', 'document'), locator=row.get('locator', ''), metadata=row.get('metadata', {}))
    if not isinstance(result['metadata'], dict) or not all(isinstance(result[k], str) for k in ('kind', 'locator')):
        raise ValueError("Invalid unit metadata")
    try:
        # Metadata must survive storage without coercing keys/types or retaining
        # nonstandard NaN/Infinity values that cannot be rendered in run artifacts.
        encoded_metadata = json.dumps(result['metadata'], ensure_ascii=False, sort_keys=True, allow_nan=False)
        if json.loads(encoded_metadata) != result['metadata']:
            raise ValueError("Metadata changes when serialized")
    except (TypeError, ValueError, RecursionError):
        raise ValueError("Unit metadata must contain finite JSON values and string object keys") from None
    result['hash'] = digest(result)
    return result
