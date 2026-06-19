import os
import sqlite3
from datetime import datetime, timezone
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
  id          INTEGER PRIMARY KEY,
  name        TEXT UNIQUE NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  id          INTEGER PRIMARY KEY,
  person_id   INTEGER NOT NULL REFERENCES people(id),
  vec         BLOB NOT NULL,
  source_path TEXT,
  det_score   REAL,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_files (
  sha256       TEXT PRIMARY KEY,
  path         TEXT,
  result       TEXT,
  processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_embeddings (
  id          INTEGER PRIMARY KEY,
  group_id    TEXT NOT NULL,
  vec         BLOB NOT NULL,
  source_path TEXT,
  det_score   REAL
);
"""


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype("<f4").tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def load_known_embeddings(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT e.vec, e.person_id, p.name FROM embeddings e JOIN people p ON e.person_id = p.id"
    ).fetchall()
    if not rows:
        return np.zeros((0, 512), dtype="<f4"), [], {}
    vecs = np.stack([blob_to_vec(r[0]) for r in rows])
    person_ids = [r[1] for r in rows]
    person_names = {r[1]: r[2] for r in rows}
    return vecs, person_ids, person_names


def file_already_processed(conn: sqlite3.Connection, sha256: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM processed_files WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return row is not None


def record_processed_file(conn: sqlite3.Connection, sha256: str, path: str, result: str):
    conn.execute(
        "INSERT OR REPLACE INTO processed_files (sha256, path, result, processed_at) VALUES (?,?,?,?)",
        (sha256, path, result, now_iso()),
    )


def get_person_by_name(conn: sqlite3.Connection, name: str):
    return conn.execute("SELECT id, name FROM people WHERE name = ?", (name,)).fetchone()


def get_person_by_id(conn: sqlite3.Connection, person_id: int):
    return conn.execute("SELECT id, name FROM people WHERE id = ?", (person_id,)).fetchone()


def delete_person(conn: sqlite3.Connection, person_id: int):
    conn.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))


def delete_processed_files_for_person(conn: sqlite3.Connection, name: str):
    conn.execute("DELETE FROM processed_files WHERE result = ?", (f"person:{name}",))


def get_or_create_person(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM people WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO people (name, created_at) VALUES (?,?)", (name, now_iso())
    )
    return cur.lastrowid


def add_embedding(conn: sqlite3.Connection, person_id: int, vec: np.ndarray,
                  source_path: str, det_score: float):
    conn.execute(
        "INSERT INTO embeddings (person_id, vec, source_path, det_score, created_at) VALUES (?,?,?,?,?)",
        (person_id, vec_to_blob(vec), source_path, det_score, now_iso()),
    )


def count_embeddings_for_person(conn: sqlite3.Connection, person_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
    ).fetchone()[0]


def get_lowest_det_score_embedding(conn: sqlite3.Connection, person_id: int):
    return conn.execute(
        "SELECT id, det_score FROM embeddings WHERE person_id = ? ORDER BY det_score ASC LIMIT 1",
        (person_id,),
    ).fetchone()


def delete_embedding(conn: sqlite3.Connection, emb_id: int):
    conn.execute("DELETE FROM embeddings WHERE id = ?", (emb_id,))


def insert_pending_embedding(conn: sqlite3.Connection, group_id: str, vec: np.ndarray,
                              source_path: str, det_score: float):
    conn.execute(
        "INSERT INTO pending_embeddings (group_id, vec, source_path, det_score) VALUES (?,?,?,?)",
        (group_id, vec_to_blob(vec), source_path, det_score),
    )


def get_pending_embeddings(conn: sqlite3.Connection, group_id: str):
    return conn.execute(
        "SELECT id, vec, source_path, det_score FROM pending_embeddings WHERE group_id = ?",
        (group_id,),
    ).fetchall()


def get_pending_embeddings_by_filenames(conn: sqlite3.Connection, filenames: list[str]):
    """Pick up embeddings from photos that were manually moved between review groups."""
    if not filenames:
        return []
    fname_set = set(filenames)
    return [r for r in conn.execute(
        "SELECT id, vec, source_path, det_score FROM pending_embeddings"
    ).fetchall() if os.path.basename(r[2]) in fname_set]


def delete_pending_embeddings(conn: sqlite3.Connection, group_id: str):
    conn.execute("DELETE FROM pending_embeddings WHERE group_id = ?", (group_id,))


def delete_pending_embeddings_by_ids(conn: sqlite3.Connection, ids: list[int]):
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM pending_embeddings WHERE id IN ({placeholders})", ids)
