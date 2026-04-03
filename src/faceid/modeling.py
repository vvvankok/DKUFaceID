from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainOutputs:
    classifier: LogisticRegression
    label_encoder: LabelEncoder
    metrics: Dict[str, float]
    report: str


@dataclass(frozen=True)
class IdentityRecord:
    name: str
    embedding: np.ndarray


@dataclass(frozen=True)
class AccessEvent:
    decision: str
    reason: str
    identity: str
    score: float
    liveness_ok: bool
    source: str = "camera"
    is_suspicious: bool = False


def train_classifier(
    embeddings: np.ndarray,
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainOutputs:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 unique identities to train classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        multi_class="auto",
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return TrainOutputs(
        classifier=clf,
        label_encoder=label_encoder,
        metrics=metrics,
        report=report,
    )


def save_artifact(output_path: Path, outputs: TrainOutputs) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": outputs.classifier,
        "label_encoder": outputs.label_encoder,
        "metrics": outputs.metrics,
    }
    joblib.dump(payload, output_path)


def save_embeddings_sqlite(
    db_path: Path, embeddings: np.ndarray, labels: Sequence[str]
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    per_label = {}
    for vector, label in zip(embeddings, labels):
        per_label.setdefault(label, []).append(vector)

    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        conn.execute("DELETE FROM identities")
        for name, vectors in sorted(per_label.items()):
            centroid = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
            centroid = normalize_vector(centroid)
            conn.execute(
                """
                INSERT INTO identities (name, embedding, dim, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (name, centroid.tobytes(), int(centroid.shape[0])),
            )
        conn.commit()
    finally:
        conn.close()


def ensure_identities_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def upsert_identity_embedding(db_path: Path, name: str, embedding: np.ndarray) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector = normalize_vector(embedding.astype(np.float32))
    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        ensure_access_events_schema(conn)
        conn.execute(
            """
            INSERT INTO identities (name, embedding, dim, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                embedding = excluded.embedding,
                dim = excluded.dim,
                updated_at = CURRENT_TIMESTAMP
            """,
            (name, vector.tobytes(), int(vector.shape[0])),
        )
        conn.commit()
    finally:
        conn.close()


def load_identities(db_path: Path) -> list[IdentityRecord]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        ensure_access_events_schema(conn)
        rows = conn.execute(
            "SELECT name, embedding, dim FROM identities ORDER BY name ASC"
        ).fetchall()
    finally:
        conn.close()

    records: list[IdentityRecord] = []
    for name, embedding_blob, dim in rows:
        vector = np.frombuffer(embedding_blob, dtype=np.float32)
        if vector.shape[0] != int(dim):
            continue
        records.append(IdentityRecord(name=str(name), embedding=normalize_vector(vector)))
    return records


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_vector(a.astype(np.float32))
    b_norm = normalize_vector(b.astype(np.float32))
    return float(np.dot(a_norm, b_norm))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for better concurrent read/write."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_access_events_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS access_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            identity TEXT NOT NULL,
            score REAL NOT NULL,
            liveness_ok INTEGER NOT NULL,
            source TEXT NOT NULL,
            is_suspicious INTEGER NOT NULL DEFAULT 0,
            is_false_positive INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _ensure_column(conn, "access_events", "is_suspicious", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "access_events", "is_false_positive", "INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_access_events_ts
        ON access_events(ts_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_access_events_decision
        ON access_events(decision)
        """
    )


def log_access_event(db_path: Path, event: AccessEvent) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        ensure_access_events_schema(conn)
        conn.execute(
            """
            INSERT INTO access_events (
                ts_utc, decision, reason, identity, score, liveness_ok, source, is_suspicious
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                event.decision,
                event.reason,
                event.identity,
                float(event.score),
                1 if event.liveness_ok else 0,
                event.source,
                1 if event.is_suspicious else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def cleanup_old_events(db_path: Path, ttl_days: int = 7) -> int:
    if ttl_days <= 0:
        return 0
    if not db_path.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        cur = conn.execute(
            "DELETE FROM access_events WHERE ts_utc < ?",
            (cutoff.isoformat(),),
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()
        return deleted
    finally:
        conn.close()


def mark_recent_false_incidents(
    db_path: Path,
    identity: str,
    lookback_sec: int = 20,
    max_rows: int = 30,
) -> int:
    if not db_path.exists():
        return 0
    if not identity or identity == "unknown":
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=lookback_sec)
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        cur = conn.execute(
            """
            UPDATE access_events
            SET is_false_positive = 1
            WHERE id IN (
                SELECT id
                FROM access_events
                WHERE ts_utc >= ?
                  AND decision IN ('DENY_UNKNOWN', 'DENY_LIVENESS')
                  AND is_false_positive = 0
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (cutoff.isoformat(), int(max_rows)),
        )
        updated = int(cur.rowcount or 0)
        conn.commit()
        return updated
    finally:
        conn.close()


def _ensure_column(
    conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    columns = {row[1] for row in rows}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")


def get_attendance_today_utc(db_path: Path) -> list[tuple[str, str]]:
    """
    Return today's unique visitors as ``(identity, first_entry_time_utc)`` pairs,
    ordered by arrival time ascending.  Each person appears at most once per
    calendar day regardless of how many ALLOW events were logged.
    The time string is formatted as ``"HH:MM"`` in UTC.
    """
    if not db_path.exists():
        return []
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """
            SELECT identity, MIN(ts_utc) AS first_entry
            FROM access_events
            WHERE decision = 'ALLOW'
              AND identity NOT IN ('unknown', 'no_face', 'admin_override')
              AND ts_utc >= ?
              AND ts_utc < ?
            GROUP BY identity
            ORDER BY first_entry ASC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()

    result: list[tuple[str, str]] = []
    for identity, first_entry in rows:
        try:
            # ts_utc stored as ISO-8601; extract HH:MM
            time_str = str(first_entry)[11:16]
        except Exception:
            time_str = "—"
        result.append((str(identity), time_str))
    return result


def backup_db(db_path: Path, backup_path: Path) -> None:
    """
    Create a plain (unencrypted) SQLite backup of *db_path* using the
    built-in streaming backup API.  Safe to call while the DB is in use.
    Satisfies ТЗ §4.1.4 alt-flow "восстановление из резервной копии".
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Source DB not found: {db_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    src = _connect(db_path)
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


def export_events_csv(db_path: Path, csv_path: Path, limit: int = 10_000) -> int:
    """
    Export the most recent *limit* access events to a UTF-8 CSV file at
    *csv_path*.  Returns the number of rows written.
    Satisfies ТЗ §4.1.6 (ведение журнала событий).
    """
    import csv

    if not db_path.exists():
        return 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """
            SELECT ts_utc, decision, reason, identity, score,
                   liveness_ok, source, is_suspicious, is_false_positive
            FROM access_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    headers = [
        "ts_utc", "decision", "reason", "identity", "score",
        "liveness_ok", "source", "is_suspicious", "is_false_positive",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return len(rows)
