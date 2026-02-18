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

    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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


def get_attendance_today_utc(db_path: Path) -> list[tuple[str, int]]:
    if not db_path.exists():
        return []
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    conn = sqlite3.connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """
            SELECT identity, COUNT(*) AS cnt
            FROM access_events
            WHERE decision = 'ALLOW'
              AND identity NOT IN ('unknown', 'no_face', 'admin_override')
              AND ts_utc >= ?
              AND ts_utc < ?
            GROUP BY identity
            ORDER BY cnt DESC, identity ASC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [(str(identity), int(cnt)) for identity, cnt in rows]
    finally:
        conn.close()
