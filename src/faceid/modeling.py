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
    embedding: np.ndarray          # centroid (fast path)
    all_embeddings: tuple = ()     # individual embeddings for higher accuracy
    photo_path: str = ""           # path to face photo


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
    _ensure_column(conn, "identities", "photo_path", "TEXT NOT NULL DEFAULT ''")


def ensure_identity_embeddings_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_identity_embeddings_name
        ON identity_embeddings(name)
        """
    )


def save_individual_embeddings(
    db_path: Path, name: str, embeddings: list[np.ndarray]
) -> None:
    """Store per-sample embeddings for a person (used in multi-embedding matching)."""
    if not embeddings:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_identity_embeddings_schema(conn)
        conn.execute("DELETE FROM identity_embeddings WHERE name = ?", (name,))
        for vec in embeddings:
            v = normalize_vector(vec.astype(np.float32))
            conn.execute(
                "INSERT INTO identity_embeddings (name, embedding) VALUES (?, ?)",
                (name, v.tobytes()),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_identity_photo(db_path: Path, name: str, photo_path: str) -> None:
    """Store the path to the best face crop for a registered person."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        conn.execute(
            "UPDATE identities SET photo_path = ? WHERE name = ?",
            (photo_path, name),
        )
        conn.commit()
    finally:
        conn.close()


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
        ensure_identity_embeddings_schema(conn)
        rows = conn.execute(
            "SELECT name, embedding, dim, photo_path FROM identities ORDER BY name ASC"
        ).fetchall()
        # Load individual embeddings grouped by name
        ind_rows = conn.execute(
            "SELECT name, embedding FROM identity_embeddings ORDER BY name, id ASC"
        ).fetchall()
    finally:
        conn.close()

    ind_by_name: dict[str, list[np.ndarray]] = {}
    for iname, iblob in ind_rows:
        vec = np.frombuffer(iblob, dtype=np.float32).copy()
        ind_by_name.setdefault(str(iname), []).append(normalize_vector(vec))

    records: list[IdentityRecord] = []
    for name, embedding_blob, dim, photo_path in rows:
        vector = np.frombuffer(embedding_blob, dtype=np.float32)
        if vector.shape[0] != int(dim):
            continue
        ind = tuple(ind_by_name.get(str(name), []))
        records.append(IdentityRecord(
            name=str(name),
            embedding=normalize_vector(vector),
            all_embeddings=ind,
            photo_path=str(photo_path) if photo_path else "",
        ))
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
    _ensure_column(conn, "access_events", "chain_hash", "TEXT")
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


def _compute_chain_hash(prev_hash: str, ts_utc: str, decision: str,
                         identity: str, score: float) -> str:
    import hashlib
    payload = f"{prev_hash}|{ts_utc}|{decision}|{identity}|{score:.6f}"
    return hashlib.sha256(payload.encode()).hexdigest()


def log_access_event(db_path: Path, event: AccessEvent) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_identities_schema(conn)
        ensure_access_events_schema(conn)
        # Chain integrity: compute hash chained from previous row
        prev_row = conn.execute(
            "SELECT chain_hash FROM access_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = (prev_row[0] or "GENESIS") if prev_row else "GENESIS"
        ts_now = datetime.now(timezone.utc).isoformat()
        chain_hash = _compute_chain_hash(
            prev_hash, ts_now, event.decision, event.identity, float(event.score)
        )
        conn.execute(
            """
            INSERT INTO access_events (
                ts_utc, decision, reason, identity, score, liveness_ok, source,
                is_suspicious, chain_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_now,
                event.decision,
                event.reason,
                event.identity,
                float(event.score),
                1 if event.liveness_ok else 0,
                event.source,
                1 if event.is_suspicious else 0,
                chain_hash,
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


def export_attendance_report(
    db_path: Path,
    start_date: str,
    end_date: str,
    csv_path: Path,
) -> int:
    """
    Export attendance (ALLOW events) for a date range to CSV.
    *start_date* / *end_date* are ISO date strings ``"YYYY-MM-DD"`` (inclusive).
    Returns number of rows written.
    """
    import csv

    if not db_path.exists():
        return 0
    try:
        start_dt = datetime.fromisoformat(start_date).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        end_dt = datetime.fromisoformat(end_date).replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
        )
    except ValueError:
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """
            SELECT identity, MIN(ts_utc) AS first_entry, COUNT(*) AS event_count,
                   DATE(ts_utc) AS day
            FROM access_events
            WHERE decision = 'ALLOW'
              AND identity NOT IN ('unknown', 'no_face', 'admin_override')
              AND ts_utc >= ?
              AND ts_utc <= ?
            GROUP BY identity, DATE(ts_utc)
            ORDER BY day ASC, first_entry ASC
            """,
            (start_dt.isoformat(), end_dt.isoformat()),
        ).fetchall()
    finally:
        conn.close()

    headers = ["date", "identity", "first_entry_utc", "entry_count"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for identity, first_entry, event_count, day in rows:
            time_str = str(first_entry)[11:16] if first_entry else ""
            writer.writerow([day, identity, time_str, event_count])
    return len(rows)


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


# ── HASH-CHAIN INTEGRITY ──────────────────────────────────────────────────────

def verify_chain_integrity(db_path: Path) -> tuple[bool, int, int]:
    """
    Walk all access_events in insertion order and verify the SHA-256 chain.
    Returns (ok, verified_count, first_broken_id).  first_broken_id == -1 if intact.
    """
    if not db_path.exists():
        return True, 0, -1
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, ts_utc, decision, identity, score, chain_hash "
            "FROM access_events ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()

    prev_hash = "GENESIS"
    verified = 0
    for row_id, ts_utc, decision, identity, score, chain_hash in rows:
        if not chain_hash:
            prev_hash = "GENESIS"
            verified += 1
            continue
        expected = _compute_chain_hash(prev_hash, ts_utc, decision, identity, float(score or 0))
        if expected != chain_hash:
            return False, verified, int(row_id)
        prev_hash = chain_hash
        verified += 1
    return True, verified, -1


# ── ACCESS SCHEDULES ──────────────────────────────────────────────────────────

def ensure_schedules_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_schedules (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            weekday   INTEGER NOT NULL,
            start_hour INTEGER NOT NULL,
            start_min  INTEGER NOT NULL DEFAULT 0,
            end_hour   INTEGER NOT NULL,
            end_min    INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_schedules_name ON identity_schedules(name)"
    )


def upsert_schedule(db_path: Path, name: str, entries: list[dict]) -> None:
    """Replace all schedule rows for *name*. Pass entries=[] to remove restrictions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        ensure_schedules_schema(conn)
        conn.execute("DELETE FROM identity_schedules WHERE name = ?", (name,))
        for e in entries:
            conn.execute(
                """INSERT INTO identity_schedules
                   (name, weekday, start_hour, start_min, end_hour, end_min)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, int(e["weekday"]), int(e["start_hour"]),
                 int(e.get("start_min", 0)), int(e["end_hour"]),
                 int(e.get("end_min", 0))),
            )
        conn.commit()
    finally:
        conn.close()


def get_schedule(db_path: Path, name: str) -> list[dict]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        ensure_schedules_schema(conn)
        rows = conn.execute(
            "SELECT weekday, start_hour, start_min, end_hour, end_min "
            "FROM identity_schedules WHERE name = ? ORDER BY weekday",
            (name,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"weekday": r[0], "start_hour": r[1], "start_min": r[2],
         "end_hour": r[3], "end_min": r[4]}
        for r in rows
    ]


def check_schedule_allowed(db_path: Path, name: str) -> tuple[bool, str]:
    """Returns (allowed, reason). No schedule entries → always allowed."""
    entries = get_schedule(db_path, name)
    if not entries:
        return True, "no_schedule"
    now = datetime.now()
    weekday = now.weekday()
    now_min = now.hour * 60 + now.minute
    for e in entries:
        if e["weekday"] != weekday:
            continue
        start = e["start_hour"] * 60 + e["start_min"]
        end = e["end_hour"] * 60 + e["end_min"]
        if start <= now_min < end:
            return True, "schedule_ok"
    return False, "outside_schedule"


def get_all_schedules(db_path: Path) -> dict[str, list[dict]]:
    if not db_path.exists():
        return {}
    conn = _connect(db_path)
    try:
        ensure_schedules_schema(conn)
        rows = conn.execute(
            "SELECT name, weekday, start_hour, start_min, end_hour, end_min "
            "FROM identity_schedules ORDER BY name, weekday"
        ).fetchall()
    finally:
        conn.close()
    result: dict[str, list[dict]] = {}
    for name, weekday, sh, sm, eh, em in rows:
        result.setdefault(str(name), []).append(
            {"weekday": weekday, "start_hour": sh, "start_min": sm,
             "end_hour": eh, "end_min": em}
        )
    return result


# ── UNKNOWN FACE CLUSTERING ───────────────────────────────────────────────────

def ensure_unknown_visits_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS unknown_visits (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc     TEXT    NOT NULL,
            embedding  BLOB    NOT NULL,
            cluster_id INTEGER DEFAULT -1
        )
        """
    )


def ensure_unknown_clusters_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS unknown_clusters (
            cluster_id  INTEGER PRIMARY KEY,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            visit_count INTEGER NOT NULL DEFAULT 0,
            centroid    BLOB NOT NULL
        )
        """
    )


def log_unknown_visit(db_path: Path, embedding: np.ndarray) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    v = normalize_vector(embedding.astype(np.float32))
    conn = _connect(db_path)
    try:
        ensure_unknown_visits_schema(conn)
        conn.execute(
            "INSERT INTO unknown_visits (ts_utc, embedding) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), v.tobytes()),
        )
        conn.commit()
    finally:
        conn.close()


def recluster_unknowns(db_path: Path, min_samples: int = 3,
                        eps: float = 0.30) -> int:
    """Run DBSCAN on unknown visit embeddings. Returns number of clusters."""
    if not db_path.exists():
        return 0
    conn = _connect(db_path)
    try:
        ensure_unknown_visits_schema(conn)
        ensure_unknown_clusters_schema(conn)
        rows = conn.execute(
            "SELECT id, ts_utc, embedding FROM unknown_visits ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < min_samples:
        return 0

    from sklearn.cluster import DBSCAN

    ids = [r[0] for r in rows]
    timestamps = [r[1] for r in rows]
    embeddings = np.stack(
        [normalize_vector(np.frombuffer(r[2], dtype=np.float32).copy()) for r in rows],
        axis=0,
    )

    labels = DBSCAN(eps=eps, min_samples=min_samples,
                    metric="cosine", algorithm="brute").fit_predict(embeddings)

    conn = _connect(db_path)
    try:
        ensure_unknown_visits_schema(conn)
        ensure_unknown_clusters_schema(conn)
        for visit_id, label in zip(ids, labels):
            conn.execute(
                "UPDATE unknown_visits SET cluster_id = ? WHERE id = ?",
                (int(label), visit_id),
            )
        conn.execute("DELETE FROM unknown_clusters")
        unique_labels = set(labels) - {-1}
        for cid in unique_labels:
            mask = labels == cid
            cluster_embs = embeddings[mask]
            cluster_ts = [timestamps[i] for i, m in enumerate(mask) if m]
            centroid = normalize_vector(cluster_embs.mean(axis=0))
            conn.execute(
                """INSERT OR REPLACE INTO unknown_clusters
                   (cluster_id, first_seen, last_seen, visit_count, centroid)
                   VALUES (?, ?, ?, ?, ?)""",
                (int(cid), min(cluster_ts), max(cluster_ts),
                 int(mask.sum()), centroid.tobytes()),
            )
        conn.commit()
        return len(unique_labels)
    finally:
        conn.close()


def get_unknown_clusters(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        ensure_unknown_clusters_schema(conn)
        rows = conn.execute(
            "SELECT cluster_id, first_seen, last_seen, visit_count "
            "FROM unknown_clusters ORDER BY visit_count DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"cluster_id": r[0], "first_seen": str(r[1])[:16],
         "last_seen": str(r[2])[:16], "visit_count": r[3]}
        for r in rows
    ]


# ── ANALYTICS QUERIES ─────────────────────────────────────────────────────────

def get_hourly_heatmap(db_path: Path, days: int = 30) -> np.ndarray:
    """Returns (7, 24) int32 array: ALLOW count by weekday × hour (UTC)."""
    matrix = np.zeros((7, 24), dtype=np.int32)
    if not db_path.exists():
        return matrix
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """SELECT ts_utc FROM access_events
               WHERE decision = 'ALLOW'
                 AND identity NOT IN ('unknown', 'no_face', 'admin_override')
                 AND ts_utc >= ?""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    for (ts_utc,) in rows:
        try:
            dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
            matrix[dt.weekday(), dt.hour] += 1
        except Exception:
            pass
    return matrix


def get_daily_attendance(db_path: Path, days: int = 30) -> list[tuple[str, int]]:
    """Returns (date_str, unique_visitor_count) for last *days* days, oldest first."""
    if not db_path.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _connect(db_path)
    try:
        ensure_access_events_schema(conn)
        rows = conn.execute(
            """SELECT DATE(ts_utc) as day, COUNT(DISTINCT identity) as cnt
               FROM access_events
               WHERE decision = 'ALLOW'
                 AND identity NOT IN ('unknown', 'no_face', 'admin_override')
                 AND ts_utc >= ?
               GROUP BY day ORDER BY day ASC""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    return [(str(r[0]), int(r[1])) for r in rows]


# ── PDF REPORT ────────────────────────────────────────────────────────────────

def _get_pdf_font(pdf) -> str:
    """Add a Unicode TTF font to *pdf* and return its name, or '' for fallback."""
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                pdf.add_font("UniFont", "", fp)
                return "UniFont"
            except Exception:
                continue
    return ""


def export_pdf_report(
    db_path: Path,
    start_date: str,
    end_date: str,
    pdf_path: Path,
    org_name: str = "Казахстанско-Немецкий Университет",
) -> int:
    """Export attendance report as PDF. Returns number of data rows."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 не установлен. Выполните: pip install fpdf2")

    # Fetch data
    rows_data: list[tuple] = []
    try:
        start_dt = datetime.fromisoformat(start_date).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Неверный формат даты: {start_date} / {end_date}")

    if db_path.exists():
        conn = _connect(db_path)
        try:
            ensure_access_events_schema(conn)
            rows_data = conn.execute(
                """SELECT identity, MIN(ts_utc), COUNT(*), DATE(ts_utc)
                   FROM access_events
                   WHERE decision = 'ALLOW'
                     AND identity NOT IN ('unknown', 'no_face', 'admin_override')
                     AND ts_utc >= ? AND ts_utc <= ?
                   GROUP BY identity, DATE(ts_utc)
                   ORDER BY DATE(ts_utc), MIN(ts_utc)""",
                (start_dt.isoformat(), end_dt.isoformat()),
            ).fetchall()
            stats = conn.execute(
                """SELECT SUM(decision='ALLOW'), SUM(decision LIKE 'DENY%'), SUM(is_suspicious)
                   FROM access_events WHERE ts_utc >= ? AND ts_utc <= ?""",
                (start_dt.isoformat(), end_dt.isoformat()),
            ).fetchone()
        finally:
            conn.close()
    else:
        stats = (0, 0, 0)

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    font_name = _get_pdf_font(pdf) or "Helvetica"

    # Header
    pdf.set_font(font_name, size=16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, org_name, ln=True, align="C")
    pdf.set_font(font_name, size=12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, f"FaceID — Otchet o poseshchyaemosti / Attendance Report", ln=True, align="C")
    pdf.cell(0, 7, f"Period: {start_date}  —  {end_date}", ln=True, align="C")
    pdf.ln(4)

    # Summary box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(15, pdf.get_y(), 180, 22, "FD")
    pdf.set_font(font_name, size=10)
    pdf.set_text_color(15, 23, 42)
    allow_cnt, deny_cnt, sus_cnt = (int(v or 0) for v in (stats or (0, 0, 0)))
    pdf.set_xy(17, pdf.get_y() + 3)
    pdf.cell(55, 7, f"Razresheno: {allow_cnt}", border=0)
    pdf.cell(55, 7, f"Otkazov: {deny_cnt}", border=0)
    pdf.cell(55, 7, f"Podozritelnykh: {sus_cnt}", border=0)
    pdf.ln(16)

    # Table header
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font_name, size=10)
    col_w = [35, 80, 35, 30]
    headers_txt = ["Data", "Sotrudnik / Identity", "Pervyi vkhod", "Sobytii"]
    for w, h in zip(col_w, headers_txt):
        pdf.cell(w, 8, h, border=0, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_text_color(15, 23, 42)
    for i, (identity, first_ts, cnt, day) in enumerate(rows_data):
        fill = i % 2 == 0
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        time_str = str(first_ts)[11:16] if first_ts else ""
        pdf.set_font(font_name, size=9)
        pdf.cell(col_w[0], 7, str(day or ""), border=0, fill=True, align="C")
        pdf.cell(col_w[1], 7, str(identity or "")[:35], border=0, fill=True)
        pdf.cell(col_w[2], 7, time_str, border=0, fill=True, align="C")
        pdf.cell(col_w[3], 7, str(cnt), border=0, fill=True, align="C")
        pdf.ln()

    # Footer
    pdf.set_y(-20)
    pdf.set_font(font_name, size=8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f"FaceID Access Control System  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             align="C")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return len(rows_data)
