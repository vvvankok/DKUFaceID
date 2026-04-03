from __future__ import annotations

"""
Fernet symmetric encryption helpers.

Used for encrypting database backups and sensitive incident data at rest,
satisfying ТЗ §4.5 (cryptography) and §4.7 (data protection).

Typical usage:
    key = ensure_key(Path("artifacts/faceid.key"))
    encrypted = encrypt_bytes(data, key)
    original  = decrypt_bytes(encrypted, key)

    # encrypted SQLite backup:
    backup_encrypted(db_path, Path("artifacts/backup.db.enc"), key)
"""

from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken  # noqa: F401


def generate_key() -> bytes:
    """Return a fresh 256-bit Fernet key (URL-safe base64, 44 bytes)."""
    return Fernet.generate_key()


def save_key(key: bytes, key_path: Path) -> None:
    """Write *key* to *key_path* with restrictive permissions."""
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    try:
        import stat
        key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600 on POSIX
    except Exception:
        pass  # Windows does not support POSIX chmod — acceptable


def load_key(key_path: Path) -> bytes:
    """Load a Fernet key from *key_path*.  Raises FileNotFoundError if absent."""
    return key_path.read_bytes()


def ensure_key(key_path: Path) -> bytes:
    """Return existing key or generate and save a new one."""
    if key_path.exists():
        return load_key(key_path)
    key = generate_key()
    save_key(key, key_path)
    return key


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Encrypt *data* with *key* using Fernet (AES-128-CBC + HMAC-SHA256)."""
    return Fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    """Decrypt a Fernet *token* with *key*.  Raises InvalidToken on bad data."""
    return Fernet(key).decrypt(token)


def backup_encrypted(db_path: Path, out_path: Path, key: bytes) -> None:
    """
    Create an encrypted binary backup of the SQLite database at *db_path*.

    Uses SQLite's native streaming backup API so the source DB is never
    partially read.  The output file is Fernet-encrypted and unreadable
    without the key.
    """
    import sqlite3, tempfile

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1 — dump to an in-memory DB first (clean consistent snapshot)
    src = sqlite3.connect(str(db_path))
    mem = sqlite3.connect(":memory:")
    try:
        src.backup(mem)
    finally:
        src.close()

    # Step 2 — write in-memory DB to a temp file so we can read raw bytes
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        dst = sqlite3.connect(str(tmp_path))
        try:
            mem.backup(dst)
        finally:
            dst.close()
        mem.close()
        raw = tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Step 3 — encrypt and write
    out_path.write_bytes(encrypt_bytes(raw, key))


def restore_encrypted(enc_path: Path, restore_path: Path, key: bytes) -> None:
    """
    Decrypt *enc_path* (produced by :func:`backup_encrypted`) and write the
    plain SQLite database to *restore_path*.

    Raises ``cryptography.fernet.InvalidToken`` if the key is wrong or the
    file has been tampered with.
    """
    restore_path.parent.mkdir(parents=True, exist_ok=True)
    raw = decrypt_bytes(enc_path.read_bytes(), key)
    restore_path.write_bytes(raw)
