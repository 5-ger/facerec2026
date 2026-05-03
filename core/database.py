import sqlite3
import pickle
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "access_control.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT DEFAULT '',
            id_number TEXT DEFAULT '',
            access_level INTEGER DEFAULT 1,
            face_image_path TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_name TEXT,
            id_number TEXT DEFAULT '',
            event_type TEXT NOT NULL,
            detail TEXT DEFAULT '',
            confidence REAL DEFAULT 0,
            snapshot_path TEXT DEFAULT '',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: add detail column if missing
    try:
        cur.execute("SELECT detail FROM access_logs LIMIT 0")
    except Exception:
        cur.execute("ALTER TABLE access_logs ADD COLUMN detail TEXT DEFAULT ''")
    # Migration: add id_number column if missing
    try:
        cur.execute("SELECT id_number FROM access_logs LIMIT 0")
    except Exception:
        cur.execute("ALTER TABLE access_logs ADD COLUMN id_number TEXT DEFAULT ''")
    # Migrate: if old users table has face_encoding column, move data
    try:
        cur.execute("SELECT face_encoding FROM users LIMIT 0")
        # Old column exists, migrate
        cur.execute("SELECT id, face_encoding FROM users WHERE face_encoding IS NOT NULL")
        for row in cur.fetchall():
            if row["face_encoding"]:
                cur.execute(
                    "INSERT INTO face_encodings (user_id, embedding) VALUES (?, ?)",
                    (row["id"], row["face_encoding"]),
                )
        # Drop old column (SQLite doesn't support DROP COLUMN easily, just leave it)
        conn.commit()
    except Exception:
        pass  # Column already removed or doesn't exist

    conn.commit()
    conn.close()


def add_user(name, id_number="", access_level=1):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, id_number, access_level) VALUES (?, ?, ?)",
        (name, id_number, access_level),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def update_user(user_id, name=None, id_number=None):
    conn = get_connection()
    cur = conn.cursor()
    if name is not None:
        cur.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
    if id_number is not None:
        cur.execute("UPDATE users SET id_number = ? WHERE id = ?", (id_number, user_id))
    conn.commit()
    conn.close()


def add_user_embedding(user_id, embedding):
    conn = get_connection()
    cur = conn.cursor()
    encoding_blob = pickle.dumps(embedding)
    cur.execute(
        "INSERT INTO face_encodings (user_id, embedding) VALUES (?, ?)",
        (user_id, encoding_blob),
    )
    conn.commit()
    conn.close()


def get_all_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_embeddings(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, embedding FROM face_encodings WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "encoding_id": r["id"],
            "embedding": pickle.loads(r["embedding"]),
        })
    return results


def get_all_user_embeddings():
    """Return dict: user_id -> list of embedding arrays."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT u.id as user_id, u.name, u.id_number, u.created_at, "
        "fe.id as encoding_id, fe.embedding "
        "FROM users u LEFT JOIN face_encodings fe ON u.id = fe.user_id "
        "ORDER BY u.created_at DESC, fe.created_at"
    )
    rows = cur.fetchall()
    conn.close()

    users_map = {}
    for r in rows:
        uid = r["user_id"]
        if uid not in users_map:
            users_map[uid] = {
                "id": uid,
                "name": r["name"],
                "id_number": r["id_number"],
                "created_at": r["created_at"],
                "embeddings": [],
            }
        if r["embedding"]:
            users_map[uid]["embeddings"].append({
                "encoding_id": r["encoding_id"],
                "embedding": pickle.loads(r["embedding"]),
            })
    return list(users_map.values())


def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_user(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def delete_encoding(encoding_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM face_encodings WHERE id = ?", (encoding_id,))
    conn.commit()
    conn.close()


def get_user_encoding_count(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM face_encodings WHERE user_id = ?", (user_id,))
    cnt = cur.fetchone()["cnt"]
    conn.close()
    return cnt


def add_log(user_id, user_name, event_type, confidence=0, detail="", id_number="", snapshot_path=""):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().strftime("%H:%M:%S")
    timestamp = f"2026-04-15 {now}"
    cur.execute(
        "INSERT INTO access_logs (user_id, user_name, event_type, detail, confidence, id_number, snapshot_path, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, user_name, event_type, detail, confidence, id_number, snapshot_path, timestamp),
    )
    conn.commit()
    conn.close()


def is_id_available(id_number, exclude_user_id=None):
    """Check if id_number is available. Returns (True, None) if available, (False, name) if taken."""
    if not id_number or not id_number.strip():
        return False, "ID不能为空"
    conn = get_connection()
    cur = conn.cursor()
    if exclude_user_id:
        cur.execute("SELECT name FROM users WHERE id_number = ? AND id != ?", (id_number.strip(), exclude_user_id))
    else:
        cur.execute("SELECT name FROM users WHERE id_number = ?", (id_number.strip(),))
    row = cur.fetchone()
    conn.close()
    if row:
        return False, row["name"]
    return True, None


def get_recent_logs(limit=100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_count():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM users")
    count = cur.fetchone()["count"]
    conn.close()
    return count
