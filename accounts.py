"""
Accounts and saved projects: SQLite in data/, standard library only.

Passwords are hashed with scrypt and a per-user salt. A signed-in browser holds one HttpOnly cookie
whose value is `user_id.expiry.signature`, signed with HMAC-SHA256 under AUTH_SECRET (from .env, or
generated once into data/secret.key). Nothing here touches the roof pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Response

DATA_DIR = Path(os.environ.get("ROOFLINE_DATA", Path(__file__).resolve().parent / "data"))
PROJECT_DIR = DATA_DIR / "projects"
COOKIE = "roofline_session"
SESSION_DAYS = 14
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_lock = threading.Lock()
_failures: dict[str, list[float]] = {}          # login throttling: key -> recent failure times


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATA_DIR / "roofline.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


DB = _connect()
DB.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    password_hash TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, filename TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    data TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS projects_user ON projects(user_id, updated_at);
""")


def _secret() -> bytes:
    env = os.environ.get("AUTH_SECRET", "").strip()
    if env:
        return env.encode()
    path = DATA_DIR / "secret.key"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(48))
        path.chmod(0o600)
    return path.read_text().strip().encode()


SECRET = _secret()


# --------------------------------------------------------------------------------------------
# Passwords and tokens
# --------------------------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        test = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(test, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def _sign(payload: str) -> str:
    return base64.urlsafe_b64encode(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")


def make_token(user_id: str) -> str:
    payload = f"{user_id}.{int(time.time()) + SESSION_DAYS * 86400}"
    return f"{payload}.{_sign(payload)}"


def read_token(token: str) -> Optional[str]:
    try:
        user_id, expiry, signature = token.rsplit(".", 2)
    except ValueError:
        return None
    if not hmac.compare_digest(signature, _sign(f"{user_id}.{expiry}")) or int(expiry) < time.time():
        return None
    return user_id


def set_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(COOKIE, make_token(user_id), max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=os.environ.get("COOKIE_SECURE", "").lower() in ("1", "true", "yes"), path="/")


def clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/")


# --------------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------------

def public_user(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


def signup_allowed() -> bool:
    return os.environ.get("ALLOW_SIGNUP", "true").lower() not in ("0", "false", "no")


def create_user(email: str, name: str, password: str) -> dict:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(400, "Use a password of at least 8 characters.")
    user = {"id": uuid.uuid4().hex, "email": email, "name": name.strip() or email.split("@")[0]}
    with _lock:
        try:
            DB.execute("INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                       (user["id"], email, user["name"], hash_password(password), time.time()))
            DB.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, "An account with this email already exists. Sign in instead.")
    return user


def authenticate(email: str, password: str, client: str) -> dict:
    email = email.strip().lower()
    key = f"{client}|{email}"
    now = time.time()
    recent = [t for t in _failures.get(key, []) if now - t < 600]
    if len(recent) >= 8:
        raise HTTPException(429, "Too many failed attempts. Wait ten minutes and try again.")
    row = DB.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        _failures[key] = recent + [now]
        raise HTTPException(401, "Email or password is wrong.")
    _failures.pop(key, None)
    return public_user(row)


def optional_user(request: Request) -> Optional[dict]:
    user_id = read_token(request.cookies.get(COOKIE, ""))
    if not user_id:
        return None
    row = DB.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(row) if row else None


def current_user(request: Request) -> dict:
    """FastAPI dependency: the signed-in user, or 401."""
    user = optional_user(request)
    if user is None:
        raise HTTPException(401, "Sign in to continue.")
    return user


# --------------------------------------------------------------------------------------------
# Projects: the photo on disk, everything else as one JSON document
# --------------------------------------------------------------------------------------------

def _summary(row: sqlite3.Row) -> dict:
    data = json.loads(row["data"])
    edited = data.get("edited") or {}
    detected = data.get("detected") or {}
    return {"id": row["id"], "name": row["name"], "filename": row["filename"], "width": row["width"], "height": row["height"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "segments": len((edited or detected).get("segments", [])), "edited": bool(edited),
            "design": bool(data.get("design")), "lines": sum(1 for r in (data.get("design") or {}).get("regions", []) if r.get("geometry") == "line" and r.get("visible")),
            "areas": sum(1 for r in (data.get("design") or {}).get("regions", []) if r.get("geometry") == "area"),
            "approved": bool((data.get("design") or {}).get("approved")),
            "image_url": f"/api/projects/{row['id']}/image"}


def image_path(project_id: str) -> Path:
    return PROJECT_DIR / f"{project_id}.jpg"


def list_projects(user_id: str) -> list[dict]:
    rows = DB.execute("SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
    return [_summary(r) for r in rows]


def get_project(user_id: str, project_id: str) -> dict:
    row = DB.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Project not found.")
    return {**_summary(row), "data": json.loads(row["data"])}


def create_project(user_id: str, name: str, filename: str, width: int, height: int, jpeg: bytes, data: dict) -> dict:
    project_id, now = uuid.uuid4().hex[:16], time.time()
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    image_path(project_id).write_bytes(jpeg)
    with _lock:
        DB.execute("INSERT INTO projects (id, user_id, name, filename, width, height, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (project_id, user_id, name, filename, width, height, json.dumps(data), now, now))
        DB.commit()
    return get_project(user_id, project_id)


def update_project(user_id: str, project_id: str, name: Optional[str], data: Optional[dict]) -> dict:
    current = get_project(user_id, project_id)
    merged = {**current["data"], **(data or {})}
    with _lock:
        DB.execute("UPDATE projects SET name = ?, data = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                   (name or current["name"], json.dumps(merged), time.time(), project_id, user_id))
        DB.commit()
    return get_project(user_id, project_id)


def delete_project(user_id: str, project_id: str) -> None:
    get_project(user_id, project_id)
    with _lock:
        DB.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
        DB.commit()
    image_path(project_id).unlink(missing_ok=True)
