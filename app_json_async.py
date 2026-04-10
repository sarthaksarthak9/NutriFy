"""
Async helpers that back the app's JSON documents with SQLAlchemy / SQLite
instead of plain flat files.  All agents call load_json_async / save_json_async
with the original filename paths – this module transparently routes known files
to the database and falls back to the filesystem for anything else.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from typing import Any, Dict, Optional

from db.crud import FILE_TO_KEY, delete_documents_for_user, get_document, migrate_json_files_to_db, set_document
from db.session import SessionLocal, init_db

# Override with NUTRIFY_USER_ID env var to support multiple tenants.
DEFAULT_USER_ID = os.getenv("NUTRIFY_USER_ID", "default")

# --- Default document templates ---

_DEFAULT_USER_INFO: Dict[str, Any] = {
    "profile": {
        "age": None,
        "gender": None,
        "weight": None,
        "height": None,
        "activity_level": None,
    },
    "goals": {
        "goal_weight": None,
        "goal_type": None,
    },
    "preferences": {
        "dietary_restrictions": [],
        "allergies": [],
    },
}

_DEFAULT_TASK_INFO: Dict[str, Any] = {
    "profile_complete": False,
    "wants_meal_plan": None,
    "missing_fields": [],
    "current_step": "greeting",
    "current_agent": "conversation",
    "task_type": None,
}


def _default_meal_log() -> Dict[str, Any]:
    """Return a fresh meal-log skeleton (date is computed at call time)."""
    return {
        "meal_entries": {},
        "daily_summaries": {},
        "last_updated": str(date.today()),
    }


# --- Internal helpers ---

def _doc_key_for_path(file_path: str) -> Optional[str]:
    """Resolve a filename/path to its database doc_key, or None for unknown files."""
    return FILE_TO_KEY.get(os.path.basename(file_path))


def _ensure_defaults(user_id: str) -> None:
    """Insert default documents for a user if they don't already exist."""
    defaults = {
        "user_info": _DEFAULT_USER_INFO,
        "task_info": _DEFAULT_TASK_INFO,
        "meal_log": _default_meal_log(),
    }
    db = SessionLocal()
    try:
        for doc_key, payload in defaults.items():
            if get_document(db, user_id, doc_key) is None:
                set_document(db, user_id, doc_key, payload)
    finally:
        db.close()


def _init_storage(user_id: str) -> None:
    """Create tables, migrate any legacy flat files, then seed defaults."""
    init_db()
    db = SessionLocal()
    try:
        migrate_json_files_to_db(db, user_id)
    finally:
        db.close()
    _ensure_defaults(user_id)


def _load_json(file_path: str, user_id: str) -> Dict[str, Any]:
    doc_key = _doc_key_for_path(file_path)

    # Unknown file → plain filesystem read
    if not doc_key:
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return json.loads(content) if content.strip() else {}
        return {}

    db = SessionLocal()
    try:
        data = get_document(db, user_id, doc_key)
        if data is not None:
            return data

        # Fallback: import legacy file if the row is missing (tolerates races with migration).
        filename = os.path.basename(file_path)
        if os.path.isfile(filename):
            try:
                with open(filename, encoding="utf-8") as f:
                    file_data = json.load(f)
                if isinstance(file_data, dict):
                    set_document(db, user_id, doc_key, file_data)
                    return file_data
            except (json.JSONDecodeError, OSError):
                pass
        return {}
    finally:
        db.close()


def _save_json(file_path: str, data: Dict[str, Any], user_id: str) -> None:
    doc_key = _doc_key_for_path(file_path)

    # Unknown file → plain filesystem write
    if not doc_key:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    db = SessionLocal()
    try:
        set_document(db, user_id, doc_key, data)
    finally:
        db.close()


def _document_exists(file_path: str, user_id: str) -> bool:
    doc_key = _doc_key_for_path(file_path)
    if not doc_key:
        return os.path.isfile(file_path)
    db = SessionLocal()
    try:
        return get_document(db, user_id, doc_key) is not None
    finally:
        db.close()


# --- Public async API ---

async def init_app_storage(user_id: str = DEFAULT_USER_ID) -> None:
    """Async version of storage init (used by agent constructors)."""
    await asyncio.to_thread(_init_storage, user_id)


async def load_json_async(file_path: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    """Load an app document. Known file paths are read from the database."""
    return await asyncio.to_thread(_load_json, file_path, user_id)


async def save_json_async(file_path: str, data: Dict[str, Any], user_id: str = DEFAULT_USER_ID) -> None:
    """Save an app document. Known file paths are written to the database."""
    await asyncio.to_thread(_save_json, file_path, data, user_id)


async def document_exists_async(file_path: str, user_id: str = DEFAULT_USER_ID) -> bool:
    """Return True if the document exists (in DB or on disk for unknown paths)."""
    return await asyncio.to_thread(_document_exists, file_path, user_id)


# --- Public sync API (used by FastAPI lifespan / CLI) ---

def init_app_storage_sync(user_id: str = DEFAULT_USER_ID) -> None:
    """Synchronous storage init for FastAPI startup and class constructors."""
    _init_storage(user_id)


def reset_app_documents_sync(user_id: str = DEFAULT_USER_ID) -> None:
    """Delete all documents for a user and re-seed defaults (reset / test helper)."""
    db = SessionLocal()
    try:
        delete_documents_for_user(db, user_id)
    finally:
        db.close()
    _ensure_defaults(user_id)
