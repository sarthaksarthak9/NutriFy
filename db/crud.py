import json
import os
from typing import Any, Dict, Optional

from sqlalchemy import delete
from sqlalchemy.orm import Session

from db.models import AppJsonDocument

# Maps legacy flat-file names to their database doc_key equivalents.
FILE_TO_KEY = {
    "user_info.json": "user_info",
    "task_info.json": "task_info",
    "user_meal_log.json": "meal_log",
}


def _get_row(db: Session, user_id: str, doc_key: str) -> Optional[AppJsonDocument]:
    """Fetch a document row by primary key (leverages SQLAlchemy identity-map cache)."""
    return db.get(AppJsonDocument, (user_id, doc_key))


def get_document(db: Session, user_id: str, doc_key: str) -> Optional[Dict[str, Any]]:
    """Return the payload dict for a document, or None if it doesn't exist."""
    row = _get_row(db, user_id, doc_key)
    if row is None:
        return None
    return dict(row.payload) if row.payload is not None else {}


def set_document(db: Session, user_id: str, doc_key: str, data: Dict[str, Any]) -> None:
    """Insert or update a document's payload and commit immediately."""
    row = _get_row(db, user_id, doc_key)
    if row is None:
        db.add(AppJsonDocument(user_id=user_id, doc_key=doc_key, payload=data))
    else:
        row.payload = data
    db.commit()


def delete_documents_for_user(db: Session, user_id: str) -> None:
    """Delete all documents belonging to a user (used on data reset)."""
    db.execute(delete(AppJsonDocument).where(AppJsonDocument.user_id == user_id))
    db.commit()


def migrate_json_files_to_db(db: Session, user_id: str) -> None:
    """One-time import of legacy flat JSON files into the database.

    Skips any doc_key that already has a row so re-runs are safe.
    """
    for filename, doc_key in FILE_TO_KEY.items():
        if _get_row(db, user_id, doc_key) is not None:
            continue
        if not os.path.isfile(filename):
            continue
        try:
            with open(filename, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                db.add(AppJsonDocument(user_id=user_id, doc_key=doc_key, payload=data))
        except (json.JSONDecodeError, OSError):
            continue
    db.commit()
