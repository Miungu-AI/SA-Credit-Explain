"""
Human oversight and review management.
"""
import sqlite3
import json
from pathlib import Path
from src.accountability.audit_logger import DB_PATH, init_db


def flag_for_review(probability: float, lower=0.30, upper=0.55) -> bool:
    """Flag borderline or low-confidence predictions for manual review."""
    return lower <= probability < upper


def record_review(application_id: str, reviewer: str, outcome: str, reason: str = ""):
    """Record a human review outcome (e.g., 'UPHELD', 'OVERRIDDEN')."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    override = 1 if outcome.upper() == "OVERRIDDEN" else 0
    cur.execute("""
        UPDATE audit_log
        SET human_reviewer = ?, override_flag = ?, override_reason = ?
        WHERE application_id = ?
    """, (reviewer, override, reason, application_id))
    conn.commit()
    conn.close()
    return {"application_id": application_id, "reviewed_by": reviewer, "outcome": outcome}


def get_review_queue():
    """Return decisions that require manual review (borderline or no reviewer yet)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM audit_log
        WHERE decision = 'MANUAL_REVIEW' AND human_reviewer IS NULL
        ORDER BY timestamp DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
