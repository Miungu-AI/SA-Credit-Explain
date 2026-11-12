"""
Immutable-style audit trail with SHA-256 hash chaining.
Stores minimal necessary information. No unnecessary PII.
"""
import sqlite3
import hashlib
import json
import datetime
from pathlib import Path

DB_PATH = Path("audit.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            default_probability REAL,
            decision TEXT,
            explanation_json TEXT,
            fairness_status TEXT,
            human_reviewer TEXT,
            override_flag INTEGER DEFAULT 0,
            override_reason TEXT,
            previous_hash TEXT,
            record_hash TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()
    conn.close()


def _hash_record(record: dict, previous_hash: str) -> str:
    payload = json.dumps(record, sort_keys=True).encode("utf-8") + previous_hash.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def log_decision(
    application_id: str,
    model_name: str,
    model_version: str,
    application_input: dict,
    default_probability: float,
    decision: str,
    explanation: dict,
    fairness_status: str = "NOT_AUDITED",
    human_reviewer: str = None,
    override_flag: int = 0,
    override_reason: str = None,
):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get previous hash
    cur.execute("SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    previous_hash = row[0] if row else "GENESIS"

    # Hash the input instead of storing raw PII
    input_hash = hashlib.sha256(
        json.dumps(application_input, sort_keys=True).encode("utf-8")
    ).hexdigest()

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    record_core = {
        "application_id": application_id,
        "model_name": model_name,
        "model_version": model_version,
        "timestamp": timestamp,
        "input_hash": input_hash,
        "default_probability": default_probability,
        "decision": decision,
        "explanation": explanation,
        "fairness_status": fairness_status,
        "human_reviewer": human_reviewer,
        "override_flag": override_flag,
        "override_reason": override_reason,
    }

    record_hash = _hash_record(record_core, previous_hash)

    cur.execute("""
        INSERT INTO audit_log (
            application_id, model_name, model_version, timestamp, input_hash,
            default_probability, decision, explanation_json, fairness_status,
            human_reviewer, override_flag, override_reason, previous_hash, record_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        application_id, model_name, model_version, timestamp, input_hash,
        default_probability, decision, json.dumps(explanation), fairness_status,
        human_reviewer, override_flag, override_reason, previous_hash, record_hash,
    ))
    conn.commit()
    conn.close()
    return record_hash


def verify_chain():
    """Verify the integrity of the entire audit chain."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_log ORDER BY id ASC")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    conn.close()

    if not rows:
        return {"valid": True, "records": 0, "message": "No records."}

    previous_hash = "GENESIS"
    for i, row in enumerate(rows):
        rec = dict(zip(columns, row))
        stored_hash = rec["record_hash"]
        core = {
            "application_id": rec["application_id"],
            "model_name": rec["model_name"],
            "model_version": rec["model_version"],
            "timestamp": rec["timestamp"],
            "input_hash": rec["input_hash"],
            "default_probability": rec["default_probability"],
            "decision": rec["decision"],
            "explanation": json.loads(rec["explanation_json"]),
            "fairness_status": rec["fairness_status"],
            "human_reviewer": rec["human_reviewer"],
            "override_flag": rec["override_flag"],
            "override_reason": rec["override_reason"],
        }
        expected = _hash_record(core, previous_hash)
        if expected != stored_hash:
            return {
                "valid": False,
                "broken_at_id": rec["id"],
                "expected": expected,
                "stored": stored_hash,
            }
        previous_hash = stored_hash

    return {"valid": True, "records": len(rows), "message": "Chain intact."}


def get_all_records():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_log ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
