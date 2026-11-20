"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start timestamp keyed by request_id or user_id."""
        key = request_id or user_id
        self._open[key] = {
            "started_at": time.perf_counter(),
            "timestamp_in": utc_now_iso(),
            "user_id": user_id,
            "input": text,
            "request_id": request_id,
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, and latency; append entry to self.logs."""
        key = request_id or user_id
        entry = self._open.pop(key, None)
        latency_ms = (
            round((time.perf_counter() - entry["started_at"]) * 1000, 2)
            if entry is not None
            else 0.0
        )
        record = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": entry["input"] if entry is not None else "",
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath) if filepath else Path(default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
