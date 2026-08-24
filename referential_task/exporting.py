from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SESSION_SELECT = (
    "SELECT session_id, round_number, config, shared_grid, target_baskets, "
    "ai_partial_sequence, ai_messages, ai_reasoning_log, matcher_sequence, "
    "status, ai_director_reasoning, ai_matcher_reasoning, updated_at "
    "FROM game_sessions WHERE session_id = ? ORDER BY round_number"
)


def safe_json(value: str | None) -> Any:
    try:
        return json.loads(value) if value else []
    except Exception:
        return value


def get_experiment_family(prompt_strategy: str | None) -> str:
    strategy = str(prompt_strategy or "").lower()
    if "cameron" in strategy:
        return "Cameron"
    if "acl" in strategy:
        return "ACL"
    return "Other"


def sanitize_path_part(value: Any, fallback: str = "session") -> str:
    text = str(value or "").strip() or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or fallback


def get_experiment_export_dir(config: dict[str, Any] | None, session_id: str) -> Path:
    config = config if isinstance(config, dict) else {}
    prompt_strategy = config.get("prompt_strategy", "unknown")
    prefix = sanitize_path_part(config.get("session_prefix") or session_id)
    current_date = datetime.now().strftime("%Y-%m-%d")
    return (
        Path("data")
        / "experiments"
        / get_experiment_family(prompt_strategy)
        / current_date
        / prefix
        / session_id
    )


def load_session_export_data(db_file: str, session_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_file)
    try:
        cursor = conn.cursor()
        cursor.execute(SESSION_SELECT, (session_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    sessions: list[dict[str, Any]] = []
    for row in rows:
        (
            s_id,
            round_number,
            config,
            shared_grid,
            target_baskets,
            ai_partial_sequence,
            ai_messages,
            ai_reasoning_log,
            matcher_sequence,
            status,
            ai_director_reasoning,
            ai_matcher_reasoning,
            updated_at,
        ) = row
        sessions.append(
            {
                "session_id": s_id,
                "round_number": round_number,
                "updated_at": updated_at,
                "config": safe_json(config),
                "status": safe_json(status),
                "shared_grid": safe_json(shared_grid),
                "target_baskets": safe_json(target_baskets),
                "ai_partial_sequence": safe_json(ai_partial_sequence),
                "matcher_sequence": safe_json(matcher_sequence),
                "ai_messages": safe_json(ai_messages),
                "ai_reasoning_log": safe_json(ai_reasoning_log),
                "ai_director_reasoning": safe_json(ai_director_reasoning),
                "ai_matcher_reasoning": safe_json(ai_matcher_reasoning),
            }
        )
    return sessions


def write_session_export(
    sessions: list[dict[str, Any]],
    session_id: str,
    config: dict[str, Any] | None = None,
) -> Path:
    export_config = config or (sessions[0].get("config", {}) if sessions else {})
    export_dir = get_experiment_export_dir(export_config, session_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{session_id}_data.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=4)
    return export_path


def generate_export_artifacts(export_path: Path, logger: Any | None = None) -> bool:
    result = subprocess.run(
        [sys.executable, "scripts/export_json_session.py", str(export_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        if logger is not None:
            logger.info("Generated transcript/visual artifacts for %s", export_path)
            if result.stdout:
                logger.info("Session export output: %s", result.stdout.strip())
        return True

    if logger is not None:
        logger.error(
            "Transcript/visual export failed for %s with exit code %s: %s",
            export_path,
            result.returncode,
            (result.stderr or result.stdout or "").strip(),
        )
    return False


def export_session_from_db(
    db_file: str,
    session_id: str,
    config: dict[str, Any] | None = None,
    *,
    generate_artifacts: bool = False,
    logger: Any | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    sessions = load_session_export_data(db_file, session_id)
    export_path = write_session_export(sessions, session_id, config)
    if logger is not None:
        logger.info("Exported session data to %s", export_path)
    if generate_artifacts:
        generate_export_artifacts(export_path, logger=logger)
    return sessions, export_path
