import sqlite3
import json
import argparse
import os

DB_FILE = "data.sqlite"

def export_sessions_to_json(output_file: str):
    """Query data.sqlite and dump all simulation sessions into a formatted JSON array."""
    if not os.path.exists(DB_FILE):
        print(f"Error: Database {DB_FILE} not found. Please run the simulation first.")
        return

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT session_id, round_number, config, shared_grid, target_baskets, ai_partial_sequence, ai_messages, ai_reasoning_log, matcher_sequence, status, ai_director_reasoning, ai_matcher_reasoning, updated_at FROM game_sessions")
    
    rows = c.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        (session_id, round_number, config_txt, shared_grid_txt, target_baskets_txt,
         partial_seq_txt, ai_msgs_txt, ai_reasoning_txt, matcher_seq_txt,
         status_txt, director_reasoning_txt, matcher_reasoning_txt, updated_at) = row
        
        # Safely parse JSON text back into Python objects for a clean JSON dump
        def safe_json(val):
            try:
                return json.loads(val) if val else []
            except Exception:
                return val

        session_data = {
            "session_id": session_id,
            "round_number": round_number,
            "updated_at": updated_at,
            "config": safe_json(config_txt),
            "status": safe_json(status_txt),
            "shared_grid": safe_json(shared_grid_txt),
            "target_baskets": safe_json(target_baskets_txt),
            "ai_partial_sequence": safe_json(partial_seq_txt),
            "matcher_sequence": safe_json(matcher_seq_txt),
            "ai_messages": safe_json(ai_msgs_txt),
            "ai_reasoning_log": safe_json(ai_reasoning_txt),
            "ai_director_reasoning": safe_json(director_reasoning_txt),
            "ai_matcher_reasoning": safe_json(matcher_reasoning_txt)
        }
        sessions.append(session_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=4)
        
    print(f"✅ Successfully exported {len(sessions)} sessions to {output_file}")
    
    # Automatically generate visuals and transcript
    import subprocess
    import sys
    print("🎨 Auto-generating visualizations and transcript...")
    try:
        subprocess.run([sys.executable, "scripts/export_json_session.py", output_file], check=True)
    except Exception as e:
        print(f"⚠️ Failed to generate visualizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export game sessions from SQLite to JSON.")
    parser.add_argument("--output", "-o", type=str, default="data/exported_sessions.json", help="Output JSON filename")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_sessions_to_json(args.output)
