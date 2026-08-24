import sys
import os
from dotenv import load_dotenv

load_dotenv()
import json
import logging
import random
from pathlib import Path

import sqlite3

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from referential_task.state import Player, Group, Session
from referential_task.state import Constants
from referential_task.ai_vs_ai import get_ai_vs_ai_status, run_ai_vs_ai_turn
from referential_task.exporting import export_session_from_db, load_session_export_data
from referential_task.visual_context import _load_matcher_pool_image_urls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence chatty third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Referential Game AI-AI Simulation")

ACTIVE_PROMPT_STRATEGIES = {"ACL_prompt", "cameron-prompt"}

# Database setup
DB_FILE = "data.sqlite"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            round_number INTEGER,
            config TEXT,
            shared_grid TEXT,
            target_baskets TEXT,
            ai_partial_sequence TEXT,
            ai_messages TEXT,
            ai_reasoning_log TEXT,
            matcher_sequence TEXT,
            status TEXT,
            ai_director_reasoning TEXT,
            ai_matcher_reasoning TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_state_to_db(session_id, player):
    """Save or update the current simulation state in SQLite"""
    group = player.group
    session = player.session
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if record exists
    c.execute("SELECT id FROM game_sessions WHERE session_id = ? AND round_number = ?", 
              (session_id, player.round_number))
    row = c.fetchone()
    
    status = get_ai_vs_ai_status(player)
    
    if row:
        c.execute('''
            UPDATE game_sessions SET 
                shared_grid = ?,
                target_baskets = ?,
                ai_partial_sequence = ?,
                ai_messages = ?,
                ai_reasoning_log = ?,
                matcher_sequence = ?,
                status = ?,
                ai_director_reasoning = ?,
                ai_matcher_reasoning = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            group.shared_grid,
            group.target_baskets,
            group.ai_partial_sequence,
            group.ai_messages,
            group.ai_reasoning_log,
            group.matcher_sequence,
            json.dumps(status),
            group.ai_director_perceptions_raw,
            group.ai_matcher_perceptions_raw,
            row[0]
        ))
    else:
        c.execute('''
            INSERT INTO game_sessions (
                session_id, round_number, config, shared_grid, target_baskets,
                ai_partial_sequence, ai_messages, ai_reasoning_log, matcher_sequence,
                status, ai_director_reasoning, ai_matcher_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            player.round_number,
            json.dumps(session.config),
            group.shared_grid,
            group.target_baskets,
            group.ai_partial_sequence,
            group.ai_messages,
            group.ai_reasoning_log,
            group.matcher_sequence,
            json.dumps(status),
            group.ai_director_perceptions_raw,
            group.ai_matcher_perceptions_raw
        ))
    
    conn.commit()
    conn.close()


# Mount static files
app.mount("/static", StaticFiles(directory="_static"), name="static")

# Setup Jinja2 templates location
# We will create a fresh directory for standalone templates
templates = Jinja2Templates(directory="templates")

# In-memory storage of simulation states
# Key: session_id, Value: dict containing Player object and status
active_simulations = {}

def get_total_rounds(session=None):
    """Return the configured number of rounds, falling back to the task default."""
    try:
        if session is not None:
            return int(session.config.get("num_rounds") or Constants.num_rounds)
    except Exception:
        pass
    return Constants.num_rounds

def get_preset_grid(round_number=1, set_num=5):
    """Load a grid preset like oTree did in create_shared_grid"""
    preset_filename = f"grids_presets{set_num}.json"
    preset_path = Path("referential_task") / preset_filename
    
    grid = []
    try:
        if preset_path.exists():
            with open(preset_path, "r") as f:
                presets = json.load(f)
            for round_cfg in presets.get("rounds", []):
                if round_cfg.get("round") == round_number:
                    basket_files = [f"images/{img}" for img in round_cfg["baskets"]]
                    position_index = 0
                    for row in range(1, 4):
                        for col in range(1, 5):
                            grid.append({
                                "position": f"{row}{col}",
                                "row": row,
                                "col": col,
                                "image": basket_files[position_index],
                                "basket_id": position_index + 1
                            })
                            position_index += 1
                    break
    except Exception as e:
        logger.error(f"Error loading preset: {e}")

    # Fallback to random if no preset found
    if not grid:
        all_images = [f"images/{i:03d}.png" for i in range(1, 71)]
        selected_images = random.sample(all_images, 12)
        position_index = 0
        for row in range(1, 4):
            for col in range(1, 5):
                grid.append({
                    "position": f"{row}{col}",
                    "row": row,
                    "col": col,
                    "image": selected_images[position_index],
                    "basket_id": position_index + 1
                })
                position_index += 1

    return grid

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main simulation UI."""
    # Render the AIvsAIObservation dashboard
    return templates.TemplateResponse("AIvsAIObservation.html", {"request": request})

@app.post("/api/game/start")
async def start_game(data: dict):
    """Initialize a new simulation round."""
    session_id = data.get("session_id", "local_test_1")
    round_number = int(data.get("round_number", 1))
    prompt_strategy = data.get("prompt_strategy", "ACL_prompt")
    if prompt_strategy not in ACTIVE_PROMPT_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail="prompt_strategy must be 'ACL_prompt' or 'cameron-prompt'.",
        )
    
    # Configure session
    session_config = {
        "ai_vs_ai_mode": True,
        "director_view": "grid",
        "basket_set": int(data.get("basket_set", 5)),
        "num_rounds": Constants.num_rounds,
        "session_prefix": data.get("session_prefix") or session_id,
        "prompt_strategy": prompt_strategy,
        "cross_round_history": bool(data.get("cross_round_history", False)),
        "debug_prompt_context": bool(data.get("debug_prompt_context", False)),
        "ai_director_model": data.get("director_model") or os.environ.get("AI_DIRECTOR_MODEL", "gpt-4o-mini"),
        "ai_matcher_model": data.get("matcher_model") or os.environ.get("AI_MATCHER_MODEL", "gpt-4o-mini"),
        "ai_reasoning_effort": data.get("reasoning_effort", "none"),
    }
    session = Session(session_config)
    
    group = Group()
    grid = get_preset_grid(round_number=round_number, set_num=session_config["basket_set"])
    group.shared_grid = json.dumps(grid)
    
    player = Player(role="observer", group=group, session=session, round_number=round_number)
    
    active_simulations[session_id] = {
        "player": player,
        "round": round_number,
        "status": "ready"
    }
    
    save_state_to_db(session_id, player)
    
    return JSONResponse({"status": "Simulation started", "session_id": session_id})

@app.post("/api/game/turn")
async def play_turn(data: dict):
    """Execute one turn of the AI vs AI interaction."""
    session_id = data.get("session_id")
    if session_id not in active_simulations:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
        
    sim = active_simulations[session_id]
    player = sim["player"]
    
    try:
        run_ai_vs_ai_turn(player)
        save_state_to_db(session_id, player)
        
        # Auto-advance to the next round if the current one is completely finished
        if getattr(player, "task_completed", False):
            current_round = player.round_number
            total_rounds = get_total_rounds(player.session)
            if current_round < total_rounds:
                next_round = current_round + 1
                
                # Initialize new game state for the next round
                new_group = Group()
                new_grid = get_preset_grid(
                    round_number=next_round,
                    set_num=player.session.config.get("basket_set", 5),
                )
                new_group.shared_grid = json.dumps(new_grid)

                new_player = Player(role="observer", group=new_group, session=player.session, round_number=next_round)
                
                # Overwrite active simulation with the new round's player
                active_simulations[session_id]["player"] = new_player
                active_simulations[session_id]["round"] = next_round
                
                # Save the fresh next round to DB immediately
                save_state_to_db(session_id, new_player)
                
                # Update the active reference so the status returned matches the new round
                player = new_player
            else:
                # Final round complete: Evaluate mutual perceptions
                if not getattr(player, "perceptions_generated", False):
                    player.perceptions_generated = True
                    logger.info("Round %s finished! Fetching AI vs AI mutual perceptions.", current_round)
                    from referential_task.ai_perceptions import generate_ai_vs_ai_perceptions
                    generate_ai_vs_ai_perceptions(player)
                    save_state_to_db(session_id, player)
                    
                    # --- AUTO-EXPORT DATA ---
                    try:
                        logger.info("Auto-exporting full session data to structured data folder...")
                        export_session_from_db(
                            DB_FILE,
                            session_id,
                            player.session.config,
                            generate_artifacts=True,
                            logger=logger,
                        )
                    except Exception as export_e:
                        logger.error(f"Auto-export failed: {export_e}")

        status = get_ai_vs_ai_status(player)
        return JSONResponse({"status": "Turn executed", "game_status": status})
    except Exception as e:
        logger.exception("Error during turn")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/game/state")
async def get_state(session_id: str):
    """Get the current state of the simulation for the UI."""
    if session_id not in active_simulations:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
        
    sim = active_simulations[session_id]
    player = sim["player"]
    group = player.group
    
    status = get_ai_vs_ai_status(player)
    
    ai_messages = []
    if group.ai_messages:
        try:
            ai_messages = json.loads(group.ai_messages)
        except Exception:
            pass
            
    partial_sequence = []
    if group.ai_partial_sequence:
        try:
            partial_sequence = json.loads(group.ai_partial_sequence)
        except Exception:
            pass
            
    reasoning_log = []
    if group.ai_reasoning_log:
        try:
            reasoning_log = json.loads(group.ai_reasoning_log)
        except Exception:
            pass

    pool_urls = _load_matcher_pool_image_urls(player) or []
    matcher_pool = [item.get("slot", {}).get("image", "").lstrip("/ ") for item in pool_urls if item.get("slot", {}).get("image")]

    return JSONResponse({
        "status": status,
        "round_number": player.round_number,
        "prompt_strategy": player.session.config.get("prompt_strategy", "ACL_prompt"),
        "cross_round_history": bool(player.session.config.get("cross_round_history", False)),
        "debug_prompt_context": bool(player.session.config.get("debug_prompt_context", False)),
        "director_model": player.session.config.get("ai_director_model", "unknown"),
        "matcher_model": player.session.config.get("ai_matcher_model", "unknown"),
        "ai_messages": ai_messages,
        "partial_sequence": partial_sequence,
        "reasoning_log": reasoning_log,
        "shared_grid": json.loads(group.shared_grid),
        "matcher_pool": matcher_pool
    })

@app.get("/api/game/export")
async def export_session(session_id: str, generate_artifacts: bool = False):
    """Export the current session's data as a JSON file."""
    if not os.path.exists(DB_FILE):
        return JSONResponse({"error": "Database not found"}, status_code=404)
        
    sessions = load_session_export_data(DB_FILE, session_id)
    if not sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    if generate_artifacts:
        try:
            config = sessions[0].get("config", {}) if sessions else {}
            export_session_from_db(
                DB_FILE,
                session_id,
                config,
                generate_artifacts=True,
                logger=logger,
            )
        except Exception as export_e:
            logger.error("UI export artifact generation failed: %s", export_e)

    headers = {
        "Content-Disposition": f"attachment; filename={session_id}_data.json"
    }
    return JSONResponse(content=sessions, headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, access_log=False)
