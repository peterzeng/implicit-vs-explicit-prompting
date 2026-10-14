# Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication

Code and data for the EMNLP 2026 paper *Implicit vs. Explicit Prompting
Strategies for LVLMs in Referential Communication* (Zeng, Paige, Li, Brennan,
Rambow, and Jones).

The paper asks why two recent studies reached opposite conclusions about
whether LVLMs coordinate on efficient referring expressions. We hold the task,
stimuli, rounds, pairing, and models constant and vary only the prompting
style, and find that models compress referring expressions when explicitly
instructed how to, but not when a preference for brevity is stated implicitly.

## Prompt conditions

The two conditions in the paper correspond to the following prompt names in
this codebase:

| Paper term | Code name         | Adapted from   |
|------------|-------------------|----------------|
| Implicit   | `ACL_prompt`      | Zeng et al. (2026)  |
| Explicit   | `cameron-prompt`  | Jones et al. (2026) |

The full text of both prompts, for both director and matcher, is in Appendix A
of the paper.

## Contents

- `main.py`, `referential_task/`, `scripts/` --- the experiment pipeline.
- `_static/` --- basket stimulus images and the observation dashboard assets.
- `data/` --- transcripts and metrics (see `data/README.md`; release pending).
- `emnlp-5rounds/`, `dotted-lines-4rounds/` --- generated result plots.
- `figures/` --- figures used in the paper.

This repository includes our corrected version of the visual-context pipeline
from Zeng et al. (2026): past-round transcripts and feedback images are kept in
chronological order, the current round's grid is injected last, and the
matcher's 12-slot sequence state and candidate pool are rendered into a single
composite image. Section 3.1 of the paper describes the correction.

## Changes Since The Paper

The `main` branch is the code as published. Later branches improve the game
rules for how the matcher moves baskets and submits. Each run records the
rules it used as `harness_version` in its config: absent or 1 for the paper,
3 for current. Don't pool results across versions.
[`docs/harness_changes.md`](docs/harness_changes.md) has the full record:
what failed, the evidence, and the alternatives we rejected.

**Why.** Re-running the paper's design with gpt-6-luna, 3 of 4 explicit-prompt
sessions got stuck in rounds that could never end. Under the paper's rules, a
basket that is already placed could only be moved if the matcher's chat
message contained a word like "move". Incomplete boards could not be
submitted, and neither refusal was reported to the matcher. gpt-6-luna chose
the right moves in its structured output (97 of 99 blocked moves targeted
the correct slot) but didn't use the keywords. A replay of all 200 rounds in
the paper's runs found no effect on the published results: GPT-5.x never
completed a move.

**What changed (version 3).**

- **Moves are an explicit field.** `selection.move: true` moves a placed
  basket. The decision comes from the structured output, not from the
  wording of a chat message.
- **Failed actions are reported to the matcher.** If a selection is ignored or
  a submit is refused, the matcher's private board-state note says so and
  why. This is the text equivalent of the human interface, where the submit
  button only appears once all 12 slots are filled.
- **The board-state note comes last** in the matcher's context, after the
  director's latest message, instead of above the whole conversation.
- **Turn cap.** A round ends after 120 turns (`--max-turns-per-round`; the
  paper's longest round was 46). The board is scored as it stands and the
  round is flagged `turn_cap_reached`.
- **Logging.** Each selection's and submit's outcome is logged
  (`selection_result`, `submit_result` in `ai_reasoning_log`).
  `scripts/audit_move_attempts.py` replays matcher selections to count move
  attempts in any runs, including the paper's.

Incomplete submits are still refused, as in the human interface.

## Licensing

Code is MIT licensed (see `LICENSE`). The data in `data/` is intended to be
released under CC BY 4.0; see `data/README.md`.

## Citation

Citation details will be added once the ACL Anthology entry exists.

## Provenance

Split out from the `emnlp` branch of
https://github.com/peterzeng/lvlms-referential-game so that the paper can cite
a stable location. The full experiment history, including runs not reported in
the paper, remains in that repository.

---

# AI-AI Referential Director-Matcher Game

This repository runs an AI-AI simulation of a referential director-matcher game. It was originally built around oTree, but the current codebase uses a lightweight **FastAPI** backend plus headless batch scripts for automated VLM-to-VLM experiments.

## Experiment Overview

Two AI agents are paired as the **Director** and the **Matcher** for 5 rounds.

- The Director sees a 2x6 target grid and describes the baskets in left-to-right, top-to-bottom order.
- The Matcher sees a staging area with 18 baskets and reconstructs the Director's 12-basket target sequence.
- The Matcher places baskets by generating a sequence of selections rather than by human clicks.
- The run records chat messages, reasoning logs, selected sequences, accuracy, model configuration, and mutual post-task perceptions.
- Completed sessions are exported with JSON traces, transcripts, and per-round comparison images.

Prompt strategies currently include `ACL_prompt` and `cameron-prompt`. Basket-grid presets live in `referential_task/grids_presetsN.json`.

## Technology Stack

- **FastAPI** and **Uvicorn** for the local simulation server.
- **SQLite** for local run persistence in `data.sqlite`.
- **Jinja2**, **JavaScript**, and **Bootstrap** for the observation dashboard.
- **OpenAI** and **Google Gemini** SDKs for model calls.
- **Pillow** for generating comparison images.

## Setup

```bash
conda create -n langviscog python=3.11
conda activate langviscog
pip install -r requirements.txt
```

Set the API keys needed by the models you plan to run:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

You can also put these values in a local `.env` file; `main.py` and the scripts load it automatically.

## Run The Dashboard

Start the FastAPI app:

```bash
conda run -n langviscog python main.py
```

Open `http://127.0.0.1:8000/`, choose the prompt strategy, basket set, models, and reasoning effort, then start the simulation. Use **Next Turn** or **Auto-Play** to advance the agents.

The dashboard writes each round to `data.sqlite` as it runs. When the final round completes, the app also generates post-task perceptions and auto-exports the session artifacts.

## Run The Paper's Design With A New Model

`scripts/run_replication.py` runs both prompt conditions with the paper's
settings: basket set 5, 5 rounds, cross-round history on, reasoning effort
`none`, the same model as director and matcher, and 10 sessions per condition.
It uses the current game rules (see [Changes Since The Paper](#changes-since-the-paper)),
and only counts sessions run under them.

```bash
python scripts/run_replication.py --model gpt-6-luna --dry-run
mkdir -p logs
nohup caffeinate -i python scripts/run_replication.py --model gpt-6-luna \
  --baseline-metrics "$HOME/Documents/dev/lvlms-referential-game/data/(ACTUAL LATEST EMNLP)metrics_gpt55_latest_all_rounds_reorganized_fixed_identity" \
  > logs/gpt6-luna.log 2>&1 &
```

- Sessions are saved under the prefixes `gpt6-luna-acl` and `gpt6-luna-cameron`, matching the paper's `gpt5.5-acl` naming.
- Runs are resumable: re-running the same command counts the complete exports and only runs the missing sessions. Stalled sessions are topped up automatically (`--max-passes`).
- `--workers` sets the number of parallel sessions per condition (default 2). Each worker has its own SQLite file and log under `data/runs/<prefix>/`.
- At the end, the script prints token totals, then runs `analyze_metrics.py` with GPT-5.5 referring-expression extraction (as in the paper) and plots to `figures/replication_<model>/`.
- `--baseline-metrics` adds the paper's GPT-5.2/5.5 rows from their existing metrics CSV instead of re-extracting them, so the baseline numbers stay as published. Those runs used the paper's rules, so treat the overlay as a rough reference, not a controlled comparison.
- `--max-turns-per-round` sets the turn cap (default 120).

## Run Headless Batches

Use `scripts/run_batch_experiment.py` for repeatable command-line experiments:

```bash
conda run -n langviscog python scripts/run_batch_experiment.py \
  --sessions 5 \
  --session-prefix gpt55-acl \
  --prompt-strategy ACL_prompt \
  --model gpt-5.5 \
  --reasoning-effort none \
  --basket-set 5
```

Useful options:

- `--sessions`: number of complete 5-round sessions to run.
- `--prompt-strategy`: `ACL_prompt` or `cameron-prompt`.
- `--model`: model for both agents.
- `--director-model` and `--matcher-model`: override the shared model per role.
- `--reasoning-effort`: `none` (the paper's setting), `minimal`, `low`, `medium`, `high`, or `xhigh` for GPT-5/GPT-6 models. Not every model accepts every value (gpt-6-luna rejects `minimal`).
- `--session-prefix`: prefix used in session IDs and export folders.
- `--basket-set`: selects `referential_task/grids_presetsN.json`.
- `--cross-round-history` / `--no-cross-round-history`: toggles prior-round text history and submitted-grid feedback images for both agents.
- `--debug-prompt-context`: writes each model call's text context and attached images under `_static/ai_debug/prompt_context/`.
- `--stop-after-round N`: stop each session after round N. Prompts still describe a 5-round session, so rounds 1..N are directly comparable to truncated full-length runs such as the paper's 4-round results. Post-task perceptions are skipped; the export still runs.
- `--db`: SQLite file for live state (default `data.sqlite`). Give parallel batches separate files.
- `--max-turns-per-round`: end a round after this many turns and score the board as it stands (default 120, `0` disables).

Session IDs include the process ID, so parallel batches with the same `--session-prefix` don't collide.

Headless runs enable cross-round history by default and export artifacts when all rounds finish.

### ReAct baseline

`--react` layers a ReAct loop (`referential_task/react.py`) on top of either
prompt strategy. Before each reply, the agent may take private
thought → action → observation steps: `zoom` (enlarged view of 1–4 baskets),
`search_history` (keyword search over the dialogue it can already see), or
`finish`. A final call then produces the strategy's usual structured reply.
Investigation steps never change the board or reveal correctness; feedback
still arrives only at the end of a round.

- `--react-max-steps`: step budget per turn (default 4).
- `--react-step-model`: model for the investigation steps (defaults to each role's model; final replies always use the role's model).
- `--react-roles`: roles that use ReAct, e.g. `matcher` (default `director,matcher`).

Each step is logged under `react_trace` in the round's `ai_reasoning_log`, with token usage per call.
`scripts/track_react_runs.py <export folder>` adds a run to `docs/react_runs.csv`
(one row per session-round); `docs/react_runs.md` is the run log and
`docs/react_baseline_writeup.md` summarizes the design and results so far.

## Data And Exports

Live state is stored in the `game_sessions` table in `data.sqlite`. Each row is one session-round snapshot with config, grid, target baskets, messages, reasoning logs, matcher sequence, status, and perception fields.

Completed sessions are exported under:

```text
data/experiments/<ACL|Cameron|Other>/<YYYY-MM-DD>/<session-prefix>/<session-id>/
```

Each exported session folder contains:

- `<session-id>_data.json`: structured round-by-round trace.
- `<session-id>_transcript.txt`: readable transcript and summary.
- `<session-id>_round_<N>_comparison.png`: Director target sequence vs Matcher reconstruction.

To export all rows from `data.sqlite` into one JSON file, run:

```bash
conda run -n langviscog python scripts/export_to_json.py \
  --output data/exported_sessions.json
```

To regenerate transcript and comparison artifacts for a single exported trace:

```bash
conda run -n langviscog python scripts/export_json_session.py \
  data/experiments/ACL/2026-05-15/gpt55-acl/gpt55-acl_120000_1/gpt55-acl_120000_1_data.json
```

## Run Analysis

The main analysis entry point is `scripts/analyze_metrics.py`. It accepts one or more experiment folders or individual `*_data.json` traces and writes CSV summaries.

Analyze current exported ACL and Cameron sessions:

```bash
conda run -n langviscog python scripts/analyze_metrics.py \
  data/experiments/ACL \
  data/experiments/Cameron \
  --output-dir data/metrics
```

Analyze one trace:

```bash
conda run -n langviscog python scripts/analyze_metrics.py \
  data/experiments/ACL/2026-05-15/gpt55-acl/gpt55-acl_120000_1/gpt55-acl_120000_1_data.json \
  --output-dir data/metrics_single
```

Limit analysis to paper-style first-four-round tables:

```bash
conda run -n langviscog python scripts/analyze_metrics.py \
  data/experiments/ACL \
  data/experiments/Cameron \
  --max-round 4 \
  --output-dir data/metrics_paper_rounds
```

By default, referring expressions are extracted deterministically from Director utterances. To use GPT extraction instead, set `--extractor gpt`; responses are cached under the output directory unless you provide `--gpt-cache-dir`.

```bash
conda run -n langviscog python scripts/analyze_metrics.py \
  data/experiments/ACL \
  data/experiments/Cameron \
  --extractor gpt \
  --gpt-model gpt-5.5 \
  --output-dir data/metrics_gpt55
```

Analysis outputs:

- `metrics_by_session_round.csv`: one row per session round.
- `metrics_by_setting_round.csv`: round-level summary grouped by prompt/model setting.
- `metrics_by_setting.csv`: overall summary grouped by prompt/model setting.
- `referring_expressions.csv`: object-level referring-expression rows with lexical-overlap metrics.

To compare deterministic vs GPT referring-expression extraction:

```bash
conda run -n langviscog python scripts/compare_referring_expressions.py \
  data/metrics/referring_expressions.csv \
  data/metrics_gpt55/referring_expressions.csv \
  --output data/metrics_gpt55/re_extraction_comparison.csv
```

## Model Configuration

For dashboard runs, choose Director and Matcher models in the UI or set:

```bash
export AI_DIRECTOR_MODEL="gpt-5.5"
export AI_MATCHER_MODEL="gpt-5.5"
```

For headless runs, prefer the CLI flags shown above. OpenAI models use `OPENAI_API_KEY`; Gemini models use `GEMINI_API_KEY`.
