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

## Run Headless Batches

Use `scripts/run_batch_experiment.py` for repeatable command-line experiments:

```bash
conda run -n langviscog python scripts/run_batch_experiment.py \
  --sessions 5 \
  --session-prefix gpt55-acl \
  --prompt-strategy ACL_prompt \
  --model gpt-5.5 \
  --reasoning-effort low \
  --basket-set 5
```

Useful options:

- `--sessions`: number of complete 5-round sessions to run.
- `--prompt-strategy`: `ACL_prompt` or `cameron-prompt`.
- `--model`: model for both agents.
- `--director-model` and `--matcher-model`: override the shared model per role.
- `--reasoning-effort`: `none`, `minimal`, `low`, `medium`, or `high` for GPT-5-family models.
- `--session-prefix`: prefix used in session IDs and export folders.
- `--basket-set`: selects `referential_task/grids_presetsN.json`.
- `--cross-round-history` / `--no-cross-round-history`: toggles prior-round text history and submitted-grid feedback images for both agents.
- `--debug-prompt-context`: writes each model call's text context and attached images under `_static/ai_debug/prompt_context/`.

Headless runs enable cross-round history by default and export artifacts when all rounds finish.

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
