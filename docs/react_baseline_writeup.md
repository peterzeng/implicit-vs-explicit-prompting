# ReAct baseline for the basket matching task: writeup

Branch: `react-baseline` (off `main`, which stays as the EMNLP 2026 paper's
cited code). Started 2026-10-08.

## Why a ReAct baseline

The NSF proposal's multi-agent system (MAS) puts explicit state into a
Reasoning Agent's context through a Common Ground Agent and a Reflection
Agent. ReAct is the single-agent version of the same idea: one model that
gathers that information itself through tools, using interleaved
thought → action → observation steps. Given the same tools and information,
MAS vs. ReAct isolates the effect of splitting the work across agents, which
is a cleaner comparison than MAS vs. the paper's one-call prompts.

## What was built

**ReAct as a switch on top of the existing instruction strategies.** Either
`ACL_prompt` (implicit) or `cameron-prompt` (explicit) supplies the
instructions and the final reply schema, so the choice of instructions stays
open. Before each reply, an agent may take up to N private steps:

| Action | Observation |
|---|---|
| `zoom` | Enlarged side-by-side image of 1–4 baskets (matcher: candidates 1–18; director: target positions 1–12) |
| `search_history` | Keyword search over the dialogue the agent can already see, including earlier rounds |
| `finish` | Ends the investigation |

A final call then produces the usual `{reasoning, utterance, selection}`, so
the turn loop, board updates, scoring, exports and analysis scripts are
unchanged.

**Design constraints:**
- Investigation steps never change the board and never reveal correctness.
  Feedback still arrives only at the end of a round.
- At most one placement per turn, so turn counts stay comparable to the paper.
- The step model is configurable (`--react-step-model`): it can be smaller
  than the role's model. The final reply always comes from the role's model.
- ReAct can be limited to one role (`--react-roles matcher`).

**Run tracking:**
- Token usage (input, cached input, output, reasoning) is recorded for every
  call, split into ReAct steps vs. final replies.
- `scripts/track_react_runs.py` turns exported sessions into one row per
  session-round in `docs/react_runs.csv`, for programmatic analysis.
- `docs/react_runs.md` is the human-readable log: commands, purpose, results,
  and observations from the traces.
- Paper metrics (referring-expression length, lexical overlap) still come from
  `scripts/analyze_metrics.py` on the same export folders.

**Fixes made along the way:**
- The batch script now creates its SQLite table. Before, it only worked if
  `main.py` had already been run once.
- The OpenAI client timeout is 180s instead of the SDK's 600s, so a dropped
  connection (network switch, sleep) stalls a run for 3 minutes, not 10.

**Files:** `referential_task/react.py` (new), `referential_task/ai_vs_ai.py`
(hook before the final call, logging), `referential_task/ai_providers.py`
(token usage, timeout), `scripts/run_batch_experiment.py` (flags, DB setup),
`scripts/track_react_runs.py` (new), `tests/test_react.py` (new), `README.md`.

## How to run

```bash
python scripts/run_batch_experiment.py \
  --sessions 1 --session-prefix react-smoke --prompt-strategy ACL_prompt \
  --model gpt-5.5 --reasoning-effort low \
  --react --react-roles matcher --react-step-model gpt-5-nano

python scripts/track_react_runs.py data/experiments/ACL/<date>/react-smoke
```

For long batches, launch with `nohup caffeinate -is ...` so the run survives
closing Claude Code and keeps the Mac from idle-sleeping. A closed lid still
pauses it.

## Smoke test (one session)

Implicit prompt, `gpt-5.5` for both roles, ReAct on the matcher with
`gpt-5-nano` steps. Details are in `docs/react_runs.md`.

- **Everything works end to end.** 5/5 rounds completed and exported, all 137
  steps parsed, and usage was recorded for all 265 calls.
- **Accuracy was 90.0% on average** (75, 100, 83, 92, 100). The paper's
  GPT-5.5 implicit condition without ReAct scored 100% in every round across
  10 sessions.
- **The nano steps mostly scan in fixed blocks** (candidates 1–4, 5–8, ...)
  rather than zooming on likely matches. The chosen candidate was among those
  zoomed in only 20 of 52 committing turns, and in 5 of the 6 errors the
  wrong pick was one the scan had just shown.
- **History search was almost unused** (2 searches), and keyword matching on
  any word returned half the dialogue.
- **Cost grows quickly.** 2.76M tokens for the session, rising from 291k to
  784k per round, because every step resends the full context with images.

## What this means for the baseline

1. **GPT-5.5 leaves no headroom on accuracy.** It is at ceiling without ReAct,
   so ReAct (or the MAS) can only lose accuracy there. A meaningful comparison
   needs a setting with headroom: a weaker model (the paper's GPT-5.2 implicit
   condition starts at 90.8% in round 1), harder basket sets, or human-AI play.
   Efficiency (words, turns) remains comparable for GPT-5.5.
2. **A tiny step model changes what is being tested.** With `gpt-5-nano`, the
   steps are mostly uninformative and possibly harmful. The next run should
   use the role's own model for steps, to see whether ReAct itself helps.
3. **The search tool needs work** to be useful, for example ranking by phrase
   overlap and returning fewer, more relevant messages.

## Open decisions

- Implicit vs. explicit instructions for ReAct (pending advisor discussion).
- Which model(s) and settings give headroom for the MAS comparison.
- Matcher-only vs. both roles.

## Suggested next runs

Each costs real API usage, so confirm the batch size before launching.

1. Same smoke setup with steps on `gpt-5.5` (no `--react-step-model`): does
   ReAct with a capable step model match the 100% baseline?
2. A weaker model with headroom (e.g., GPT-5.2), with vs. without ReAct.
3. Multiple sessions per condition once the configuration is settled (the
   paper used 10).
