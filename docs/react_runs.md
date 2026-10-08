# ReAct baseline: run log

Human-readable log of ReAct runs on the `react-baseline` branch. The
per-round numbers live in [`react_runs.csv`](react_runs.csv), one row per
session-round, generated from exported sessions with:

```bash
python scripts/track_react_runs.py data/experiments/<ACL|Cameron>/<date>/<session-prefix>
```

Re-running the script on the same session updates its rows rather than
duplicating them. Paper metrics (referring-expression length, lexical
overlap) come from `scripts/analyze_metrics.py` on the same export folders.

Each entry below records the command, what we were testing, the headline
numbers, and anything odd seen in the `react_trace` logs.

---

## 2026-10-08 · react-smoke · first live ReAct session

**Status:** complete (5/5 rounds, exported and in `react_runs.csv`).

**Purpose:** check that real models follow the ReAct step format and that
everything (steps → final reply → board update → export) works end to end.
Not a comparison run: one session only.

**Command:**
```bash
python scripts/run_batch_experiment.py \
  --sessions 1 --session-prefix react-smoke --prompt-strategy ACL_prompt \
  --model gpt-5.5 --reasoning-effort low \
  --react --react-roles matcher --react-step-model gpt-5-nano
```

**Session:** `react-smoke_121834_1`

**Results** (one session, so no error bars):

| Round | Accuracy | Turns | Words | ReAct steps (zoom / search / finish) | Total tokens |
|---|---|---|---|---|---|
| 1 | 75.0% | 30 | 1003 | 10 / 0 / 15 | 291k |
| 2 | 100.0% | 24 | 788 | 14 / 0 / 12 | 418k |
| 3 | 83.3% | 26 | 858 | 17 / 1 / 13 | 609k |
| 4 | 91.7% | 24 | 785 | 15 / 0 / 12 | 656k |
| 5 | 100.0% | 24 | 772 | 15 / 1 / 12 | 784k |

Mean accuracy 90.0%. For reference, the paper's Implicit Prompt / GPT-5.5
condition without ReAct (10 sessions, `emnlp-5rounds/`) scored 100% in every
round, with about 700 words and 26 turns per round. Reasoning effort for those
runs is not recorded in the summary file; this run used `low`.

Session totals: 265 calls, all with token usage recorded. 2.76M tokens:
- `gpt-5-nano` steps: 1.55M input (0.80M cached), 73k output.
- `gpt-5.5` final replies: 1.11M input (0.38M cached), 24k output.

Tokens per round grow from 291k to 784k as history and feedback images
accumulate, and every ReAct step resends that context.

**Cost: about $4.64** at OpenAI standard prices as of 2026-10-08
(`gpt-5.5`: $5.00 / $0.50 cached / $30.00 output per 1M; `gpt-5-nano`:
$0.05 / $0.005 / $0.40):
- `gpt-5.5` final replies: $4.57 ($3.66 uncached input, $0.19 cached, $0.72 output).
- `gpt-5-nano` steps: $0.07.
- Excludes the two end-of-session perception calls (not token-logged, text only).

The largest single call was 17.7K input tokens, well under the 272K
long-context threshold. Estimates: about $11/session with steps on `gpt-5.5`
instead of nano, and about $45 (nano steps) or $110 (`gpt-5.5` steps) for 10
sessions.

**What the traces show:**
- **The step format works.** `gpt-5-nano` produced 137 steps with no
  malformed output and no invalid basket numbers.
- **The steps mostly scan rather than investigate.** A typical turn is one
  zoom then finish. A third of zooms (24/71) were candidates 1–4, and most
  others were fixed blocks (5–8, 9–12). The candidate finally chosen was
  among the zoomed ones in only 20/52 committing turns.
- **Errors may be primed by the zooms.** In 5 of the 6 wrong placements, the
  wrong candidate was one the scan had just zoomed on. Only 6 cases, so this
  is a hypothesis to test, not a finding.
- **History search is barely used and not selective.** 2 searches in 5 rounds.
  Both queries were whole descriptions and matched half of all messages
  (keyword matching on any word).

**Run notes:**
- The first launch failed before any API calls: the batch script assumed the
  `game_sessions` table already existed (only `main.py` created it). Fixed by
  having the batch script create the table on startup.
- Switching networks mid-run (to a hotspot) stalled one request in round 4 on
  a dead connection. The OpenAI client times out after 10 minutes and then
  retries, and the run resumed by itself at 12:48.
- The laptop then slept on and off from 13:13 to 16:00, and the last request
  hung on another dead connection (different network). The run resumed after
  wake and finished at 16:04. Wall-clock time is not meaningful for this run.
- Since then, the OpenAI client timeout is 180s instead of the SDK's 600s
  (`OPENAI_TIMEOUT_SECONDS` to override), so a dead connection costs about
  3 minutes.
