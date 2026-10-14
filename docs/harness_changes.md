# Game-rule changes since the EMNLP 2026 paper

This file records every change to the game rules (the "harness") made after
the paper. It also covers why each change was made, the evidence, and the
alternatives we rejected. The `main` branch is the code as published.

Every run stores the rules it used as `harness_version` in its config:

| Version | Status | Rules |
|---|---|---|
| 1 (or absent) | EMNLP 2026 paper | Moving a placed basket requires a keyword in the matcher's chat message. Incomplete submits are silently refused. No turn limit. |
| 2 | Interim, 2 test sessions (2026-10-09) | Moves use `selection.move`. Ignored selections are reported to the matcher. 120-turn cap per round. |
| 3 | Current | Version 2, plus refused submits are reported, and the matcher's board-state note is the last message in its context. |

Do not pool results across versions. `analyze_metrics.py` groups its
summaries by `harness_version`, and `run_replication.py` only counts sessions
run under the current version.

## Background: how a round ends

The matcher places one basket per turn through a structured `selection`
(candidate number 1–18, position 1–12, and `ready_to_submit`). The round ends
when the matcher submits a full board. Before the paper, the end-of-round
logic was patched several times as models lost track of the board. The commit
messages do not describe these changes, so the dates below come from
`git log -S` in the original repo
([lvlms-referential-game](https://github.com/peterzeng/lvlms-referential-game)):

| Date | Patch |
|---|---|
| 2026-03-02 | "STILL EMPTY" positions in the matcher prompt, and regex detection of "I'm submitting" in chat. |
| 2026-04-08 | "Pending refill" note when a move leaves a slot empty. |
| 2026-05-13 | "ALL 12 POSITIONS ARE FILLED — set ready_to_submit to true now." |
| 2026-05-15 | **Incomplete submits refused** (before this, any board with at least one basket could be submitted, with empty slots scored wrong), and **moves gated on keywords**. |

The 2026-05-15 commit is the one that matters here. The keyword rule's
docstring gives its purpose: to prevent loops "where the model says it is
filling the last slot but reuses a candidate already sitting elsewhere,
causing positions to alternate between filled and empty." Our reading, which
the commit does not state: accidental moves were emptying slots just before
submission, so refusing incomplete submits and blocking accidental moves
were introduced together. Both fail silently: the matcher is not told that
its move or its submit was ignored.

The paper's runs used these rules. They are also close to the human
interface: in the `human-human` branch, the submit button is hidden until all
12 slots are filled (`selectedSequence.length === 12`) and then asks for
confirmation.

## Problem 1: deliberate moves blocked (fixed in version 2)

Found while re-running the paper's design with gpt-6-luna (2026-10-09,
reasoning effort `none`, basket set 5, 3 sessions per prompt planned).

**What happened.** Under the explicit prompt, 3 of 4 sessions got stuck in a
round that could never end (158, 482, and 250 turns before we stopped them;
one more round recovered after 154 turns). The paper's longest round was 46
turns. The sequence:

1. The matcher moved a basket with a keyword ("that one was basket 12, not
   4"), which left a slot empty.
2. The basket that belonged in the empty slot was already placed elsewhere, so
   filling it took another move.
3. The matcher chose that move correctly in its structured selection, but its
   message ("All set, thanks!") had no keyword. The move was ignored, nothing
   told the matcher, and the board stayed at 11/12, so it could not be
   submitted.

**Evidence.** `scripts/audit_move_attempts.py` replays every matcher selection
through the placement code. Every replayed board matches the recorded one.

| Data | Rounds | Move attempts | Blocked | Blocked moves to the correct slot |
|---|---|---|---|---|
| Paper, GPT-5.2 implicit | 50 | 1 | 1 | 0 |
| Paper, GPT-5.2 explicit | 50 | 5 | 5 | 4 |
| Paper, GPT-5.5 (both prompts) | 100 | 0 | 0 | — |
| gpt-6-luna implicit, v1 | 15 | 0 | 0 | — |
| gpt-6-luna explicit, v1 | 11 | 102 | 99 | 97 |

**Why only the explicit condition.** The implicit (ACL) matcher prompt says:
"If you decide that an earlier guess was wrong and you want to move a basket
from one position to another, you must say so explicitly in your utterance."
The explicit (Cameron) prompt has no such instruction and asks for very short,
casual messages. That combination likely explains why only the explicit
condition got stuck; we have not tested it directly.

**Effect on the paper's results: none found.** GPT-5.x never made a move
that went through (step 1 above), so it never got stuck. For the 5 paper
rounds with blocked moves, re-scoring with the moves allowed gives the same or
lower accuracy in every round (`audit_move_attempts.py --rounds`): each move
would also have emptied the basket's old slot. This is a rough counterfactual,
because the dialogue after an allowed move would have been different.

**What we did with those runs.** The stuck sessions were stopped and never
exported. Their full state is kept in local worker SQLite files under
`data/runs/gpt6-luna-cameron/` (not in git; see `DISCARDED.md` there), which
the audit script reads directly:

| Session | Where it got stuck |
|---|---|
| `gpt6-luna-cameron_153701_74888_1` | Round 1, 158 turns, 11/12 filled |
| `gpt6-luna-cameron_154254_75535_1` | Round 1, 482 turns, 11/12 filled |
| `gpt6-luna-cameron_154254_75536_1` | Round 4, 250 turns, 11/12 filled (rounds 1–3 complete) |

The one explicit session that finished (`gpt6-luna-cameron_153701_74889_1`)
needed 154 turns in round 4. The 3 implicit sessions finished without
problems. None of these version-1 sessions count toward version-3 results.

**Changes in version 2.**

- **`selection.move`.** Both response schemas (`prompts/acl.py`,
  `prompts/cameron.py`) have a `move` boolean, and both prompts explain it. A
  placed basket moves only when `move` is true. Picking a placed basket with
  `move: false` is still ignored, which keeps the original loop prevented.
  The keyword check (`_utterance_explicitly_moves_selection`) is removed. The
  ACL prompt's "say so explicitly in your utterance" now reads "set
  `selection.move` to true and say so in your utterance".
- **Ignored selections are reported.** If the matcher's last selection was not
  applied, its private board-state note says so and why, e.g. "candidate 8 is
  already in position 4 … To move it, set `move` to true". The director never
  sees this. Each selection's outcome is logged as `selection_result` in the
  round's `ai_reasoning_log`.
- **Turn cap.** A round ends after `max_turns_per_round` turns (default 120;
  `0` disables). The board is scored as it stands, with empty slots counted
  wrong, and the round's status gets `turn_cap_reached: true`.
  `analyze_metrics.py` reports the flag per round and the capped fraction per
  setting. Capped rounds have inflated turn and word counts, so report effort
  metrics with and without them.

**Version 2 test (2 explicit sessions).** One session finished all 5 rounds
normally and used `move` successfully in round 4. The other hit the cap in
round 1 (50% accuracy) through a different failure, described next.

## Problem 2: refused submits are silent, and the board state is easy to miss (fixed in version 3)

**What happened** (session `gpt6-luna-cameron_164408_80070_1`, round 1):

1. A selection was correctly rejected (the basket was already placed). The
   matcher read the new note and recovered on its next turn.
2. It moved the cat basket from slot 5 to 6 with `move: true`. That worked
   and left slot 5 empty.
3. In the same message it said it was placing the next basket (slot 7), but a
   turn carries one selection, so slot 7 was never filled.
4. It placed slots 8–12, decided the board was complete, and for about 48
   matcher turns sent "All set, I'm submitting" with no basket. Each submit
   was refused because only 10 of 12 slots were filled, and nothing told it so.

Its board-state note said "STILL EMPTY: 5, 7" on every turn, and it also saw
an image of its board with the empty slots outlined. That note was placed at
the very top of its context, above the whole conversation (about 100 messages
and several images by then).

**Changes in version 3.**

- **Refused submits are reported.** The board-state note now says, e.g.
  "YOUR SUBMIT WAS NOT ACCEPTED: position(s) 5, 7 still empty. The sequence
  can only be submitted once all 12 positions are filled." Logged as
  `submit_result` in `ai_reasoning_log`. This is the text equivalent of the
  human interface, where the submit button is visibly missing.
- **Board state goes last.** The board-state note (including these reports)
  is now the final message in the matcher's context, after the director's
  latest message, instead of the first message after the instructions. The
  reasoning: in a long context, the most recent message is more likely to
  inform the next action. This is a hypothesis; see open questions.

## Alternatives considered and rejected

| Option | Why not |
|---|---|
| Accept an incomplete submit if the matcher insists | People can't submit an incomplete board in the human interface, so this would move AI–AI results away from the human comparison. The turn cap covers the cases where a round would otherwise never end. |
| Several placements per turn (`selection` as a list) | Would remove the "said two, placed one" mismatch, but changes the turn structure that the turns and words metrics depend on, and moves away from the human comparison. |
| Swap baskets when moving onto a filled slot | Treats a symptom: moving into an empty slot still empties another, and it does not address silent failures. |
| Check that the chat message matches the selection | Infers intent from casual wording, which is exactly the fragility of the keyword rule. |
| Go back to the pre-May-15 rules (submit anything) | Re-opens the problem those rules fixed and departs from the human interface. |

## Open questions

- **Is moving the board state last what helps, or the submit reports?** Both
  changed together in version 3, so their effects are confounded. An ablation
  (version 3 with the note at the top) would separate them.
- **Turn cap semantics.** Scoring empty slots as wrong is a choice; analyses
  could also exclude capped rounds. Both are reported so either is possible.
- **Comparability with the paper.** Versions 2 and 3 change what the matcher
  sees. Plots that overlay the paper's GPT-5.2/5.5 results
  (`run_replication.py --baseline-metrics`) compare across versions and are a
  rough reference, not a controlled comparison. A controlled comparison
  needs GPT-5.x re-run under version 3.
- **The director has no view of the matcher's board.** A human director could
  ask "what's in slot 5?" and get an answer from a person looking at the
  board. In the AI–AI game, neither agent can check the other's board.

## Results under version 3

3 sessions per prompt with gpt-6-luna on 2026-10-14, using the paper's
settings (reasoning effort `none`, basket set 5, 5 rounds). The 6 sessions ran
in parallel in about 10 minutes and cost about $0.45.

Accuracy (%) and turns per round:

| Prompt | Session | Accuracy, rounds 1–5 | Turns, rounds 1–5 |
|---|---|---|---|
| Implicit | 1 | 100 / 100 / 100 / 100 / 100 | 26 / 28 / 26 / 28 / 28 |
| Implicit | 2 | 100 / 100 / 100 / 100 / 92 | 28 / 26 / 30 / 26 / 26 |
| Implicit | 3 | 100 / 100 / 100 / 100 / 92 | 26 / 28 / 26 / 26 / 26 |
| Explicit | 1 | 50 / 58 / 50 / 58 / 67 | 26 / 24 / 36 / 30 / 26 |
| Explicit | 2 | 83 / 33 / 83 / 67 / 92 | 28 / 32 / 26 / 26 / 26 |
| Explicit | 3 | 92 / 92 / 100 / 100 / 92 | 28 / 24 / 26 / 28 / 28 |

- **No round reached the cap.** The longest round was 36 turns, within the
  paper's range (max 46). Under version 1, 3 of 4 explicit sessions got stuck.
- **The new feedback was used and worked.** In explicit session 1, the matcher
  had 2 selections ignored and 1 submit refused; it recovered each time
  within the round. Explicit session 2 moved baskets twice with `move: true`.
  The implicit sessions never needed either.
- **The accuracy gap between prompts is not a harness effect.** Explicit
  session 1 scored 50–67% in every round without any loop: it placed full
  boards with wrong baskets. This is gpt-6-luna's performance under the
  explicit prompt, not something the rules caused.
- **Caveats.** Three sessions per prompt is a small sample, and the version-3
  changes were introduced together, so we can't say which one prevented the
  loops (see open questions).

## Reproducing the numbers

```bash
# Move attempts in the paper's runs (traces from the original repo's emnlp branch)
python scripts/audit_move_attempts.py ~/Documents/dev/lvlms-referential-game/data/experiments --rounds

# gpt-6-luna version-1 runs, including the unexported stuck sessions
python scripts/audit_move_attempts.py data/runs/gpt6-luna-cameron/worker*_20261009-15*.sqlite \
  data/runs/gpt6-luna-acl/worker*_20261009-153700.sqlite
```

### GPT-5.2 and GPT-5.5 under version 3

To check whether the new rules change GPT-5.x's results, and to compare
models under identical rules: 2 sessions per prompt for each model
(2026-10-14, same settings, about $16 in total).

| Model | Prompt | Sessions | Mean accuracy by round (%) | Director words/turn |
|---|---|---|---|---|
| GPT-5.2 | Explicit | 2 | 100 / 100 / 100 / 100 / 100 | 13 |
| GPT-5.2 | Implicit | 2 | 100 / 100 / 96 / 92 / 92 | 43 |
| GPT-5.5 | Explicit | 2 | 96 / 96 / 100 / 100 / 100 | 4 |
| GPT-5.5 | Implicit | 2 | 100 / 100 / 100 / 100 / 100 | 28 |
| gpt-6-luna | Explicit | 3 | 75 / 61 / 78 / 75 / 84 | 8 |
| gpt-6-luna | Implicit | 3 | 100 / 100 / 100 / 100 / 95 | 32 |

- **Version 3 does not change GPT-5.x's results.** Both score 92–100%, in
  line with the paper. None of their rounds triggered a move, an
  ignored-selection note, a refused-submit note, or the cap; the longest round
  was 32 turns.
- **gpt-6-luna's explicit-prompt drop is not caused by the rules.** It is a
  combination of model and prompt: gpt-6-luna is near ceiling with the
  implicit prompt and about 20 points lower with the explicit one.
- **Director brevity alone does not explain it.** GPT-5.5's explicit-prompt
  directors are terser than gpt-6-luna's (4 vs 8 words per turn) and GPT-5.5
  still scores ~100%. Two explanations remain: gpt-6-luna's matcher handles
  short descriptions worse, or it benefits from the implicit prompt's
  required reasoning fields (likely confusions and a question to tell them
  apart). Mixed-model pairs (gpt-6-luna director with a GPT-5.5 matcher and
  vice versa) would tell whether the director or the matcher is the weak link.
- Sample sizes are small (2–3 sessions per condition).

### Tier vs generation: gpt-5.6-luna and gpt-6-sol

gpt-6-luna is priced as a budget model ($0.10/$0.50 per 1M input/output
tokens vs GPT-5.5's $5/$30), so comparing it with GPT-5.5 mixes generation
with tier. To separate them (2026-10-14, 3 sessions per prompt, same
settings, about $9): **gpt-5.6-luna** keeps the budget tier and changes the
generation; **gpt-6-sol** ($2/$10) keeps the generation and moves up a tier.
Tiers are inferred from pricing; OpenAI does not publish model sizes. Both
models passed the parameter check (reasoning effort `none`, structured
output, images).

| Model | Tier (by price) | Sessions | Implicit, overall accuracy | Explicit, overall accuracy | Explicit director words/turn |
|---|---|---|---|---|---|
| GPT-5.2 | middle | 2 | 96% | 100% | 13 |
| GPT-5.5 | top | 2 | 100% | 98% | 4 |
| gpt-6-sol | middle | 3 | 100% | 97% | 6 |
| gpt-5.6-luna | budget | 3 | 97% | 68% | 5 |
| gpt-6-luna | budget | 3 | 99% | 74% | 8 |

gpt-5.6-luna explicit sessions by round: 75/75/75/58/67, 67/58/67/75/75,
75/58/75/75/50. gpt-6-sol explicit: 92/92/100/100/100, 92/92/92/100/100,
100/100/100/100/100.

- **The explicit-prompt accuracy drop is a budget-tier effect, not a GPT-6
  effect.** Both budget models drop 25–30 points under the explicit prompt,
  and the mid-tier gpt-6-sol does not. Comparing gpt-6-luna with GPT-5.x and
  calling the difference a generation effect would have been wrong.
- **All models are near ceiling with the implicit prompt.**
- **Director brevity alone does not explain the drop.** gpt-6-sol's explicit
  directors are about as terse as the budget models' (6 vs 5–8 words per
  turn), and it still scores 97%. This points to the budget models' matchers.
  The mixed-model pairs described above would test that directly.
- **No round hit the turn cap** (longest 44 turns). gpt-5.6-luna made 1 move
  with `move: true`, and gpt-6-sol none.
