# Plan: human-likeness metrics, open-model baseline, fine-tuning

Status as of 2026-10-15: planning only. No open-model runs yet; Empire AI
access is pending. Nothing in this file has been implemented. For the game
rules and the GPT results so far, see `harness_changes.md`.

## Goal

**The goal is AI that collaborates well with people.** In this task, a good
partner:

- picks up the human's terms;
- asks for clarification when unsure, and not otherwise;
- settles on shared, efficient names across rounds the way people do;
- recovers from misunderstandings;
- leaves the human with less work and a sense of being understood.

Accuracy in AI–AI play is near ceiling for mid- and top-tier models under
harness version 3 (see `harness_changes.md`). It says little about any of
these.

**Human-likeness in AI–AI play is a stand-in for that goal, not the goal
itself.** We use it because:

- human–human play is the best reference we have for how successful partners
  communicate;
- AI–AI sessions are cheap enough to iterate on.

**Two cautions follow:**

- **Not everything human is worth copying.** Some human behavior doesn't help
  a partner, such as errors or off-task chat. Metrics should favor what
  helps the partner: adaptation, useful clarification, conventions the
  partner can follow.
- **AI–AI optimization can produce shorthand only AIs understand.** Self-play
  in particular can do this. So a model counts as improved only once it does
  better **with human partners** (see "Evaluation with people" below).

Plan, agreed with Owen:

1. Baseline: a large open vision-language model served with vLLM on Empire
   AI, with no fine-tuning.
2. Supervised fine-tuning (SFT) on the human–human transcripts.
3. Filtered self-play (rejection sampling or DPO) using the game engine.
4. Online RL, only if step 3 stops improving.
5. Evaluation with people, in both roles.

## Human–human data

**Source:** `data/consolidated_pairs.csv` on the `human-human` branch of
[lvlms-referential-game](https://github.com/peterzeng/lvlms-referential-game).
It has 34 rows: 32 valid pairs and 2 empty rows.

| Property | Value |
|---|---|
| Pairs | 32 valid |
| Rounds per pair | 4 (AI–AI runs use 5, so compare rounds 1–4 only) |
| Basket set | 5, the same 18 baskets as `fullList` in `referential_task/grids_presets5.json` |
| Chat | `round{k}_director_grid_messages` and `round{k}_matcher_grid_messages`. Both logs can hold the same messages, so merge and de-duplicate on (text, sender_role, timestamp). |
| Boards | Only the final board of each round, in `round{k}_matcher_selected_sequence`. Placements have no timestamps, and moves can't be recovered. |

**Use each pair's recorded `round{k}_shared_grid`, not the preset file.** In
the human–human games, two pairs of positions are swapped relative to
`grids_presets5.json`: round 2 positions 11–12 and round 3 positions 10–11.

**`matcher_sequence` and `matcher_selected_sequence` disagree in 4
pair-rounds.** Use `matcher_selected_sequence`, the player-level field.

**The Drive folder** ("LVLMs-and-Humans-Ground-Differently") has the same
chat transcripts as .txt files, but no grids or boards. One file there is
useful: `EntrainmentCodingExamples-ByHand/Human-Human entrainment.xlsx`. It
codes the expressions for each basket across rounds, with error tags, for
2 pairs. Use it to check the RE extractor on human text.

## Preliminary comparison

These metrics are deterministic (no LLM extraction). The scripts were
throwaway and are not in the repo; the definitions below are enough to
recompute the numbers. Rounds 1–4. The AI rows are every v3 session in
`data/experiments`.

| Metric | Humans, R1 → R4 | AI, implicit (ACL) prompt | AI, explicit (Cameron) prompt |
|---|---|---|---|
| Director words per message | 12.6 → 9.6 | 29–49, flat | 3–19 |
| Matcher share of all words | 0.27 → 0.19 | 0.39–0.48 | 0.50–0.76 |
| Director vocabulary overlap with previous round (Jaccard on content-word types) | 0.37 → 0.53 | 0.71–0.95 | 0.63–0.98 |
| Director words, round 4 ÷ round 1 (median per session) | 0.58 (middle half 0.35–0.90) | 0.88–1.04 | 0.53–0.91 |
| Matcher messages containing "?" per round | 5.3 → 1.2 | 0–7 | 4.7–11.7, flat |

What the numbers show:

- **The biggest gap is the matcher.** AI matchers repeat the description
  back, e.g. "Placing the squat rectangular basket with the chunky
  twisted-rope handle in position 5. What's basket 6?". Human matchers
  mostly say "ok" or "got it", or ask a real clarification question.
- **Humans rephrase across rounds; models repeat almost word for word.**
- **"?" counts mean different things.** Most AI question marks are "what's
  basket N?", not clarification requests. Clarification needs a
  dialogue-act labeler.
- **The rules fix message counts.** The harness enforces strict turn-taking
  and requires the matcher to speak every turn. Humans send several messages
  in a row and often place baskets without saying anything. So director and
  matcher message counts are set by the rules and can't be used as metrics.

## Proposed metrics

All metrics are computed per round, rounds 1–4, and separately for the
director and the matcher where that applies.

| # | Metric | Definition | Needs |
|---|---|---|---|
| 1 | RE-length curve | RE words per round (already in `analyze_metrics.py`) | gpt-6-sol extraction of the human data |
| 2 | Compression ratio | Director words, round 4 ÷ round 1, per session | — |
| 3 | Director words per message | Mean over director messages | — |
| 4 | Lexical overlap per basket | Overlap with the same basket's expression in the previous round (already in `analyze_metrics.py`) | Extraction |
| 5 | Convention convergence | The first round after which a basket's expression stays stable (overlap ≥ θ), and how much of the final round's expression already appeared in round 1 | Extraction |
| 6 | Matcher share of words | Matcher words ÷ all words | — |
| 7 | Matcher dialogue acts | Rates of acknowledgement, clarification question, echoing the description back, and next-item request | An LLM labeler, checked against a hand-labeled sample |
| 8 | Entrainment between partners | Share of the director's words for a basket that the matcher introduced earlier for that same basket. Counting words over a whole round gives about 0 for everyone, so measure it per basket. | Extraction |
| 9 | Recovery after an error | How the next round's expression for a basket changes after it was misplaced: length change and overlap. The 2 hand-coded pairs record errors, which makes this checkable. | Extraction |

Effort metrics such as turns and duration are not human-likeness metrics.
Human rounds take 7–20 minutes, AI rounds 1–3 minutes, and turn counts are
fixed by the rules.

To get the extraction-based metrics, convert `consolidated_pairs.csv` into
the `*_data.json` trace format. It already uses the same `shared_grid` and
message fields. Then run `analyze_metrics.py --extractor gpt --gpt-model
gpt-6-sol` unchanged, so humans and models go through the same extractor.

## Scoring distance to humans

For each metric *m* and round *r*:

- **Distance:** the Wasserstein-1 distance between the model's
  session-level values and the 32 human pair values, divided by the human
  interquartile range. Comparing whole distributions, not means, catches a
  model that is right on average but has too little variance (the AI
  sessions are very uniform).
- **Coverage:** the share of model sessions that fall inside the middle 90%
  of the humans.

Calibrating and summarizing:

- **Human noise floor.** Split the 32 pairs in half at random 1,000 times and
  compute the same distance between the halves. Report model distances
  relative to this floor. With n=32 the floor is far from zero, and
  differences below it mean nothing.
- **Summary score.** Average the distances over metrics, with each metric
  scaled by its noise floor. Always report the per-metric numbers too, since
  an average can hide a model that gets brevity right and the matcher badly
  wrong.
- **Classifier test.** Train a simple classifier to tell human sessions from
  model sessions on the metric vector, and report cross-validated AUC.
  0.5 means indistinguishable. Later, a text-level classifier or an LLM
  judge could do the same.
- **Score director and matcher separately.** The Human-AI and AI-Human
  folders on the Drive could let us score one role against a human partner.
- **Don't optimize on the evaluation humans.** Once metrics are used as a
  reward (self-play), score against a held-out half of the human pairs.

## Evaluation with people

AI–AI metrics are used to develop models; human–AI sessions are the test.
What we already have:

- **The paper's human–AI data** (public Drive folder
  "LVLMs-and-Humans-Ground-Differently"). These are the baseline a fine-tuned
  model has to beat.
  - `human-ai-pairs.xlsx`: 22 pairs, human director with an AI matcher.
  - `ai-human-pairs.xlsx`: 17 pairs, AI director with a human matcher.
  - The `*-with-times.xlsx` versions add timing columns.
  - Both use basket set 5 in the human–AI/AI–AI order, which matches
    `grids_presets5.json` for rounds 1–4.
  - The AI model is not recorded in these files; get it from the paper.
- **Partner ratings.** All three conditions ask the same post-task survey
  (checked 2026-10-15). It has 5 items on a 1–5 scale:
  - `partner_capable`
  - `partner_helpful`
  - `partner_understood`
  - `partner_adapted`
  - `collaboration_improved`

  It also has `partner_human_vs_ai` (0–100), a free-text comment, and AI
  familiarity. Only the human's columns are filled: the `director_*`
  columns in human–AI and the `matcher_*` columns in AI–human.

  | Item | Human–human, director / matcher (n=32) | Human director rating AI matcher (n=22) | Human matcher rating AI director (n=17) |
  |---|---|---|---|
  | `partner_capable` = 5 | 27 / 27 | 2 | 3 |
  | `partner_adapted` = 5 | 27 / 24 | 3 | 3 |
  | `partner_human_vs_ai` = 0 | 0 / 0 | 14 | 13 |

  People rate human partners near the top of the scale. They rate AI
  partners low and mostly identify them as AI. There is plenty of room for
  improvement, and the ratings are a direct outcome measure.

What to measure with human partners:

| Measure | What it tells us |
|---|---|
| Accuracy | Whether the pair succeeds |
| The human's effort: words, turns, time per round | Whether the AI makes the task easier |
| Whether the human's own expressions compress across rounds | Whether a shared convention forms with the human, not just inside the AI |
| Whether the AI takes up the human's terms | Adaptation to this particular partner |
| Partner ratings | How the collaboration felt to the human |

Running a human–AI study means a new data collection with the fine-tuned
model behind the human interface. Peter will raise the IRB question with
Susan. Plan for this early. It is the
step that decides whether the project worked.

## Open model choice

- **Baseline:** Qwen3-VL-235B-A22B-**Instruct** (mixture-of-experts, 22B
  active parameters).
- **Fine-tuning:** Qwen3-VL-32B-Instruct, with an 8B dry run first.
- Use the Instruct variants, not Thinking. Thinking models emit reasoning
  text, which interferes with structured JSON output.

## Empire AI

Source: the public knowledge base at
`https://empireai.freshdesk.com/support/solutions/articles/<id>`.

Checked against the knowledge base on 2026-10-15:

- **Alpha+ hardware (157000363466):** 18 nodes with 8× H100-80GB and 6 nodes
  with 8× H200-141GB, x86 CPUs, 2TB RAM per node.
- **Alpha queues (157000374474):** one `alpha` partition, and `--account` is
  required.

  | QoS | Walltime | GPU limit | Cost |
  |---|---|---|---|
  | test | 2h | 8 GPUs, 1 node | 0.5× |
  | standard | 48h | 32 GPUs | 1× |
  | long | 7 days | 32 GPUs | 0.5× |
  | priority | 24h | 64 GPUs | 2× |
  | burst | 7 days | 32 GPUs | free, preemptible |

  GPU request names: `nvidia_h100_80gb_hbm3`, `nvidia_h200`.
- **Beta:** a GB200 NVL72 system (288 B200 GPUs) with ARM CPUs, so it needs
  ARM containers run through Pyxis and Enroot.

Reported by a research agent; not re-checked:

- **Allocation:**
  - About 20k service units (SU) per project per year. One Alpha GPU-hour
    costs 1 SU; one Beta GPU-hour costs 2 SU.
  - Each campus's technical contact handles accounts through ColdFront.
- **Storage:**
  - Home: 100GB.
  - Project: on VAST, billed per TB-month.
  - Scratch: 5TB on Lustre, purged after 30 days without access.
- **Containers:** Apptainer on Alpha.
- **SSH tunnels to compute nodes:** documented for Jupyter.

Not documented anywhere; ask help@empire-ai.org:

- vLLM.
- Whether compute nodes can reach the internet to download Hugging Face
  weights.
- A shared cache of model weights.
- Whether long-running services (an OpenAI-compatible server) are allowed.

**Estimated GPUs needed** (our estimates):

| Workload | GPUs |
|---|---|
| Serve 235B, BF16 (~470GB weights) | 1 H200 node, 8-way tensor parallelism. 8× H100 fits but leaves little room for the KV cache. |
| Serve 235B, FP8 (~240GB) | 4× H200 |
| Serve 32B (~66GB) | 1× H200 or 2× H100 |
| LoRA SFT, 32B | 1 node of 8 GPUs (FSDP/DeepSpeed), 48h `standard` job |
| LoRA SFT, 8B | 1 GPU |
| Online RL, 8B | 1 node (rollout and training GPUs split) |
| Online RL, 32B with LoRA | 8–16 GPUs |

One 8-GPU node for a week is about 1.3k SU. Long jobs on `burst` need
checkpointing, because they can be preempted.

## Code changes needed for a vLLM endpoint (not made yet)

From a read-only review on 2026-10-15. The pipeline uses Chat Completions
only, sends images as base64 PNG data URLs, and honors `OPENAI_API_BASE`
(`referential_task/ai_providers.py:225`). In priority order:

1. **The `developer` role.** It is used in `prompts/acl.py:229`,
   `prompts/cameron.py:129`, `ai_vs_ai.py:688/707/757` and `react.py:404`.
   The only remapping (`ai_providers.py:374`) is on the Gemini path. If vLLM
   or Qwen's chat template doesn't map `developer` to `system`, the
   instructions and the board-state note may be dropped without any error.
   Fix: merge the leading developer/system messages into one `system`
   message, and turn later ones into `user` messages, for non-OpenAI
   endpoints.
2. **No `max_tokens` on game calls, and `temperature=0` is sent.** With
   greedy decoding and open-ended arrays in the schema, a response can run
   until the timeout. Cap `max_tokens` (e.g. 1024) for local models.
3. **Failed parses aren't retried.** A parse failure or a 400 error returns
   `None` (`ai_providers.py:571-578`). The batch script stops the session
   after 5 turns without progress. Retry once, and log the server's error
   message.
4. **The RE extractor builds its own client** (`scripts/analyze_metrics.py:246`).
   The OpenAI SDK reads `OPENAI_BASE_URL`, so setting that variable sends
   extraction to vLLM. Set only `OPENAI_API_BASE`.
5. **One client per process**, so mixed GPT/Qwen pairs can't run yet.
6. **A "/" in the model name breaks session resume** in
   `run_replication.py:42`. Serve with `--served-model-name qwen3-vl-235b`.
7. **The second client** (`ai_vs_ai.py:208`, used only for end-of-session
   perceptions) has no timeout.

The JSON schemas look compatible with vLLM's structured-output backends:
they use `$defs`, `anyOf` with null, and `additionalProperties: false`, and
have no integer enums or numeric ranges. So positions 1–12 and candidates
1–18 are enforced only by the prompt, as on OpenAI.

**Context size:**

- Round 5, matcher: 5 images, about 4k image tokens. With ReAct (`--react`):
  9 images.
- Text history: 10–40k tokens, with no truncation.
- A 64k context is enough.

## 1-round smoke test checklist (once we have access)

1. On a `test` job, run
   `vllm serve Qwen/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --max-model-len 65536 --limit-mm-per-prompt '{"image":10}'`.
2. Set `unset OPENAI_BASE_URL; export OPENAI_API_BASE=http://<host>:8000/v1 OPENAI_TIMEOUT_SECONDS=600`.
3. Send a developer + user message to `/tokenize`, and check that the
   instructions appear in the rendered prompt.
4. Make one manual `beta.chat.completions.parse` call with
   `ACLMatcherResponse` and a 1404×1344 PNG, and check that `.parsed` is
   filled in.
5. Run
   `python scripts/run_batch_experiment.py --sessions 1 --model qwen3-vl-32b --prompt-strategy ACL_prompt --stop-after-round 1 --max-turns-per-round 40 --db /tmp/smoke.sqlite`,
   then again with `cameron-prompt`.
6. In the logs, check for:
   - no "OpenAI API error" or "Failed to parse";
   - no stall exit;
   - `usage` recorded;
   - non-null selections;
   - a round that ends with a score.
7. Repeat with `--stop-after-round 2` (feedback images) and with `--react`
   (9 images).
8. Confirm the run never called OpenAI.
9. Then move to the 235B model on an H200 node.

## Fine-tuning plan

### Turning human transcripts into training examples

1. **Turns.** Merge each speaker's consecutive messages into one turn, to
   match the harness's strict turn-taking.
2. **Placements.** For each matcher turn, decide which basket it placed:
   - Only 47% of human director messages start with a position number, and
     only 49 of 128 rounds name all 12 positions. So a regular expression
     isn't enough.
   - Instead, use the RE extractor's basket → phrase output to find the
     message that describes each position. The matcher turn after it places
     that position, using the basket the human ended up putting there
     (`selected_sequence`).
3. **Candidate numbers.** Map each basket to the harness's candidate number
   through its image path (`prompt_context.py:246`). The 18-basket pool is
   the same as in the human games.
4. **Known limits.** Moves can't be recovered, and these placement labels
   are approximate. Keep human errors in the data, since they are part of
   human behavior.
5. **Prompts.** Render each training prompt with the real harness: images,
   board-state note, and cross-round history. That way the model is trained
   on exactly what it will see at test time.

### Held-out data

All the human data uses basket set 5, so we can't hold out a basket set
without new human data. Options:

- Split by pair: e.g. 24 pairs for training, 8 for testing.
- Hold out a few baskets: drop every turn about them from training.
- Evaluate AI–AI play on basket sets 1–4 to measure generalization to new
  baskets. There is no human reference there.

The data is small: 32 pairs × 4 rounds, about 2.7k director messages. This is
the main limit of the SFT step and is worth discussing with Owen.

### LoRA

- Qwen3-VL-32B, rank 16–64 on the language model's attention and MLP layers.
  Freeze the vision encoder.
- Compute the loss only on the trained role's own turns.
- Do an 8B dry run first.

### Filtered self-play

- Serve the SFT model with vLLM and run `ai_vs_ai` at temperature
  ~0.7–1.0, with N sessions per grid.
- Score each session with the human-distance metrics above, using the
  held-out human half, plus accuracy.
- Keep the top-k sessions for another round of SFT (rejection sampling or
  expert iteration), or build best-vs-worst pairs for DPO. Repeat.
- `ai_reasoning_log` already records each selection's outcome
  (`selection_result`, `submit_result`).

### Online RL, only if needed

- Wrap the engine as an environment with reset, step and reward.
- Train with GRPO or PPO, with a KL penalty toward the SFT model.
- Rewards come at the end of a round, which makes credit assignment across a
  long round the hard part.

### Risks

- **Language drift.** Optimizing brevity or accuracy pushes the two copies
  of the model toward private shorthand. The human-distance reward and the
  KL penalty counter this.
- **Memorizing basket names.** All training data uses basket set 5.
- **Train/test mismatch.** We train on human–human data but evaluate
  AI–AI play.

## Open questions

- **Silent matcher turns.** Should a later harness version allow a matcher
  turn with no message? Humans can place baskets silently, so this fits the
  comparability standard in `harness_changes.md`. It would change the turn
  structure that turn-count metrics depend on.
- **Human-AI interface.** Does the human–AI interface from the paper still
  work with an open model behind it?
- **Generalizing to new baskets.** Would we need more human–human data on a
  second basket set to test this properly?
