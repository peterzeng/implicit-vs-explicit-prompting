# Data (placeholder)

The transcripts and per-round metrics for the experiments reported in the paper
will be released here.

**Planned contents**

- 40 complete AI--AI sessions: 2 prompt conditions (implicit, explicit) x
  2 model versions (GPT-5.2, GPT-5.5), 10 runs per cell, 5 rounds per session.
- Per-session JSON traces (chat messages, selected sequences, accuracy).
- Per-round metrics: total words, turns, referring-expression length, and
  lexical overlap with prior rounds.
- The plain-text transcripts used for the qualitative analysis in the paper.

**Open items before release** (see the group discussion, items B4 and C2)

- Confirm redistribution rights for the basket stimulus images in `_static/`.
- Confirm the data license. The intended license is CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/), which differs from the MIT
  license covering the code in this repository.
- Decide whether to ship the full run history or only the sessions reported in
  the paper.

Until then, the raw runs remain in the `emnlp` branch of
https://github.com/peterzeng/lvlms-referential-game
