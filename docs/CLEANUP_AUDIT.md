# Codebase Cleanup Audit

This project is now centered on AI-vs-AI referential-game experiments. The active runtime path is:

- `main.py` for the FastAPI dashboard and one-at-a-time local simulation.
- `scripts/run_batch_experiment.py` for headless experiment batches.
- `referential_task/ai_vs_ai.py` for AI-vs-AI turn orchestration and status.
- `referential_task/visual_context.py` for image loading and visual composites.
- `referential_task/prompt_context.py` for shared task background, prompt history, and prompt strategy config.
- `referential_task/sequence.py` for matcher sequence updates.
- `referential_task/ai_providers.py` for OpenAI/Gemini provider calls.
- `referential_task/prompts/acl.py` and `referential_task/prompts/cameron.py` for currently selectable prompt strategies.
- `referential_task/ai_perceptions.py` for final mutual AI evaluations.
- `referential_task/common_ground_agent.py` only when `enable_conceptual_pacts` is enabled.

## Removed In This Cleanup

- `referential_task/ai_reply.py`, `referential_task/ai_sequence.py`, and `referential_task/ai_context.py`, which were partial extractions from the old human-AI/oTree architecture and were not imported by the active app or batch runner.
- `_static/js/draggable-grid.js`, `_static/js/basket-selection.js`, `_static/css/basket-selection.css`, `_static/docs/consent.pdf`, and `_static/js/typing-indicator-test.html`, which were human-facing/oTree UI remnants not referenced by `templates/AIvsAIObservation.html`.
- `_static/ai_debug/`, `_static/global/empty.css`, and broken legacy symlinks under `_static/css/` and `_static/images/`.
- `data/old/`, `data/archive/`, `data/exported_sessions/`, `data/experiments/`, `prompt_exports/`, `data.sqlite`, `.DS_Store`, and `__pycache__/`, which are generated or historical artifacts rather than source code.

## Remaining Cleanup Targets

- Continue moving any new runtime helpers into the AI-vs-AI-only modules below instead of adding more behavior to `referential_task/ai_utils.py`.
- `referential_task/ai_utils.py` is now a compatibility facade for older imports. It can be deleted once downstream scripts and notebooks no longer import it.

## Suggested Refactor Shape

Use an AI-vs-AI-only module layout and delete or archive the human-AI layer:

- `referential_task/ai_vs_ai.py`: `generate_ai_vs_ai_reply`, `run_ai_vs_ai_turn`, `get_ai_vs_ai_status`.
- `referential_task/visual_context.py`: image loading and director/matcher composite builders.
- `referential_task/sequence.py`: matcher partial-sequence updates and submit validation.
- `referential_task/prompts/`: prompt strategies (`acl.py`, `cameron.py`).
- `referential_task/exporting.py`: shared export helpers used by both `main.py` and `scripts/run_batch_experiment.py`.

The immediate low-risk cleanup was completed. The follow-up split moved runtime behavior out of `ai_utils.py`, added explicit `ai_role` prompt construction, and centralized session export helpers.
