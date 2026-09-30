# Qwen3 exact-source extraction experiment

## Status

The frozen two-fold addition pilot completed on 2026-09-29 at approximately 04:08 Asia/Shanghai and **failed promotion**. The best configuration scored 0.6747896 versus the same-fold baseline 0.6749592, with mean fold delta -0.0001308. No remaining-fold expansion or B prediction was launched for this policy.

| Policy | Fold 0 delta | Fold 1 delta | Pooled two-fold score |
| --- | ---: | ---: | ---: |
| Strict mapping | -0.0001022 | -0.0010981 | 0.6744104 |
| Established mapping | -0.0004392 | +0.0001776 | 0.6747896 |
| Recall mapping | -0.0070011 | -0.0066295 | 0.6680329 |

The best policy added 20 entity records, one marked NO. Of the 19 new scored units, 6 were correct and 13 false positives (31.58% precision); patient associations gained 2 true positives and 4 false positives. Errors included ordinary anatomy, disease names mapped to a nearby HPO concept, annotation mismatches and phrase boundaries. All new mappings were semantic lookups. Simply loosening similarity thresholds worsened both folds, so this route was not expanded. Detailed provenance and diagnosis are in `pilot/summary.json` and `pilot/error_audit.json`.

The next separately declared hypothesis uses these same completed LLM spans to filter existing mapped mentions, rather than add candidates: `experiments/llm_corroboration/plan.json`. It compares three scopes with fixed overlap rules and preserves negated and unmapped entities. This is a new development experiment; it does not change the rejected addition result.

The established candidate remains `submissions/patientphex_b_context.jsonl`: repeated fivefold development score **0.6819157**, with no official B feedback. The only confirmed official B score is **0.6152** for `submissions/patientphex_b_cpu.jsonl`; see `reports/official_feedback/b_cpu.json`. Neither development score nor runtime acceptance establishes an official score near 0.70.

## Motivation and fixed experiment

The current residual audit found 941 gold mentions with no overlapping prediction. The next hypothesis is that a stronger language model can recover missing descriptions while the frozen HPO mapping and patient-association model preserve the existing predictions.

- Official model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218), Apache-2.0, 8,190,735,360 parameters, pinned SHA-256 for every file.
- Runtime: Python 3.12.13, torch 2.6.0+cu124, transformers 4.51.3, peft 0.15.2, accelerate 1.6.0; separate project environment `experiments/llm_runtime/.venv`.
- Fixed pilot: outer folds 0 and 1, 32 validation documents and 1,164 original-text chunks. Each fold retrieves at most two demonstrations only from its 64 training documents.
- Inference: FP16 on authorized V100 GPUs 0/2, eager attention, thinking disabled, greedy generation, context limit 4,096 tokens, output limit 512 tokens.
- Output: JSON arrays of literal original-source phrases. Invented, normalized or internally matched word fragments are rejected. Original global offsets are reconstructed from exact occurrences.
- HPO: the existing fold-trained span linker and fixed SapBERT/HPO/UMLS index. Only non-overlapping additions are allowed; patient predictions use the exact frozen context model, with baseline replay checked first.
- Pilot: three predeclared mapping thresholds. Promotion requires mean fold gain at least 0.008, worst-fold loss at most 0.002, better mention F1, association F1 loss at most 0.002, and format success at least 98%.
- Confirmation: a selected fixed policy must also pass folds 2–4, fivefold component guards and a document bootstrap before a separate full-fit/B-inference plan can be issued. All folds remain repeatedly used development data.

The authoritative source/input hashes, thresholds and confirmation requirements are in `experiments/llm_extraction/plan.json`. Raw prompt/response caches and weights live in ignored `work/` and `models/` paths on the server.

## Verification and transport

All **125 regression tests passed** in the existing server runtime, including five new tests for exact text/offsets, held-out demonstration exclusion and prompt-budget behavior. A full retrieval preflight confirmed 0 held-out examples across the 1,164 pilot chunks. This is a code/provenance result, not a model-accuracy result.

Direct server downloads suffered timeouts and approximately 2 MB/s weight throughput despite higher concurrency. The first model shard was preserved and SHA-256 verified. The remaining identical official shards were transferred through the workstation and SSH, then verified again before publication. All 13 files, 16,397,443,036 bytes, matched the pinned model manifest. The four unavailable dependency wheels were likewise matched to the existing lock; offline `uv sync --locked` installed the isolated environment without changing the original root environment. The four temporary workstation weight copies were deleted only after confirming the remote and local hashes; actual model files remain under `/home/dcf/chip2026/models/qwen3-8b`.

Real model loading/generation, a seeded finite LoRA optimizer step and an independent agent following `remote_runbook.md` all passed. The full model produced exact-source JSON with a peak allocation of 16.433453 GiB; the small Qwen3 LoRA witness had loss 4.19812155, gradient norm 0.75897253 and a confirmed parameter update. This proves runtime behavior, not full 8B fine-tuning or PatientPheX accuracy. The pilot runner rejects missing or mismatched acceptance records. Evidence is stored in `environment_smoke.json`, `environment_acceptance.json` and `agent_acceptance.json`.

## Reproduction

After runtime acceptance, use `bash scripts/run_llm_extraction.sh --fold 0` on GPU 2 and `LLM_GPU_INDEX=0 bash scripts/run_llm_extraction.sh --fold 1` on GPU 0. Both wrappers use the existing per-GPU lock and refuse an occupied device. Each prompt has a content identity and atomic response cache for exact resumability.

After both folds complete, `bash scripts/run_llm_extraction.sh evaluate` runs the frozen mapping/association evaluation in the original project runtime. It refuses to overwrite an earlier evaluation report. These commands generate development artifacts only.

The completed tmux session `chip-qwen3-pilot` executed these steps automatically after verifying independent acceptance. `pilot_queue.json` stores process IDs and terminal status; `scripts/check_llm_progress.py` checks live processes, cache progress and GPU use without mutating the experiment. Never restart a job solely because a status observation timed out.
