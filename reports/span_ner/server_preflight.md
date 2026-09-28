# 177 server preflight and training preparation

Observed at 2026-09-28 21:28 Asia/Shanghai over authenticated SSH as `dcf`.
These readings are snapshots, not a resource reservation.

| Resource | Observation |
|---|---|
| Host | `10.253.27.177`, `dgxs-DGX-Station` |
| RAM | 251 GiB total, 15 GiB used, 44 GiB free, 191 GiB buffers/cache, 231 GiB available |
| Memory pressure | PSI `some` and `full` averages all zero; no swap configured |
| CPU | Xeon E5-2698 v4, 20 cores / 40 logical CPUs; existing jobs heavily use CPU |
| Root filesystem | 117 GiB available, 94% used |
| Other filesystems | `/disk2`: 161 GiB; `/disk3`: 115 GiB; `/disk1`: 21 GiB available |
| GPUs | 4 x Tesla V100-DGXS-32GB; GPU 2 had 14 MiB occupied by display processes and 0% utilization |
| Driver | 550.163.01; reported CUDA compatibility 12.4 |

Available RAM includes reclaimable cache. The low completely-free value does not
indicate pressure here. Disk headroom and concurrent jobs are the practical limits.
GPU and disk availability must be checked again immediately before launch.

## User-selected storage

All actual data is stored beneath `/home/dcf`, not on `/disk2` and not through a
symlink. The user selected this after the disk comparison.

- `/home/dcf/umls/`: the user uploaded a licensed 2026AA installed subset and indexes
  from another computer, totaling 36,031,260,551 bytes. All 2,985 manifest entries
  independently passed existence/size checks. No UMLS was downloaded by this run.
- `/home/dcf/chip2026/`: isolated training deployment, `.venv`, project-local uv,
  model weights, caches, logs and checkpoints.

For the current 2026AA release, the complete Metathesaurus subset is 5.4 GB
compressed and 38.0 GB extracted; the full UMLS release is 5.3 GB and 40.1 GB.
The uploaded installed subset and indexes differ from these full-release packages.
Source: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

Reserve approximately 60–80 GB for a complete UMLS archive, extraction and working
space; this is a planning allowance, not an official RAM requirement. Stream the
RRF tables and extract HPO-related terms; a complete in-memory database is not
required. Keep the competition's supplied HPO 2026-06-23 as the output authority.

## Training scope

Start with BiomedBERT span extraction on the existing document-level fold 0
(64 train / 16 validation). Preserve overlapping spans and exact original offsets.
Negation is a separate span label. HPO normalization and patient association are
subsequent modules; span-only diagnostics must not be reported as competition scores.
Do not rerun or overwrite the completed CPU/GPU/RAG/fusion experiments.

The model is pinned to Microsoft revision
`e1354b7a3a09615f6aba48dfad4b7a613eef7062`. Direct server connections to Hugging Face
were refused. The local computer resolved the official repository's CDN redirect;
the server then downloaded the pinned bytes directly from that official CDN and
verified the official Git/LFS hashes. No substitute model was used.

Dedicated SSH key authentication was configured and independently verified with
native OpenSSH. Credentials are not stored in this report or the project.

## After UMLS upload

The root filesystem had approximately 82 GiB free after upload and during training
environment installation. RAM remained healthy: 232 GiB available, despite only
8.7 GiB completely free and 226 GiB in reclaimable buffers/cache.

The uploaded UMLS term SQLite index is approximately 1.54 GB. The supplied HPO
SQLite is approximately 8.79 MB and its source OBO hash matches this project.
Use read-only, indexed SQL queries instead of loading all raw RRF data into RAM.
Use `/usr/bin/python3` for server-side SQLite reads, as explicitly requested by
the user. Model training uses the independent project Python 3.12 `.venv`.
The UMLS index covers an English ten-source subset. Its older HPO version covers
18,627 of the competition's 19,119 allowed IDs, so the current HPO must remain
the output authority. Detailed independent checks and their limits are recorded
in `umls_validation.json`; the full upload SHA256 check is the uploader's claim,
not a repeat hash scan performed by this task.

## During the completed initial training run

At 2026-09-28 22:40:24 Asia/Shanghai, RAM was 251 GiB total, 31 GiB used,
1.5 GiB completely free, 218 GiB buffers/cache and 217 GiB available. The
filesystem backing `/home/dcf` had approximately 72 GiB available (96% used).
GPU 2 used 3,758 MiB at that instant; the trainer's peak allocated memory was
3.05750 GiB. Low free RAM still reflected reclaimable cache, not a shortage.
These are host snapshots and include concurrent jobs belonging to other users.

The five-epoch training finished at 22:40:38. Keep raw UMLS tables on disk and use
the indexed SQLite resources. Disk headroom, especially when retaining optimizer
checkpoints for more folds, remains the main storage constraint. This initial
span detector did not read UMLS or train an HPO/patient-association model.
