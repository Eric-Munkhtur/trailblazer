# HINT Baseline — Full Runtime, GPU & Data Accounting
(All times UTC, 2026-10-11; local Mac time = UTC − 4h. Compiled from Colab CLI history logs + VM logs.)

## 1. Wall clock
- Start (first session created): 02:23 UTC (10:23 PM Oct 10 local)
- Finish (last artifact downloaded, sessions stopped): ~07:15 UTC (3:15 AM Oct 11 local)
- **Total elapsed: ~4 h 52 m.** Science completed in first ~35 min; the rest was compute logistics (see §4).

## 2. Session-by-session ledger

| # | Session | Hardware | Life (UTC) | Duration | Work performed | How it ended | Est. units |
|---|---|---|---|---|---|---|---|
| 1 | hint-base | CPU 2-core 12.7GB | 02:23→~06:2x | ~4 h | env inspect; repo clone @8dc0497; full code audit; sentence census (513,863); data validation; authors'-predictions scoring; BioBERT equivalence check v1 (caught token_type_ids bug) + v2 (EQUIVALENT) | pruned by Google (keep-alive broken account-wide) | 0 (CPU free) |
| 2 | hint-gpu | T4 15.6GB | 02:25→~03:28 | ~63 m | deps; embedding gen v1 (wrong token_type_ids, discarded); gen v2 FIXED: 513,863 sentences, 9.2 min GPU, 1.76 GB pkl (sha fca6a55a…); ckpt smoke load; authors'-ckpt eval (2.8 min); train s0 launched 02:58 + s1 03:20 (parallel) | pruned ~60 min mark (s0 at epoch 2/5, s1 at epoch 0 — lost; metrics for eval preserved in logs/downloads) | ~2.1 |
| 3 | hint-gpu2 | T4 15.6GB | 03:44:49→04:49:57 | 65 m | clone+deps; embedding regen (8.6 min, pkl sha e351d634… — byte-different, VALUES identical: ckpt-eval reproduced to 1e-9); authors'-ckpt eval (provenance r=0.742); equivalence check (EQUIVALENT, <7e-7); train s0 launched (duplicate of a concurrent agent's identical run — mine killed at 6 min; theirs ran to epoch 1) | pruned at 60 min (reason logged: "pruned") | ~2.2 |
| 4 | hint-gpu3 | T4 15.6GB | ~04:54→05:14 | ~20 m | Phase II-subset embedding gen attempt 1 (my dropped-pickle bug → moved stub instead; caught by hash check); attempt 2 OK: 3.2 min GPU, 688 MB pkl (sha 61e5933c…); downloaded to Mac (3 min) | stopped deliberately ("Session terminated") | ~0.7 |
| 5 | hint-cpu1 | CPU | 05:21→06:22:49 | 62 m | 11×64 MB chunked pkl upload; **train s0 COMPLETE** 05:22:51→05:50:24 (1592 s; artifacts downloaded 05:55); train s2 started 05:50:24, killed by prune mid-epoch-4 | pruned at ~60 min | 0 |
| 6 | hint-cpu2 | CPU | 05:22→~06:25 | ~63 m | chunked pkl upload; **train s1 COMPLETE** 05:22:53→~05:56 (5 epochs ≈380 s each + eval; artifacts downloaded) | pruned at ~60 min | 0 |
| 7 | hint-cpu3 | CPU | ~06:14→07:12 | ~58 m | chunked pkl upload + reassemble + hash verify; **train s2 COMPLETE** 06:25:11→06:54:09 (1677 s; bundle downloaded 06:56); finisher: env freeze + stub-embedding ckpt eval + evidence tar (downloaded) | stopped deliberately | 0 |

**GPU time attached: ≈ 2 h 28 m → ≈ 5 units at T4 ≈ 2 units/h.**
Plus: one orphaned T4 (session #2's VM) billed idle until Google reclaimed it (unobservable from here; ≤ ~1–1.5 h ≈ ≤ 3 units). Two orphaned CPU VMs cost nothing.
**Total compute-unit spend: ≈ 5–8 of 447 (≈1–2%).** Verify in Colab → Manage compute units.

## 3. Training runs (final, all artifacts downloaded)

| run | machine | wall | epochs (valid loss) | best | test ROC / AP / F1 / bin-PR-AUC |
|---|---|---|---|---|---|
| s0 | cpu1 | 1592 s (26.5 m) | init 9.7028 → 9.5997, 9.0211, **8.8914**, 9.2580, (ep4 ~9.0) | ep2 | 0.6303 / 0.6703 / 0.6489 / 0.6152 |
| s1 | cpu2 | ~31 m | init 10.8886 → 9.5680, 9.0185, **8.8551**, 9.2556, 8.8843 | ep2 | 0.6390 / 0.6759 / 0.6496 / 0.6155 |
| s2 | cpu3 | 1677 s (28 m) | init 9.7224 → 9.6260, 9.2872, 8.8868, 8.7908, **best ep4** | ep4 | 0.6391 / 0.6823 / 0.5950 / 0.6199 |

- All: batch 32, Adam lr 3e-4, wd 0, BCEWithLogits, 5 epochs, best-valid-loss selection, PYTHONHASHSEED=0, device=cpu.
- s0 used the pre-fix seeding path (effective seed 0 via import chain); s1/s2 seeded after imports (fixed).
- Determinism evidence: the reaper-killed s2 attempt on cpu1 reproduced identical per-epoch losses (9.6260/9.2872/8.8868/8.7908) on cpu3.
- ~325–385 s/epoch on CPU VMs vs ~497 s on T4 (code is Python-loop-bound; GPU util 11%).

## 4. Failure/overhead ledger (why 5 hours)
1. GPU session 1 pruned at 60 min → lost s0(ep2)+s1(ep0); root cause: Colab keep-alive RPC 403 USER_PROJECT_DENIED (Google-side), daemon died 1 min after every session creation → all sessions (GPU and CPU) have ~60-min lives on this account.
2. GPU session 2 pruned at 60 min → lost s0(ep1). Also a concurrent agent on this Mac launched a duplicate s0 on the shared VM (6 min wasted; mine killed).
3. CPU session 1 pruned at 60 min → lost s2(ep4). CPU session 2 died similarly.
4. colab CLI upload breaks on ~700 MB files (BrokenPipe) → 688 MB pkl shipped as 11×64 MB chunks + reassembly + sha256 verify (done 3 times: cpu1, cpu2, cpu3; ~6–8 min each).
5. My bugs (each caught by a check, none affected final numbers): token_type_ids=0 vs 1 in embedding semantics (caught by equivalence check — this one WOULD have affected numbers); dropped pickle.dump in subset generator (caught by hash); duplicate process collision; empty-output execs on busy kernels (~5-min exec timeout; long jobs re-run detached).
Net: ~2 h of GPU/science + ~2.9 h of infra recovery/transfer.

## 5. Every artifact (Mac: "HackRU project/")

**hint_baseline/** — evidence package
- BASELINE_REPORT.md · RESULTS_TABLE.md · RUNTIME_ACCOUNTING.md (this file)
- final_predictions/{authors_shipped, ckpt_authors, ckpt_authors_stubemb, hint_p2_s0, hint_p2_s1, hint_p2_s2}/preds_test.csv + metrics.json (per-example: nctid, label, probability, class@0.5, run_id, seed) + regenerated_metrics.json
- artifacts/data_integrity/data_integrity_report.json — split counts, class balance, overlap=0, hashes
- artifacts/authors_shipped_predictions/{metrics.json, predictions.csv}
- artifacts/authors_checkpoint_eval/{metrics.json, preds_valid.csv, preds_test.csv} — ckpt sha da2c94fe…, provenance vs shipped (r=0.7419, mean|Δp|=0.0795)
- artifacts/hint_evidence/authors_checkpoint_eval_stub/… — stub-embedding supplementary (ΔROC −0.0002)
- artifacts/hint_evidence/environment/{environment.json, requirements_freeze.txt, git_status.txt, upstream_code_changes.patch (0 bytes = zero upstream code changes)}
- artifacts/training/hint_p2_s{0,1,2}/ — config.json, log.json (per-epoch losses/times), metrics.json, preds_valid/test.csv, best_model_state.pt + final_model_state.pt + checkpoint_resume.pt, gram_mapping.pt; bundles s{0,1,2}_bundle.tar.gz + evidence_final.tar.gz
- artifacts/embedding_generation/{equivalence_check.json, embedding_meta.json, progress.log}
- artifacts/pipeline_logs/ — validate/score/eval/equivalence/freeze/eval_stub/train_s{0,1,2}/runner logs
- scripts/ — 15 scripts (validate_data, score_authors_predictions, gen_embeddings, gen_subset_embeddings, equivalence_check, eval_checkpoint, train_hint, runner_cpu, freeze_env, rebuild_results_table, finish_cpu1, final_build, finalize_package + copies)

**hint_handoff/** — private MSI transfer bundle (do not publish)
- HANDOFF.md (10 items) · checksums.sha256 · example_inference.py · requirements_1050ti.txt
- phase2_sentence2embedding.pkl — 688,673,494 B, sha256 61e5933cb7ff91d59d4f29f6dce17a6bb818ebe4bb6968517f45b1e2fee3602b
- phase2_subset_provenance.json (produced by the concurrent agent)
- scripts/ (15) · evidence_summary/{BASELINE_REPORT.md, RESULTS_TABLE.md, regenerated_metrics.json}

## 6. Key SHA-256 index
| asset | sha256 |
|---|---|
| authors' phase_II.ckpt (immutable start) | da2c94feca23c751925faceab118895b8ccb68c65a454daad01e1977d7af8e63 |
| phase_II_train.csv (4,005 rows) | 23e767816483b53ffad04c1e570d7ad4192e09a314ac0dd52375443c2afe2e39 |
| phase_II_valid.csv (446) | c5905bfc5b646e30321bd8e671d7bd946b8fb1dfe5c30a74bb29fa40e2a711e7 |
| phase_II_test.csv (1,654) | 6e829c430ce1e397993dfc003f522f9e8089a6b13117800ff7943a252423ea91 |
| authors' shipped test predictions | a45453e312c38086adfe29f9e0fc5001770b179816a7d6d7a653b011eb18d95f |
| subset embedding cache (SHIPPED) | 61e5933cb7ff91d59d4f29f6dce17a6bb818ebe4bb6968517f45b1e2fee3602b |
| full 1.76 GB cache (session 3) | e351d63475ab021ee2e6c55e912fdf570b305e072d7891333913c05e65d22952 |
| full cache (session 2, first gen) | fca6a55ae33a64d1a01b535b63d24b1c331b5c471150a4aa6fd7d64662e7cf46 |
| upstream stub sentence2embedding.pkl | 8ec16a78d9c9d3646c559df239aea3f15333605ddc702eeacebf6dd3ebeff684 |
| upstream commit | 8dc0497f23fdb84e2905da7655924a91e6e79798 (2025-06-25) |

(Not hashed: admet_model.ckpt — loads + used as documented; hash it before MSI use if desired.)

## 7. Environments
- Training/eval VMs (CPU): Python 3.13.16, torch 2.11.0 (CPU), rdkit/icd10-cm/sklearn per artifacts/hint_evidence/environment/requirements_freeze.txt
- Embedding gen (T4): same Python, torch 2.11.0+cu130, transformers (dmis-lab/biobert-base-cased-v1.1)
- MSI target: requirements_1050ti.txt (torch 2.4.1+cu121 — Pascal support)
