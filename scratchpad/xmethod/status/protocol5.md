READY-TO-MERGE
# protocol5 status (2026-10-03): R-41a separate GPU shard + equal dataset hash across all arms of a dataset

Branch xm/protocol5 from feat/v2 7435420. Local commit, not pushed. No EVAL seed claimed or generated;
SEED_REGISTRY untouched; no dependency added; spec unchanged (sha unchanged).

## HAND-BACK
- Record field (checked on xm/dev-runs): `dataset_sha256` = `generate.dataset_hash(ds)` (sha256 over every array
  and name of the dataset, designs included). It is stamped by `runner.run_one`, so every record of a unit that ran
  carries it, including in the campaign's isolated child. Campaign's not-ok records do NOT carry it: see "For
  dev-runs".
- `scratchpad/xmethod/eval_analysis.py` V11 integrity:
  - Hashes are grouped per dataset (world, regime, lam, n, kappa, seed) over ALL records of all arms and shards. A
    not-ok record is now compared too, when it carries a hash. Any difference fails `dataset_hash`.
  - New failing check `dataset_hash_stamped`: an ok record without `dataset_sha256` makes the run PROVISIONAL.
    Before this change such records were skipped silently.
  - Not-ok records without a hash (a unit not run has no dataset) are only counted
    (`n_dataset_hash_unstamped_not_ok`).
  - New output keys: `n_dataset_hash_mismatch`, `dataset_hash_unstamped` (first 20), `n_dataset_hash_unstamped`.
    The markdown V11 line prints these counts.
  - `one_platform_per_dataset` is unchanged: R-41a requires the same platform; a GPU shard on another node passes.
- `docs/xmethod/PROTOCOL_A.md` (DRAFT v3, R-1..R-41a; 273 lines):
  - Header: xm/protocol5, feat/v2 7435420.
  - s.7 Platforms: a dataset's arms are on one platform and in one shard, except that the cmi_knn GPU arms may run in
    a separate Kaggle GPU shard. Every record stamps `dataset_sha256`, and all arms of a dataset must carry the same
    hash (the V11 rule as above).
  - s.11 FINAL list: the dataset sha must be stamped on every ok record and equal across all arms and shards (R-41a).
- Tests `tests/test_xmethod_eval_analysis.py`: 27 pass + 1 skip (the campaign.expand test needs xm/dev-runs). Only
  this file was run (memory note). ruff clean. New test `test_dataset_hash_equal_across_arms_and_shards_r41a` checks:
  - a separate GPU shard with equal hashes gives FINAL;
  - a GPU-shard record with another hash gives a mismatch;
  - a stamped infeasible record with another hash gives a mismatch;
  - an ok record without a hash fails `dataset_hash_stamped`;
  - an unstamped T3 record is only counted;
  - the markdown line.

## For dev-runs (R-41a: "every record stamps the dataset hash")
- `campaign._infeasible` / `_failed` / `_over` (xm/dev-runs) build not-ok records without `dataset_sha256`.
- Where the dataset was generated, these records could stamp `dataset_hash(ds)`:
  - over-budget (CPU or wall);
  - a child killed (OOM) or exiting without a record;
  - a method-load failure.
  This would let a GPU-shard failure be checked against the CPU shard's dataset too.
- T3 / registry skips never generate the dataset, so they have no hash. eval_analysis counts them and does not fail.
- The ok path is already fine (runner stamps; the isolated child's record keeps it).

## QUESTIONS
- Q12 ANSWERED (R-41a): a separate Kaggle GPU shard is allowed; equal dataset hashes are required (implemented).
- Q13 (non-blocking) Is the eval_analysis reading right?
  - Proposed: a hash is required on ok records only, and a not-run unit (T3 / registry skip) stays unstamped and is
    counted.
  - The alternative: campaign generates the dataset for skipped units just to stamp it, and eval_analysis requires
    a hash on every record.
