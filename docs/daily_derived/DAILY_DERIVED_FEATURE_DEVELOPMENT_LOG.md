# Daily-Derived Feature Development Log

Related source of truth: `DAILY_DERIVED_FEATURE_DESIGN.md`  
Log created: 2026-08-18  
Current status: **STAGE 5 PARTIAL — TRAINING PAUSED AT 98/100 CYCLES — FORMAL STAGE 6 NOT STARTED**  
Status verified from live artifacts: 2026-08-21

## Working boundary

This log records only the independent Daily-Derived experiment. It does not duplicate the Raw Daily Baseline development history.

Before an approved implementation Step:

1. Read the frozen design and this log in full.
2. Inspect current code and git status.
3. Implement only the approved Step.
4. Use focused tests and stop at the stated boundary.

If implementation conflicts with the frozen contract, stop and request an explicit amendment. Do not change the design implicitly.

## Step summary

| Step | Scope | Status | Key result |
|---|---|---|---|
| 1 | Raw / Daily-Derived integration audit | Complete | One shared framework with feature-space-aware data, vocabulary, fingerprints, registries, checkpoints, and output isolation is feasible; no code changed. |
| 2A | Sixteen-feature data feasibility audit | Complete | Required fields exist; adjustment and units were verified; exact market-calendar lags and engineering shares PIT are feasible; no code changed. |
| 2B | Freeze Daily-Derived v1 Feature Contract | Complete | Sixteen ordered engineering-reconstruction formulas and data/PIT/validity/schema/isolation contracts frozen in the related design. |
| 3A | Pure numerical builder and synthetic tests | Complete | Aligned `(date, stock)` inputs now produce the frozen `(date, 16, stock)` float64 tensor; 12 focused and 4 adjacent tests passed. |
| 3B | Real input alignment and artifact builder | Complete | Raw axes/market/raw close and the existing PIT shares matrix feed a blockwise isolated float32 artifact with metadata, schema fingerprint, QA, and a manual Notebook entry; full real build not run. |
| 3C | Positive-price validity amendment | Complete | Every price dependency is finite and strictly positive; the schema fingerprint was updated. |
| 4B | Dual Feature Space vocabulary/model plumbing | Complete | Raw remains 6 leaves / 142 actions; Derived uses an isolated 16-leaf / 152-action registry and identity. |
| 4C | Runtime dual Feature Space plumbing | Complete | Sampler, trajectory, trainer, checkpoint and candidate paths propagate the selected registry; synthetic assembly passed. |
| 5 | Real Derived artifact | Complete | User-run artifact completed with shape `(4027, 16, 5424)`, float32 and frozen metadata/fingerprint. |
| 6 | Derived N=1/2 Exact-TB resources | Complete | Independent registry contains 1712 canonical candidates and audited N=1/2 exact mass/logZ. |
| 7 | Formal Derived Stage 5 | **Partial / paused** | Canonical run is at 98/100 cycles, 1470/1500 optimizer steps and 23520/24000 trajectories; `complete=false`. |
| 8 | Formal Derived Stage 6 | Not started | Compatibility and synthetic smoke exist, but there is no formal Derived Stage 6 Notebook, run or Factor Pool. |
| 9 | Strategy and OOS | Synthetic plumbing only | Synthetic matrix/strategy/Test-score assembly exists; no formal Derived Test labels or OOS evaluator were run. |

## Entry 001 — Step 1 integration audit

Date: 2026-08-18  
Status: Complete — read-only

Key conclusions:

- Raw Daily and Daily-Derived should be parallel feature spaces in one shared GFlowNet framework.
- Feature tensor/schema, leaf vocabulary, fingerprints, N=1/2 exhaustive registry, exact logZ, checkpoints, runs, and downstream artifacts require natural isolation.
- Grammar operator semantics, policy architecture, Conditional `N`, Hybrid Exact-TB/LPV, Reward, Stage 6, Strategy, and OOS framework remain shared.
- No production code, tests, configuration, Notebook, or artifact was changed.

## Entry 002 — Step 2A data feasibility audit

Date: 2026-08-18  
Status: Complete — read-only

Key conclusions:

- Adjusted OHLC, raw close, raw volume, raw actual amount, and historical `list_a_shares` are available in the current data sources.
- Numeric QA confirmed `volume` and `list_a_shares` are both in shares, while `amount` is actual RMB trading amount on the unadjusted price scale.
- Adjusted VWAP must remain `amount_raw * (close_adjusted / close_raw) / volume`.
- Exact global-calendar lag 1/5 behavior is feasible without suspension backfill.
- The existing `change_date <= trade_date` share alignment is usable as an engineering PIT standard but is not strict publication-time PIT.
- Simple volume/amount percentage changes have heavy finite right tails; the user explicitly froze simple changes for v1 rather than log-ratios or builder-level winsorization.
- No real data pipeline or training was launched.

## Entry 003 — Step 2B contract freeze

Date: 2026-08-18  
Status: Complete — documentation only

Work performed:

- Created `docs/daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md` as the frozen implementation source of truth.
- Frozen the exact 16-feature names, order, engineering-reconstruction formulas, inputs, units, adjustment, PIT/lag, warm-up, numerical validity, tensor/schema, mask, identity, and isolation rules.
- Recorded the experiment as a replacement of six Raw leaves by sixteen Derived leaves, not a combined leaf space.
- Added a short authority entry to `DEVELOPMENT_SPEC.md`; the detailed contract is not duplicated there.

Validation performed:

- Documentation-only static consistency checks for the ordered 16 names, formulas, field notation, constants, lag/warm-up groups, and source-of-truth links.
- Git diff and status review.
- No unit test suite, data processing, tensor generation, registry construction, or training was run.

Not implemented:

- Derived feature builder or artifact;
- feature-space-aware production data context;
- leaf/token/action-space changes;
- model, checkpoint, registry, exact-logZ, Stage 5, Stage 6, Strategy, or OOS integration.

Stop point: Step 2B documentation freeze complete. Await a separately approved implementation Step.

## Entry 004 — Step 3A pure numerical builder

Date: 2026-08-18  
Status: Complete — focused production implementation

Work performed:

- Added `factor_gfn/data/daily_derived.py` with the immutable 16-feature order and a keyword-only `build_daily_derived_features(...)` API.
- Accepted only eight already aligned `(date, stock)` inputs and calculated entirely in float64.
- Implemented exact row-position lag 1/5, feature-specific finite/denominator checks, OHLC geometry failure, CLV tolerance, daily turnover, and `1e8`-scaled daily illiquidity.
- Returned an in-memory `(date, 16, stock)` float64 array without reading or writing any artifact.
- Exported the narrow public API from `factor_gfn.data`.
- Added synthetic coverage for all formulas, order, dtype/shape, exact lags, NaN/Inf propagation, nonpositive denominators, K-line identity, invalid geometry, CLV boundaries/tolerance, turnover, illiquidity, causality, and input shape validation.

Validation performed:

```text
.\.venv\python.exe -m unittest tests.test_daily_derived -v
12 tests passed

.\.venv\python.exe -m unittest tests.test_masks -v
4 tests passed
```

Contract review:

- No formula, feature order, lag, mask, dtype, or PIT contract conflict was found.
- No Raw `FEATURE_COLUMNS`, preprocessing output, Grammar, Interpreter, GFlowNet, Exact-TB, Stage 5, Stage 6, Strategy, OOS, or real-data artifact was changed.

Stop point: pure numerical builder complete. Real Parquet/share alignment and formal Derived artifact construction remain a separately approved next Step.

## Entry 005 — Step 3B real alignment and artifact builder

Date: 2026-08-18  
Status: Complete — implementation and fixture validation; full real build not run

Work performed:

- Added `factor_gfn/data/daily_derived_artifact.py` for real Parquet alignment, five-row overlap date blocks, float32 artifact writing, minimal metadata, schema fingerprint, and QA.
- Reused the existing read-only `data/processed/barra/list_a_shares.npy`. Its producer uses the shared Raw date/stock axes and `trade_date >= change_date` ASOF semantics; preflight validates its shape, floating dtype, Barra metadata shape, missing values, and absence of nonpositive finite shares.
- Preserved Raw `amount` and `volume`, and reconstructed adjusted VWAP only as `amount_raw * (close_adjusted / close_raw) / volume_raw` with the existing Raw VWAP tolerance/range convention.
- Bound metadata to the shared Raw date/stock axis file hashes and the frozen schema fingerprint; recorded the shared Raw universe mask as read-only.
- Added `notebooks/prepare_daily_derived_data.ipynb` as a thin manual-only preflight/build/QA entry.
- Formal output is isolated under `data/processed/daily_derived_v1/`; existing formal output is never overwritten, and metadata becomes formal only after the tensor is complete and validated.

Validation performed:

```text
.\.venv\python.exe -m unittest \
    tests.test_daily_derived_artifact \
    tests.test_daily_derived \
    tests.test_masks -v
24 tests passed
```

Focused fixture coverage includes shuffled-axis alignment, reused shares boundaries/missing semantics, adjusted VWAP, raw amount preservation, Step 3A integration, block-boundary equivalence, float32 roundtrip, schema fingerprint sensitivity/stability, no-overwrite behavior, and atomic failure cleanup.

Read-only real preflight:

```text
date_count=4027
stock_count=5424
shares_shape=(4027, 5424)
shares_dtype=float32
shares_source_schema=factor_gfn.barra_five_style.v1
schema_fingerprint=0358f3c3008d491ae7b88276edae0c80883e1e7975973b822c4033cc5ffd2b61
```

A non-writing seven-date real-input smoke also completed through Parquet alignment, adjusted VWAP, reused shares, and the Step 3A builder with output shape `(7, 16, 5424)`, dtype `float64`, and 130964 finite values.

Contract review:

- No Step 2B or Step 3A contract conflict was found.
- No Raw artifact, Grammar, Interpreter, GFlowNet, Exact-TB, Stage 5, Stage 6, Strategy, or OOS component was changed.
- The full `4027 x 16 x 5424` real artifact was not built by Codex.

Stop point: implementation, fixture tests, and real read-only preflight complete. Await the user's manual Notebook build or a separately approved downstream integration Step.

## Entry 006 — Notebook import bootstrap fix

Date: 2026-08-18  
Status: Complete — lightweight Notebook patch

- Fixed the first code cell of `notebooks/prepare_daily_derived_data.ipynb` to locate the project root by walking upward from `Path.cwd()`, prepend it to `sys.path`, and only then import `factor_gfn`.
- This bootstrap is required for future project Notebooks so imports do not depend on Jupyter having been launched from the repository root.
- No production code, data artifact, or experiment contract changed.

## Entry 007 — Step 3C positive-price validity patch

Date: 2026-08-18  
Status: Complete — lightweight contract amendment and builder patch

- Required every adjusted price actually used by a feature (`open`, `high`, `low`, `close`, or `VWAP`) to be finite and strictly positive, including numerator prices.
- Required all O/H/L/C inputs of geometry features to be finite and strictly positive before the existing geometry checks.
- Kept all 16 formulas, order, lag/PIT, raw amount/volume/shares, CLV tolerance, and all downstream layers unchanged.
- Added focused tests for zero and negative price numerators and nonpositive geometry inputs.
- Updated the schema missing-value contract, changing the fingerprint from `0358f3c3008d491ae7b88276edae0c80883e1e7975973b822c4033cc5ffd2b61` to `a5973b625ab3fd187c8137f64bbdc2649104ea7107d09ef0f5adfc046db7486e`.

Validation:

```text
.\.venv\python.exe -m unittest tests.test_daily_derived tests.test_daily_derived_artifact -v
21 tests passed
```

The previously generated full artifact uses the old validity contract and must be preserved separately or replaced by a user-run rebuild before downstream integration. Codex did not run the rebuild.

## Entry 008 — Step 4B dual feature-space vocabulary and model plumbing

Date: 2026-08-18  
Status: Complete — stopped before sampler, trajectory, and trainer integration

- Added two immutable feature-space specifications and action registries: the historical six-leaf Raw Daily vocabulary and the frozen 16-leaf Daily-Derived v1 vocabulary.
- Kept every module-level grammar/token API as a Raw-compatible default. Raw still has 142 actions, leaf IDs `0..5`, action fingerprint `5689dbceb1bb42716773bcaf4cb5845041e578a3bb11fe67445ede6cde7938cc`, and unchanged structural hashes.
- Threaded the selected registry through expression/partial-AST nodes, `GrammarState`, exact-node reachability, `StateAdapter`, and `ForwardPolicyNetwork`; cross-registry composition and adapter/model batches now fail closed.
- Daily-Derived has 152 actions. Non-leaf operator semantics and ordering are shared with Raw, while leaf names, token IDs, action fingerprint, adapter dimensions, and model vocabulary-dependent dimensions are isolated.
- Preserved the historical Raw hybrid-config manifest and fingerprint byte-for-byte. Derived config identity explicitly includes the feature-space fingerprint and Derived action-space fingerprint, preventing Raw/Derived checkpoint compatibility.
- Did not modify sampler, trajectory, trainer, hybrid trainer, Reward mathematics, Exact-TB registries/logZ, Stage 5 execution, or Stage 6 selection mathematics.

Validation:

```text
.\.venv\python.exe -m unittest \
    tests.test_dual_feature_space \
    tests.test_grammar_tokens tests.test_expression tests.test_grammar_state \
    tests.test_grammar_integration tests.test_exact_node_grammar \
    tests.test_gfn_model tests.test_interpreter \
    tests.test_hybrid_variance_config tests.test_no_anchor_config \
    tests.test_search_space_config tests.test_gfn_real_data \
    tests.test_gfn_real_reward tests.test_backtest_stage6_evaluation -v
150 tests passed

git diff --check
passed (line-ending conversion warnings only)
```

Stop point: vocabulary, Grammar, StateAdapter, Model, and necessary config compatibility are complete. Runtime registry propagation through sampler/trajectory/trainer and construction entry points remains Step 4C.

## Entry 009 — Step 4C runtime dual Feature Space plumbing

Date: 2026-08-18  
Status: Complete — synthetic assembly only; stopped before Derived Exact-TB and formal training

- Propagated the adapter-owned `ActionRegistry` through sampler state creation, `DAGAction`, grammar diagnostics, trajectory validation, and replay.
- Made `HybridVarianceTrainer` assemble its adapter/model from `config.action_registry`; ordinary `GFNTrainer` remains explicitly Raw by default.
- Added early failure for mismatched Hybrid config and real-data expression Feature Space.
- Preserved the historical Raw state-hash payload exactly. Frozen Raw source/close state hashes remain `c0200f02f624b04c75e1159189aea602937199a1e1dda8b022b6080993d56323` and `a401a62c74f686979055b6ea9908b0b6f8db59e9003581a77b1a6fb618051eb8`.
- Added the action-space fingerprint to non-Raw state hashes only.
- Kept the Raw candidate artifact schema and fields unchanged; only Derived artifacts/records receive explicit feature-space and action-space vocabulary identity.
- Made Derived Exact-TB registry reuse and exact-mass registration fail closed until independent Derived N=1/2 resources exist.
- Replaced recursive `asdict()` of exhaustive count results with an explicit, byte-equivalent manifest so registry-bound Expressions are not deep-copied; enumeration and Exact-TB semantics were unchanged.

Validation:

```text
.\.venv\python.exe -m unittest \
    tests.test_dual_feature_space tests.test_gfn_policy_sampler \
    tests.test_gfn_trajectory tests.test_gfn_trainer \
    tests.test_hybrid_single_condition_batch \
    tests.test_hybrid_variance_trainer \
    tests.test_hybrid_checkpoint_runner \
    tests.test_train_candidate_artifact tests.test_exhaustive_pool -v
71 tests passed

git diff --check
passed (line-ending conversion warnings only)
```

The Derived smoke sampled and replayed a legal trajectory containing token ID 142, assembled a Derived Hybrid trainer, and collected a synthetic fixed-N=3 batch while leaving `optimizer_step == 0`. No real Reward run, optimizer update, exhaustive evaluation, exact-logZ construction, checkpoint, or formal Stage 5 Notebook was executed.

Stop point: runtime sampler/trajectory/trainer assembly is Feature-Space aware. Derived N=1/2 exhaustive resources and the separate formal Derived Stage 5 entry remain later approved steps.

## Entry 010 — Real Daily-Derived v1 artifact completion

Date: 2026-08-18  
Status: Complete — user-run real artifact; read-only metadata verification

The user completed the manual build through `notebooks/prepare_daily_derived_data.ipynb`. The authoritative local outputs are:

```text
data/processed/daily_derived_v1/data_tensor.npy
data/processed/daily_derived_v1/metadata.json
```

Verified artifact identity:

```text
status             = completed
shape              = (4027, 16, 5424)
dtype              = float32
tensor bytes       = 1,397,916,800
builder schema fp  = a5973b625ab3fd187c8137f64bbdc2649104ea7107d09ef0f5adfc046db7486e
date axis SHA-256  = 4bd5a3ab20c6c37777b41a9457220b8a550e15626f4a5e66edda1210da71c8a1
stock axis SHA-256 = 122da177725a5b41ae2ca33acbcb60360755848a1d8538b2619280c143ee1565
```

This completion supersedes the earlier Entry 005 statement that the full build had not been run. The historical statement remains in place to preserve the development sequence. The real artifact is local and Git-ignored; it must not be overwritten or automatically rebuilt.

## Entry 011 — Derived N=1/2 Exact-TB resources

Date: 2026-08-18  
Status: Complete — user-authorized real exhaustive build

Added the manual entry:

```text
notebooks/build_daily_derived_v1_exact_tb_n1_n2.ipynb
```

The independent output is:

```text
runs/daily_derived_v1/exact_tb_n1_n2/exhaustive_registry.sqlite3
```

Read-only verification recorded:

```text
schema=factor_gfn.exhaustive_registry.v2
feature_space_id=daily_derived_v1
action_count=152
candidate_count=1712

N=1: canonical=16, positive=16, logZ=-0.6403115056361736
N=2: canonical=1696, positive=1692, zero=4, logZ=3.8378534846711996
```

The build was an explicitly authorized over-budget exhaustive operation. It does not authorize another exhaustive run. Raw and Derived registries, Reward values and exact logZ remain isolated and must never be cross-loaded.

## Entry 012 — Formal Daily-Derived Stage 5 entry and partial run

Date: 2026-08-18 to 2026-08-21  
Status: Partial — training paused; not complete

Added the separate formal entry:

```text
notebooks/run_stage5_daily_derived_v1_hybrid_variance_real_5_15.ipynb
```

It reuses the Raw Baseline training mathematics while explicitly binding the Daily-Derived tensor, 152-action vocabulary, independent N=1/2 registry and isolated run root. Frozen controls remain `N=1..15`, Exact-TB/LPV routing, `K=16`, `max_depth=5`, `max_nodes=15`, 100 cycles, `lr=1e-4`, grad clip `5` and seed `42`.

Canonical run:

```text
runs/daily_derived_v1/stage5_hybrid_variance_real_5_15/
  derived_hybrid_5_15_k16_seed42_20260818T154007Z/
```

Live artifact verification after the user stopped training on 2026-08-21:

```json
{
  "complete": false,
  "global_optimizer_step": 1470,
  "total_trajectories_seen": 23520,
  "pending_assignment": {
    "condition_N": 15,
    "condition_position_in_cycle": 0,
    "cycle_index": 98
  }
}
```

The completion target remains 1500 optimizer steps and 24000 trajectories. This is not a completed Stage 5 result and does not authorize Stage 6 or a Raw-versus-Derived performance conclusion.

Publication note: the current Notebook is a local resume snapshot with saved outputs and enabled manual gates. Its clean-output/default-disabled tests are expected to fail until a separately approved publication-hygiene step resets the Notebook. No Notebook was changed during the documentation sync.

## Entry 013 — Downstream compatibility and synthetic plumbing

Date: 2026-08-19  
Status: Engineering smoke complete — formal Derived Stage 6/OOS not started

Stage 6 source import, expression compatibility and evaluation context were made vocabulary-aware so Derived prefixes reconstruct with the Derived registry and evaluate against the Derived expression tensor while labels, universe, dates, stocks, industry and Barra remain bound to the Raw market context.

Synthetic downstream coverage assembled Derived Train/Validation matrices, Equal Weight, Fixed ICIR and Static LightGBM strategies, a synthetic Test matrix and finite synthetic Test scores. It did not read formal Test labels and did not call the formal OOS evaluator.

Still absent:

- a formal Derived Stage 6 Notebook or full run;
- a frozen Derived Factor Pool and ordering;
- a formal Derived Top100 StrategyInput or Strategy Bundle;
- formal Derived Test scores or OOS evaluation.

The existing Raw Stage 6 and OOS Notebooks are bound to Raw Baseline artifacts and must not be reused as if they were Derived entry points.

## Entry 014 — Public README and reproducibility documentation

Date: 2026-08-21  
Status: Complete — documentation only

Work performed:

- rewrote the root `README.md` to explain the research question, Raw/Derived separation, current stage matrix, result boundaries, repository structure and Notebook entry order;
- added `docs/REPRODUCIBILITY.md` with environment, data schema, external PIT industry requirements, new/resume controls, artifact completion checks and known reproducibility gaps;
- explicitly recorded that the repository has no license and grants no third-party reuse permission;
- documented that the Shenwan PIT daily files are user-provided external data with no stable public source;
- documented the absence of a standalone formal Barra-build Notebook instead of claiming one-command reproduction;
- updated this log from the stale Step 3B header to the verified partial Stage 5 state.

Validation boundary:

- read-only live `runner_state.json` and artifact paths were rechecked before writing;
- only Markdown files were changed;
- no Notebook, production code, test, configuration, data, run, checkpoint, registry or output was modified;
- no download, data build, exhaustive evaluation, training, Stage 6 or OOS job was started.

## Entry 015 — Public Notebook hygiene and collaboration boundary

Date: 2026-08-21  
Status: Complete — publication-only patch

Work performed:

- cleared execution counts and saved outputs from `prepare_daily_derived_data.ipynb` and the formal Derived Stage 5 Notebook;
- restored the formal Derived training Notebook to `RUN_REAL_ONE_CYCLE=False`, `MODE='new'`, `RESUME_RUN_DIR=None` and `RUN_TO_TARGET_CYCLE=False`;
- added a default-disabled `RUN_FULL_BUILD` gate to the Derived data artifact Notebook;
- added `CONTRIBUTING.md` and `docs/README.md` without adding a software license;
- kept the local presentation outside Git pending a separate public-content review.

This patch changed only Notebook publication state and public documentation. It did not modify production code, experiment configuration, run state, checkpoint, registry, data or outputs, and did not execute any data build or training operation.
