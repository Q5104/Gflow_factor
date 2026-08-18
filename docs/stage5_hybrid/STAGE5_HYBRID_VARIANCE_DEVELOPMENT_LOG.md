# Stage 5 Hybrid Variance Development Log

Related source of truth: `STAGE5_HYBRID_VARIANCE_DESIGN.md`  
Log created: 2026-08-16  
Current status: **B1 FROZEN — B2/B3/B4 DEFERRED — 100-CYCLE CUMULATIVE-TARGET NOTEBOOK READY — REAL TRAINING MANUAL ONLY**

## Working protocol

This log is maintained throughout implementation.

Before every Step:

1. Read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
2. Read this log in full.
3. Inspect `git status --short`.
4. Inspect the exact current diff of files in the approved Step.
5. Confirm the Step scope against the frozen design.
6. Do not start any later Step.

After every Step:

1. Run the Step's focused tests and record the exact commands and results.
2. Record all files changed by the Step.
3. Record assumptions, deviations, failures, and unresolved issues.
4. Mark the Step complete only if its focused tests pass.
5. Stop and wait for the user's next instruction.

If a focused test fails or a frozen contract cannot be implemented as written:

- do not broaden the implementation silently;
- record the evidence here;
- leave the Step incomplete;
- stop and ask the user before changing the frozen design.

## Frozen implementation checklist

| Step | Scope | Status | Focused tests | Notes |
|---|---|---|---|---|
| 1 | Isolated 5/15 hybrid config; configurable K; formal K=16 | Complete | 22/22 passed | Completed 2026-08-16; legacy config regression included |
| 2 | Robust Scheduling direct LPV | Complete | 15/15 passed | Completed 2026-08-16; legacy TB regression included |
| 3 | Transactional condition scheduler | Complete | 12/12 passed | Completed 2026-08-16; legacy scheduler and Trainer regressions included |
| 4 | Configurable-K single-N batch collection | Complete | 16/16 passed | Completed 2026-08-16; scheduler and legacy conditioned collector regressions included |
| 5 | Existing N=1/2 exact-Z compatibility | Complete | 17/17 passed | Completed 2026-08-16; no production adapter required |
| 6 | Hybrid routing and policy-only optimizer | Complete | 69/69 passed | Completed 2026-08-16; legacy no-anchor path preserved |
| 7 | Objective diagnostics, counts, terminal diversity | Complete | 55/55 passed | Completed 2026-08-16; no complexity-boundary or fake TB diagnostics |
| 8 | Hybrid checkpoint, runner, mid-cycle resume | Complete | 45/45 passed | Completed 2026-08-16; deterministic mid-cycle resume and schema isolation verified |
| 9 | Focused regression and synthetic smoke | Complete | 104/104 passed | Completed 2026-08-16; one synthetic K=2 cycle completed and resumed |

## Initial repository state

Recorded on: 2026-08-16

```text
branch: main
HEAD: 3ba7753
```

The worktree was already dirty before implementation of this design. Existing modified and untracked files belong to the prior project state and must not be reset, deleted, or overwritten. Several future target files already contain changes, including:

```text
factor_gfn/gfn/__init__.py
factor_gfn/gfn/checkpoint.py
factor_gfn/gfn/config.py
factor_gfn/gfn/no_anchor_config.py
factor_gfn/gfn/no_anchor_search_runner.py
factor_gfn/gfn/search_runner.py
factor_gfn/gfn/trainer.py
multiple no-anchor tests and notebooks
```

Before editing any of these files, the responsible Step must inspect and preserve the existing diff.

## Entry 000 — Design freeze

Date: 2026-08-16  
Status: Complete

### Work performed

- Added the frozen hybrid-variance design document.
- Added this development log and Step gating protocol.
- Recorded the existing branch, HEAD, and dirty-worktree warning.
- Did not begin Step 1.

### Files added

```text
STAGE5_HYBRID_VARIANCE_DESIGN.md
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md
```

### Tests and execution

```text
Tests run: none
Static checks run: none
Real Reward: not run
Structural enumeration: not run
Training: not run
```

Reason: this entry only freezes documentation and establishes the implementation protocol.

### Next gate

Wait for an explicit user instruction to start Step 1. Before Step 1, re-read both documents in full and inspect the current worktree/diffs.

## Entry 001 — Step 1 hybrid configuration

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected the existing diffs for `config.py`, `no_anchor_config.py`, `__init__.py`, and `test_no_anchor_config.py`.
- Confirmed that `__init__.py`, `no_anchor_config.py`, and legacy config tests already contained unrelated A/B1 changes; preserved them.

### Work performed

- Added an isolated hybrid configuration module instead of adding logZ-inapplicable fields to the legacy `TrainingConfig`.
- Froze `max_depth=5`, `max_nodes=15`, and conditions `1..15`.
- Froze the objective partition to exact TB `(1, 2)` and LPV `(3..15)`.
- Added configurable `trajectories_per_batch` with first-version default `K=16` and validation `K>=2`.
- Added an explicit positive `max_cycles` budget and exact cycle/optimizer/trajectory unit conversions.
- Froze the policy-only Adam contract at learning rate `1e-4` and gradient clip `5` without learned-logZ training fields.
- Added a deterministic manifest and fingerprint for the hybrid config.
- Exported the new config API from `factor_gfn.gfn` without modifying the legacy no-anchor config implementation.
- Added focused config, boundary, fingerprint, and legacy-regression tests.

### Files changed

```text
factor_gfn/gfn/hybrid_config.py                 added
tests/test_hybrid_variance_config.py            added
factor_gfn/gfn/__init__.py                      hybrid exports appended
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md       updated
```

`factor_gfn/gfn/__init__.py` was already modified before Step 1. This Step preserved its existing A/B1 exports and added only the hybrid-config imports and `__all__` entries.

### Focused tests

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_variance_config tests.test_no_anchor_config -v
```

Result:

```text
Ran 22 tests in 1.247s
OK
```

Coverage included:

- frozen `5/15` search and condition contract;
- K=16 default and configurable K;
- rejection of K<2 and invalid cycle budgets;
- exact-TB/LPV partition;
- policy optimizer settings and absence of logZ fields;
- deterministic fingerprint behavior;
- exact-node trajectories respecting `depth<=5` and `nodes<=15`;
- unchanged legacy formal no-anchor fingerprint;
- existing no-anchor config and A/B1 regression tests.

### Static checks

Command:

```powershell
.\.venv\python.exe -m py_compile factor_gfn/gfn/hybrid_config.py tests/test_hybrid_variance_config.py
```

Result: passed with no output.

An additional read-only precheck confirmed that `SearchSpaceConfig(max_depth=5, max_nodes=15)` resolves feasible exact-node conditions to exactly `(1, ..., 15)`.

Public export smoke command:

```powershell
.\.venv\python.exe -c "from factor_gfn.gfn import build_stage5_hybrid_variance_5_15_config; c=build_stage5_hybrid_variance_5_15_config(max_cycles=2); print(c.search_space.max_depth, c.search_space.max_nodes, c.training.trajectories_per_batch, c.training.total_optimizer_steps, c.training.total_training_trajectories)"
```

Result:

```text
5 15 16 30 480
```

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Structural exhaustive enumeration: not run
Training: not run
Step 2 or later implementation: not started
```

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 1 issues: none
```

The first attempted combined patch failed its context check before applying any changes because `__init__.py` contained pre-existing edits. The implementation was then applied in smaller patches without overwriting those edits.

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 2. Before Step 2, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 2 file diffs.

## Entry 002 — Step 2 direct LPV objective

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected existing diffs for `loss.py`, `trajectory.py`, and `__init__.py`.
- Confirmed that `loss.py` and `trajectory.py` had no local diff and left both unchanged.
- Confirmed that `__init__.py` already contained prior A/B1 and Step 1 exports; preserved them.

### Work performed

- Added a standalone direct log-partition variance implementation.
- Implemented the frozen loss exactly as `centered.square().mean()`.
- Added a pure Tensor entry point for exact value and analytic-gradient verification.
- Explicitly detached Reward and fixed PB inputs while preserving the PF autograd graph.
- Kept `zeta_mean` in the normal autograd expression without explicit detach.
- Added a parameter-free and state-free `LogPartitionVarianceLoss` adapter for existing `Trajectory` objects.
- Enforced `K>=2`, one fixed condition per batch, and LPV conditions `N=3..15`.
- Exported the LPV API from `factor_gfn.gfn`.
- Did not add VarGrad normalization, a leave-one-out loop, logZ parameters, or persistent LPV state.

### Files changed

```text
factor_gfn/gfn/log_partition_variance.py         added
tests/test_log_partition_variance.py             added
factor_gfn/gfn/__init__.py                       LPV exports appended
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/__init__.py` was already modified before Step 2. This Step added only the LPV imports and `__all__` entries.

### Focused tests

Initial command:

```powershell
.\.venv\python.exe -m unittest tests.test_log_partition_variance tests.test_gfn_loss -v
```

Initial result:

```text
Ran 15 tests in 1.844s
FAILED (failures=1)
```

The failure was confined to a test fixture intended to check that condition `N=2` is rejected by LPV. The fixture had removed its condition fingerprint and therefore correctly failed earlier in the existing `Trajectory.validate()` contract. No production-code change was required. The fixture was corrected to construct a valid conditioned `N=2` trajectory.

Rerun command:

```powershell
.\.venv\python.exe -m unittest tests.test_log_partition_variance tests.test_gfn_loss -v
```

Final result:

```text
Ran 15 tests in 2.212s
OK
```

Coverage included:

- equal zeta values produce zero loss and zero gradient;
- unequal zeta values produce the direct population empirical variance;
- analytic gradient `2 * (zeta_i - zeta_mean) / K`;
- the corresponding negative sign through `zeta = constant - sum_log_pf`;
- Reward and PB have no gradient while PF retains gradient;
- `zeta_mean` remains in the autograd expression;
- K<2, invalid shapes, mixed conditions, exact-TB conditions, greedy trajectories, and missing Reward are rejected;
- the LPV module has no parameters and an empty state dict;
- the selected scale is not the VarGrad-normalized scale;
- existing legacy TB loss tests remain green.

### Static checks

Command:

```powershell
.\.venv\python.exe -m py_compile factor_gfn/gfn/log_partition_variance.py tests/test_log_partition_variance.py
```

Result: passed with no output.

Public export and gradient smoke command:

```powershell
.\.venv\python.exe -c "import torch; from factor_gfn.gfn import direct_log_partition_variance, LogPartitionVarianceLoss; pf=torch.tensor([-1.,-3.],requires_grad=True); o=direct_log_partition_variance(sum_log_pf=pf,sum_log_pb=torch.zeros(2),log_reward=torch.zeros(2)); o.loss.backward(); print(float(o.loss), pf.grad.tolist(), len(list(LogPartitionVarianceLoss().parameters())))"
```

Result:

```text
1.0 [1.0, -1.0] 0
```

The final source check confirmed the production loss contains `centered.square().mean()`, detaches only Reward/PB, contains no correction factor, and has no logZ or Parameter state.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Structural exhaustive enumeration: not run
Training: not run
Step 3 or later implementation: not started
```

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 2 issues: none
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 3. Before Step 3, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 3 file diffs.

## Entry 003 — Step 3 transactional condition scheduler

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected the exact existing diffs for `complexity_scheduler.py` and `test_complexity_scheduler.py`; both had no local diff before Step 3.
- Inspected the existing `__init__.py` exports and preserved all prior A/B1, Step 1, and Step 2 changes.
- Confirmed that Step 3 is limited to the scheduler transaction contract and its focused tests; no Trainer integration was started.

### Work performed

- Added immutable `ConditionAssignment` tokens containing `cycle_index`, `condition_position_in_cycle`, and `condition_N`.
- Added `BalancedNodeCountScheduler.peek()` to return the pending assignment without consuming it or changing scheduler state.
- Added `BalancedNodeCountScheduler.commit(assignment)` to consume only the exact current assignment.
- Rejected stale, double, foreign, and incorrectly typed commit tokens.
- Preserved `next_node_count()` and `next_batch()` as legacy consuming APIs by routing them through peek/commit.
- Preserved the v1 scheduler state schema because pending state is derived completely from the existing permutation, position, and RNG state.
- At a cycle boundary, peek derives the next shuffled permutation from a copied RNG state; only commit mutates the scheduler and advances to the next cycle.
- Exported `ConditionAssignment` from `factor_gfn.gfn`.

### Files changed

```text
factor_gfn/gfn/complexity_scheduler.py           transactional API added
tests/test_complexity_scheduler.py               focused transactional tests added
factor_gfn/gfn/__init__.py                       ConditionAssignment export added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/__init__.py` was already modified before Step 3. This Step added only the `ConditionAssignment` import and `__all__` entry.

### Focused tests

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_complexity_scheduler -v
```

Result:

```text
Ran 12 tests in 6.361s
OK
```

Coverage included:

- repeated peek returns the same pending assignment and leaves scheduler state unchanged;
- no commit means no condition consumption;
- a matching commit advances exactly one position;
- stale, double, foreign, and incorrectly typed commits are rejected;
- each completed hybrid cycle contains conditions `1..15` exactly once with positions `0..14`;
- a fixed seed reproduces assignments across multiple cycles;
- state round-trip preserves the pending assignment and all future assignments, including at a cycle boundary;
- legacy `next_node_count()` and `next_batch()` balance and resume behavior remains green;
- existing conditioned-Trainer scheduling and retry regression tests remain green.

### Static checks

Commands:

```powershell
git diff --check -- factor_gfn/gfn/complexity_scheduler.py factor_gfn/gfn/__init__.py tests/test_complexity_scheduler.py
.\.venv\python.exe -m py_compile factor_gfn/gfn/complexity_scheduler.py tests/test_complexity_scheduler.py
```

Result: passed. Git emitted only the existing Windows LF-to-CRLF working-copy warnings; no whitespace error was reported.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Structural exhaustive enumeration: not run
Real training: not run
Step 4 or later implementation: not started
```

The focused scheduler module contains small synthetic conditioned-Trainer regression tests inherited from the existing test file. These are not real-data or real-Reward training.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 3 issues: none
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 4. Before Step 4, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 4 file diffs.

## Entry 004 — Step 4 configurable-K single-N batch collection

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected the existing diffs for `trainer.py`, `test_no_anchor_trainer.py`, and the sampler/test candidates.
- Confirmed that `trainer.py` and `test_no_anchor_trainer.py` already contained unrelated B1 policy-Adam/logZ-SGD work; preserved it.
- Confirmed that `policy_sampler.py` and its focused test file had no local diff and left the sampler unchanged.
- Confirmed that Step 4 is limited to collection; objective routing, backward, optimizer stepping, scheduler commit, and Step 5 exact compatibility were not started.

### Work performed

- Added immutable `SingleConditionBatchCollection` results with explicit requested, accepted, invalid, retry, retry-exhausted, round, and timing fields.
- Added `GFNTrainer.collect_single_condition_batch()` as an isolated collection helper.
- Passed `(condition_N,) * trajectories_per_batch` directly through the existing `sample_trajectories(..., target_node_counts=...)` path.
- Kept K as the validated `trajectories_per_batch` argument and did not read or hard-code the legacy `TrainingConfig.batch_size`.
- Validated K as an integer `>=2`, validated N against the actual feasible exact-node strata, and required the existing conditioned `grammar_hierarchical` policy path.
- Retried only rejected slots, always with the same condition N, through the existing configured exact-node retry budget.
- Attached only valid Reward assignments and retained each accepted trajectory in its original slot.
- Kept collection transaction-free: the helper does not call scheduler peek/commit, backward, optimizer step, or training counters.
- Exported `SingleConditionBatchCollection` from `factor_gfn.gfn`.
- Left the existing legacy mixed-N `_collect_conditioned_training_batch()` unchanged.

### Files changed

```text
factor_gfn/gfn/trainer.py                        single-N collection helper/result added
tests/test_hybrid_single_condition_batch.py      focused Step 4 tests added
factor_gfn/gfn/__init__.py                       collection result export added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/trainer.py` and `factor_gfn/gfn/__init__.py` were already modified before Step 4. This Step made only additive changes around the single-N helper/result and its export; the existing A/B1 optimizer work was preserved.

### Focused tests

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_single_condition_batch tests.test_complexity_scheduler -v
```

Result:

```text
Ran 16 tests in 20.139s
OK
```

Coverage included:

- fixed `N=7` for every requested and accepted trajectory;
- exact terminal node count remains `7`;
- configurable `K=3` and `K=5` while legacy `batch_size=2`, proving K is not inherited from the old mixed-batch setting;
- a rejected first round retries exactly the same N and only the pending slots;
- complete retry accounting and retry-exhaustion accounting;
- an incomplete batch does not change scheduler state, optimizer state, model parameters, trainer step, or optimizer-step counters;
- invalid N and K are rejected before sampling;
- all Step 3 scheduler tests remain green;
- existing legacy conditioned-Trainer balancing and retry tests remain green.

### Static checks

Commands:

```powershell
git diff --check -- factor_gfn/gfn/trainer.py factor_gfn/gfn/__init__.py tests/test_hybrid_single_condition_batch.py
.\.venv\python.exe -m py_compile factor_gfn/gfn/trainer.py tests/test_hybrid_single_condition_batch.py
```

Result: passed. Git emitted only Windows LF-to-CRLF working-copy warnings; no whitespace error was reported.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Structural exhaustive enumeration: not run
train_step / optimizer update: not run by the new Step 4 tests
Real training: not run
Step 5 or later implementation: not started
```

The included legacy scheduler regression module contains its pre-existing small synthetic Trainer update tests. No real-data or real-Reward training was executed.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 4 issues: none
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 5. Before Step 5, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 5 file diffs.

## Entry 005 — Step 5 existing N=1/2 exact-Z compatibility

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected exact/reuse diffs for `exhaustive.py`, `exhaustive_registry_reuse.py`, `loss.py`, `trainer.py`, `no_anchor_config.py`, `hybrid_config.py`, and related tests.
- Confirmed that `exhaustive.py`, `exhaustive_registry_reuse.py`, and `loss.py` had no local diff and did not modify them.
- Confirmed that `trainer.py`, `no_anchor_config.py`, and some legacy tests contain earlier unrelated work and preserved it.
- Confirmed that Step 5 is compatibility verification only; no hybrid objective routing, optimizer, checkpoint, real Reward, or real asset generation was started.

### Work performed

- Added a focused compatibility test using a temporary synthetic legacy `max_depth=6, max_nodes=20` exhaustive registry containing only N=1/2.
- Reused the existing `ExhaustiveRegistry`, `ExactMassResult`, `prove_exhaustive_stratum_reuse()`, and `ProvenExhaustiveRewardLookup` implementations without modification.
- Compared the source `6/20` and target hybrid `5/15` N=1/2 canonical structural-hash sets and confirmed exact equality despite different global search-space fingerprints.
- Verified both strata produce valid per-N reuse proofs bound to semantics and exact aggregation fingerprints.
- Verified registry Reward lookup uses cached values and does not invoke the Reward provider after the temporary source registry is prepared.
- Loaded existing exact mass values through `TrajectoryBalanceLoss.set_exact_log_z()` with hybrid `max_nodes=15` and exact conditions `(1, 2)`.
- Verified exact values reside in float64 non-gradient buffers, exact masks cover only N=1/2, and the exact buffer is not an optimizer parameter.
- Verified a Reward-semantics mismatch fails before any fixed-buffer mutation.
- Found no type, manifest, per-stratum identity, exact-mass, or fixed-buffer incompatibility; therefore no production adapter or exact-infrastructure change was needed.
- Left the legacy high-level `configure_no_anchor_exhaustive_registry()` mode gate unchanged. Hybrid Trainer wiring belongs to Step 6 and must reuse the proven lower-level path rather than redefine it.

### Files changed

```text
tests/test_hybrid_exact_z_compatibility.py       focused compatibility tests added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

No production code was modified in Step 5.

### Focused tests

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_exact_z_compatibility tests.test_no_anchor_foundations tests.test_conditional_normalizer -v
```

Result:

```text
Ran 17 tests in 13.652s
OK
```

Coverage included:

- legacy `6/20` to hybrid `5/15` N=1/2 canonical identity;
- reuse with different global search-space fingerprints;
- exact per-stratum proof and exact aggregation fingerprints;
- cached valid/invalid Reward lookup behavior and fail-closed semantics checks;
- zero Reward-provider evaluations during target proof/lookup reuse;
- hybrid-length float64 exact buffers, masks, no gradient, and exclusion from parameters;
- atomic failure before buffer mutation on semantics mismatch;
- existing no-anchor reuse foundations remain green;
- existing exact/learned conditional-normalizer math and optimizer regressions remain green.

### Static checks

Commands:

```powershell
git diff --check -- tests/test_hybrid_exact_z_compatibility.py
.\.venv\python.exe -m py_compile tests/test_hybrid_exact_z_compatibility.py
```

Result: passed with no output.

After removing one unused test import, `py_compile` and `git diff --check` were rerun during final verification.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Real exhaustive registry or exact-Z asset generation: not run
Real training: not run
Step 6 or later implementation: not started
```

The focused test did perform small synthetic N=1/2 canonical enumeration and built a temporary registry that was deleted at teardown. It did not access or mutate any project registry, checkpoint, cache, or real-data artifact. Existing regression tests include small synthetic Trainer updates only.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 5 issues: none
Production adapter required: no
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 6. Before Step 6, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 6 file diffs.

## Entry 006 — Step 6 hybrid routing and policy-only optimizer

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short`.
- Inspected existing diffs for `trainer.py`, `loss.py`, `log_partition_variance.py`, `hybrid_config.py`, `exhaustive_registry_reuse.py`, `__init__.py`, and related tests.
- Confirmed that `trainer.py`, `__init__.py`, and several legacy tests contain prior A/B1 and Steps 1–5 work; preserved all unrelated changes.
- Confirmed that `loss.py`, `log_partition_variance.py`, and exact/reuse production modules had no pre-Step-6 local diff except the already-added Step 2 LPV module.
- Limited Step 6 to objective routing, shared-policy optimization, exact-registry wiring, and transaction ordering. Step 7 diagnostics/counters and Step 8 checkpoint/runner were not started.

### Work performed

- Added an isolated `HybridVarianceTrainer` rather than branching through the legacy Trainer's learned-logZ update path.
- Reused the Step 4 `collect_single_condition_batch()` helper through the existing `GFNTrainer` interface while providing hybrid-specific initialization, Reward lookup, and retry-budget behavior.
- Created exactly one shared `ForwardPolicyNetwork` for all N=1..15.
- Created exactly one Adam optimizer with one parameter group named `policy`; its parameter identities exactly equal the shared model parameter identities.
- Added no logZ parameter, logZ optimizer, logZ learning rate, logZ gradient clipping, EMA, rolling state, or persistent LPV normalizer to hybrid Trainer state.
- Added `FixedExactTrajectoryBalanceLoss`, which implements the existing exact-TB formula with only float64 fixed exact-logZ buffers and no parameters.
- Verified the fixed-only exact objective is numerically identical to the existing `TrajectoryBalanceLoss` exact branch for the same trajectories and exact values.
- Routed N=1/2 to fixed exact TB and N=3..15 to the Step 2 direct `LogPartitionVarianceLoss`.
- Required each routed batch to contain exactly configured K trajectories under one fixed condition.
- Added hybrid high-level registry configuration that reuses the existing per-stratum canonical proof, exact mass, and cached Reward lookup path proven in Step 5.
- Kept registry setup atomic until both N=1/2 proofs and exact results have been read successfully; no Reward provider evaluation occurs during target proof/reuse.
- Added the hybrid update transaction: collect pending N, compute routed objective, backward, clip shared-policy gradients, `optimizer.step()`, then `scheduler.commit()`, then increment the optimizer-step counter.
- Incomplete collection returns without objective, optimizer step, scheduler commit, or optimizer-step increment.
- Exported the fixed exact objective, hybrid Trainer, and Step 6 output types from `factor_gfn.gfn`.
- Left the existing `GFNTrainer`, learned-logZ `TrajectoryBalanceLoss`, no-anchor optimizer variants, and legacy checkpoint behavior intact.

### Files changed

```text
factor_gfn/gfn/hybrid_trainer.py                 added
factor_gfn/gfn/loss.py                           fixed-exact-only TB module added
tests/test_hybrid_variance_trainer.py            added
factor_gfn/gfn/__init__.py                       Step 6 public exports added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/__init__.py` was already modified before Step 6. This Step added only the fixed-exact and hybrid-Trainer imports/exports. The legacy `trainer.py` was deliberately not modified during Step 6.

### Focused tests

Command 1:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_variance_trainer tests.test_log_partition_variance tests.test_gfn_loss -v
```

Result:

```text
Ran 21 tests in 20.617s
OK
```

Command 2:

```powershell
.\.venv\python.exe -m unittest tests.test_conditional_normalizer tests.test_complexity_scheduler tests.test_hybrid_single_condition_batch tests.test_hybrid_variance_config -v
```

Result:

```text
Ran 32 tests in 20.464s
OK
```

Command 3:

```powershell
.\.venv\python.exe -m unittest tests.test_no_anchor_trainer -v
```

Result:

```text
Ran 16 tests in 11.433s
OK
```

Combined focused result:

```text
Ran 69 tests
69 passed
```

Coverage included:

- N=1 routes to exact TB and matches the existing exact-TB loss/deltas;
- N=3 routes to direct LPV using population minibatch variance;
- both objectives backpropagate through and update the same shared policy;
- hybrid optimizer contains only shared-policy parameters;
- fixed exact TB and LPV modules contain no parameters;
- no learned-logZ-named parameter or normalizer optimizer exists in hybrid mode;
- hybrid registry setup reuses N=1/2 without new Reward evaluation;
- successful update commits only after policy optimizer step;
- incomplete collection leaves policy, optimizer-step count, and scheduler unchanged;
- Step 1 configuration, Step 2 LPV, Step 3 scheduler, and Step 4 collection regressions remain green;
- legacy scalar TB, conditional learned-logZ Adam/SGD, exact registry, calibration, historical/targeted initialization, and no-anchor checkpoint round-trip tests remain green.

### Static checks and smoke

Commands:

```powershell
git diff --check -- factor_gfn/gfn/loss.py factor_gfn/gfn/hybrid_trainer.py factor_gfn/gfn/__init__.py tests/test_hybrid_variance_trainer.py
.\.venv\python.exe -m py_compile factor_gfn/gfn/loss.py factor_gfn/gfn/hybrid_trainer.py factor_gfn/gfn/__init__.py tests/test_hybrid_variance_trainer.py
.\.venv\python.exe -c "from factor_gfn.gfn import HybridVarianceTrainer, FixedExactTrajectoryBalanceLoss, build_stage5_hybrid_variance_5_15_config, SyntheticRewardProvider; t=HybridVarianceTrainer(build_stage5_hybrid_variance_5_15_config(max_cycles=1, trajectories_per_batch=2), SyntheticRewardProvider()); print([g['name'] for g in t.optimizer.param_groups], len(list(t.exact_tb_loss.parameters())), len(list(t.log_partition_variance_loss.parameters())), t.optimizer_contract()['normalizer_optimizer'])"
```

Result:

```text
['policy'] 0 0 None
```

Static checks passed. Git emitted only Windows LF-to-CRLF working-copy warnings; no whitespace error was reported.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Real exhaustive registry or exact-Z asset generation: not run
Real training: not run
Hybrid checkpoint/runner: not implemented or run
Step 7 or later implementation: not started
```

Tests used manual differentiable trajectories, synthetic Reward, temporary N=1/2 registries, and existing small synthetic legacy Trainer tests. They did not access or mutate project data, registries, checkpoints, caches, or real-run artifacts.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 6 issues: none
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 7. Before Step 7, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 7 file diffs.

## Entry 007 — Step 7 diagnostics and counters

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short` and confirmed the existing dirty worktree remains present.
- Inspected the current Step 6 `hybrid_trainer.py`, its focused tests, the public exports, trajectory fields, collection counters, TB output, LPV output, and canonical expression hash implementation.
- Confirmed that `hybrid_trainer.py` and `test_hybrid_variance_trainer.py` are Step 6 files still untracked relative to HEAD; preserved the existing Step 6 implementation and modified only the Step 7 surface.
- Limited Step 7 to successful-update diagnostics, counters, and in-memory history. Hybrid checkpoint persistence and runner integration remain Step 8 work and were not started.

### Work performed

- Added immutable common, exact-TB, and LPV diagnostic result types.
- Kept exact-TB and LPV diagnostic schemas physically distinct so an LPV record contains no exact logZ or TB-delta fields, including no `selected_logZ` or `learned_logZ` aliases.
- Added a flat `to_dict()` representation for diagnostic persistence/inspection without synthesizing fields from the other objective.
- Recorded the frozen common successful-update fields: cycle and condition position, condition N, objective kind, global optimizer step, batch and cumulative trajectory units, collection counts, Reward population statistics, PF/PB means, mean trajectory length, terminal success rate, and pre-clip policy gradient norm.
- Defined `reward_std` explicitly with population convention `ddof=0` and `trajectory_length` explicitly as the arithmetic mean number of actions in the successful batch.
- Recorded exact-TB diagnostics: fixed exact logZ, TB loss, and TB delta population mean/std/RMS.
- Recorded LPV diagnostics: zeta mean/std/population variance, direct variance loss, centered-zeta RMS, and terminal diversity.
- Computed terminal diversity from `terminal_expression.canonicalize().structural_hash()`, not the trajectory state hash; commutative expression variants therefore share one stable identity.
- Added `total_trajectories_seen` and in-memory `diagnostic_history` to the isolated hybrid Trainer.
- Preserved the successful update order as `optimizer.step() -> scheduler.commit() -> optimizer/trajectory counters -> diagnostic history`; each successful batch increments the trajectory counter by accepted K.
- Kept incomplete collection transactional: no optimizer step, scheduler commit, counter increment, or diagnostic-history entry.
- Exported the new diagnostic result types from `factor_gfn.gfn`.
- Added focused tests for objective-specific field separation, population statistics, canonical diversity, successful counters/history, and incomplete-batch non-mutation.

### Files changed

```text
factor_gfn/gfn/hybrid_trainer.py                 Step 7 diagnostics/counters added
tests/test_hybrid_variance_trainer.py            Step 7 focused tests added
factor_gfn/gfn/__init__.py                       diagnostic public exports added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/__init__.py` contained prior A/B1 and Steps 1–6 changes. Step 7 added only the hybrid diagnostic imports and `__all__` entries.

### Focused tests

Command 1:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_variance_trainer -v
```

Result:

```text
Ran 8 tests in 22.595s
OK
```

Command 2:

```powershell
.\.venv\python.exe -m unittest tests.test_log_partition_variance tests.test_gfn_loss tests.test_hybrid_single_condition_batch tests.test_complexity_scheduler -v
```

Result:

```text
Ran 31 tests in 20.860s
OK
```

Command 3:

```powershell
.\.venv\python.exe -m unittest tests.test_no_anchor_trainer -v
```

Result:

```text
Ran 16 tests in 12.649s
OK
```

Combined focused result:

```text
Ran 55 tests
55 passed
```

Coverage included:

- common counters and units reflect the post-commit successful update;
- exact updates expose only exact-TB fields and satisfy `tb_delta_rms == sqrt(tb_loss)`;
- LPV updates expose only LPV fields and contain no exact/TB/logZ aliases;
- Reward standard deviation uses population `ddof=0`;
- canonical commutative terminals with distinct trajectory state hashes count as one terminal identity;
- incomplete collection leaves both trajectory counter and diagnostic history unchanged;
- existing direct LPV, TB, single-N collector, transactional scheduler, and legacy no-anchor Trainer regressions remain green.

### Static checks and smoke

Commands:

```powershell
git diff --check -- factor_gfn/gfn/__init__.py tests/test_hybrid_variance_trainer.py
.\.venv\python.exe -m py_compile factor_gfn/gfn/hybrid_trainer.py factor_gfn/gfn/__init__.py tests/test_hybrid_variance_trainer.py
.\.venv\python.exe -c "from factor_gfn.gfn import HybridCommonDiagnostics, HybridExactTBDiagnostics, HybridLPVDiagnostics, HybridUpdateDiagnostics; print(HybridCommonDiagnostics.__name__, HybridExactTBDiagnostics.__name__, HybridLPVDiagnostics.__name__, HybridUpdateDiagnostics)"
```

Result: passed. Git emitted only the existing Windows LF-to-CRLF working-copy warning; no whitespace error was reported. A source scan also confirmed the hybrid Trainer contains none of the forbidden complexity-boundary, `selected_logZ`, or `learned_logZ` diagnostic names.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Real exhaustive registry or exact-Z asset generation: not run
Real training: not run
Hybrid checkpoint/runner: not implemented or run
Step 8 or later implementation: not started
```

Tests used manual differentiable trajectories, synthetic Reward, temporary N=1/2 registries, and existing small synthetic legacy Trainer tests. They did not access or mutate project data, registries, checkpoints, caches, or real-run artifacts.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 7 issues: none
```

One combined patch application reported a context mismatch at the already-dirty public export file after the production-file hunks had applied. The export-only patch was then applied separately; static checks, imports, and all focused tests passed.

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 8. Before Step 8, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 8 file diffs.

## Entry 008 — Step 8 hybrid checkpoint and runner

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short` and the exact current diffs of `checkpoint.py`, `search_runner.py`, `no_anchor_search_runner.py`, their legacy tests, and the Step 6–7 hybrid Trainer surface.
- Confirmed that all three existing legacy checkpoint/runner modules contain prior unrelated no-anchor/A-B1 changes.
- Chose new isolated hybrid checkpoint and runner modules so Step 8 would not branch through, overwrite, or redefine any legacy schema.
- Limited Step 8 to persistence, deterministic resume, and step-aligned runner behavior. Step 9 accumulated regression/synthetic smoke and real training were not started.

### Work performed

- Added distinct checkpoint schema `factor_gfn.checkpoint.hybrid_variance.v1` and objective marker `hybrid_variance`.
- Added atomic checkpoint saving through a sibling temporary file followed by `os.replace()`.
- Persisted the shared policy state and exactly one policy-only optimizer state/contract.
- Persisted fixed exact N=1/2 buffers, fixed-state manifest, exact-mass manifests, and exhaustive-registry reuse proofs.
- Persisted the full transactional scheduler state, including cycle index, current permutation, condition position, and scheduler RNG state.
- Persisted global optimizer step, total successful training trajectories, objective-specific diagnostic history, and aggregate diagnostic counters.
- Persisted Python, NumPy, Torch CPU, and available Torch CUDA RNG state.
- Added strict load validation for hybrid schema, objective mode, config fingerprint, Reward-provider fingerprint, device type, optimizer contract, exact buffers/masses/proofs, history/counter consistency, scheduler state, and RNG shape/device compatibility before applying runtime state.
- Reconstructed exact-TB and LPV history as their distinct immutable Step 7 diagnostic types; no cross-objective fake fields are synthesized.
- Added no learned-logZ optimizer state, EMA/rolling logZ, or persistent LPV normalizer. The existing optimizer contract retains explicit `learned_log_z=False` and `normalizer_optimizer=None` absence markers.
- Added hybrid/legacy cross-loading rejection in both directions without modifying the legacy checkpoint loader.
- Added thin `HybridVarianceTrainer.save_checkpoint()` and `load_checkpoint()` wrappers.
- Added isolated runner schema `factor_gfn.hybrid_variance_runner.v1` with exclusive run-directory creation, an authoritative latest checkpoint, atomic diagnostics JSONL, and atomic runner state.
- The runner performs its first post-update write only after `train_step()` has completed `optimizer.step() -> scheduler.commit() -> counters`; that first write is the atomic checkpoint.
- Incomplete collection attempts do not replace the latest checkpoint, diagnostics, or runner state.
- Resume treats the checkpoint as authoritative, verifies the run manifest, restores deterministic mid-cycle state, and only repairs diagnostics JSONL when it is an exact prefix of checkpoint history.
- Exported all Step 8 checkpoint and runner APIs from `factor_gfn.gfn`.
- Left the dirty legacy `checkpoint.py`, `search_runner.py`, and `no_anchor_search_runner.py` untouched by Step 8.

### Files changed

```text
factor_gfn/gfn/hybrid_checkpoint.py              added
factor_gfn/gfn/hybrid_search_runner.py           added
tests/test_hybrid_checkpoint_runner.py           added
factor_gfn/gfn/hybrid_trainer.py                 thin save/load wrappers added
factor_gfn/gfn/__init__.py                       Step 8 public exports added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

`factor_gfn/gfn/__init__.py` and `hybrid_trainer.py` already contained prior work. Step 8 added only the checkpoint/runner surface described above.

### Focused tests

Initial new-test command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_checkpoint_runner -v
```

Initial result:

```text
Ran 6 tests in 9.064s
FAILED (failures=1)
```

The failure was a test-only false positive: it treated the explicit optimizer-contract absence marker `learned_log_z=False` as learned-logZ state. The assertion was narrowed to require `learned_log_z is False`, `normalizer_optimizer is None`, and absence of learned/EMA/rolling optimizer-state fields. No production change was needed.

Rerun result:

```text
Ran 6 tests in 9.250s
OK
```

Final hybrid/checkpoint/scheduler command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_checkpoint_runner tests.test_hybrid_variance_trainer tests.test_complexity_scheduler -v
```

Result:

```text
Ran 26 tests in 36.463s
OK
```

Legacy checkpoint/no-anchor command:

```powershell
.\.venv\python.exe -m unittest tests.test_checkpoint_no_anchor tests.test_no_anchor_trainer -v
```

Result:

```text
Ran 19 tests in 13.155s
OK
```

Combined final focused result:

```text
Ran 45 tests
45 passed
```

Coverage included:

- every frozen checkpoint field and the policy-only optimizer contract;
- exact buffers, exact masses, and reuse-proof manifests;
- absence of learned/EMA/rolling LPV normalizer state;
- exact mid-cycle policy, optimizer, scheduler, history, counter, and RNG round-trip;
- equality of the next 20 scheduler assignments after resume;
- Python, NumPy, and Torch CPU RNG next-value equality after resume;
- hybrid/legacy schema rejection in both directions;
- config mismatch rejection before model/counter mutation;
- runner checkpoint ordering after commit/counters;
- runner resume and diagnostics reconciliation;
- incomplete attempts leave checkpoint and runner state byte-for-byte unchanged;
- existing Step 3, Step 6, Step 7, legacy Stage 4 checkpoint, and no-anchor checkpoint/Trainer regressions remain green.

### Static checks and public-export smoke

Commands:

```powershell
git diff --check -- factor_gfn/gfn/__init__.py factor_gfn/gfn/hybrid_trainer.py tests/test_hybrid_checkpoint_runner.py
.\.venv\python.exe -m py_compile factor_gfn/gfn/hybrid_checkpoint.py factor_gfn/gfn/hybrid_search_runner.py factor_gfn/gfn/hybrid_trainer.py factor_gfn/gfn/__init__.py tests/test_hybrid_checkpoint_runner.py
.\.venv\python.exe -c "import factor_gfn.gfn as g; names=['HYBRID_CHECKPOINT_SCHEMA','HYBRID_VARIANCE_RUNNER_SCHEMA','HybridVarianceRunner','create_hybrid_variance_runner','resume_hybrid_variance_runner','save_hybrid_checkpoint','load_hybrid_checkpoint']; print([(name, hasattr(g,name)) for name in names]); print('all_unique', len(g.__all__)==len(set(g.__all__)))"
```

Result: passed. All Step 8 exports are available and `__all__` contains no duplicates. Git emitted only the existing Windows LF-to-CRLF working-copy warning; no whitespace error was reported.

### Not executed

```text
Full regression suite: not run
Real Reward: not run
Real exhaustive registry or exact-Z asset generation: not run
Real training: not run
Formal real-data hybrid runner creation: not run
Step 9 implementation/execution: not started
```

Tests used manual differentiable trajectories, synthetic Reward, a temporary N=1/2 registry, and temporary checkpoint/run directories. They did not access or mutate project data, production registries, checkpoints, caches, or real-run artifacts.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 8 issues: none
```

### Next gate

Stop here. Wait for an explicit user instruction before starting Step 9. Before Step 9, re-read the frozen design and this log in full, then inspect the current worktree and exact Step 9 file diffs.

## Entry 009 — Step 9 focused regression and synthetic smoke

Date: 2026-08-16  
Status: Complete

### Pre-Step gate

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` in full.
- Re-read this development log in full.
- Inspected `git status --short` and the accumulated hybrid/legacy diffs from Steps 1–8.
- Inspected the actual conditioned sampler entry point and the Step 8 checkpoint/runner focused tests before defining the smoke boundary.
- Confirmed that Step 9 permits synthetic-only execution but still forbids real Reward, real data, formal training, and the full regression suite.
- Limited the new test artifact to a synthetic end-to-end smoke; no production code was modified in Step 9.

### Work performed

- Added one synthetic-only end-to-end smoke using `SyntheticRewardProvider`, CPU, `max_depth=5`, `max_nodes=15`, one cycle, and test-scale configurable `K=2`.
- Built a temporary synthetic legacy-shaped N=1/2 exhaustive registry, computed temporary exact masses, and reused it through the production hybrid equivalence path.
- Exercised the actual conditioned `sample_trajectories()` path rather than patched/manual trajectories.
- Exercised all 15 shuffled conditions exactly once through the production single-N collector, exact-TB/LPV routing, backward pass, policy-only optimizer, gradient clipping, transactional scheduler commit, Step 7 diagnostics, Step 8 runner, and atomic checkpoint.
- Completed 15 successful optimizer updates and 30 successful training trajectories.
- Verified exactly two `exact_tb` updates and thirteen `log_partition_variance` updates.
- Verified finite policy gradient norms for every update.
- Verified the completed-cycle checkpoint contains optimizer step 15, total trajectory count 30, and scheduler position 15.
- Created a fresh hybrid Trainer, re-established the same temporary exact-registry proof, resumed through the production runner, and verified completion status, counters, history, and scheduler state.
- Ran the accumulated focused regression modules from Steps 1–8, including the selected legacy compatibility suites.
- Did not modify any production module during Step 9.

### Files changed

```text
tests/test_hybrid_end_to_end_smoke.py             added
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md        updated
```

All other hybrid and legacy source/test changes shown by git status predate Step 9 and were left unchanged by this Step.

### Synthetic-only end-to-end smoke

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_end_to_end_smoke -v
```

Result:

```text
Ran 1 test in 72.448s
OK
```

Verified behavior:

```text
provider: SyntheticRewardProvider
device: CPU
cycles: 1
K: 2 (test scale, configurable-K path)
successful optimizer updates: 15
successful trajectories: 30
conditions covered: N=1..15 exactly once
exact-TB updates: 2
LPV updates: 13
checkpoint: written after every successful committed update
resume: completed-cycle state restored successfully
```

This smoke validates the full synthetic control flow and state contracts. It does not claim formal K=16 throughput, real-data readiness, Reward quality, factor quality, or training convergence.

### Accumulated focused regression

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_variance_config tests.test_log_partition_variance tests.test_complexity_scheduler tests.test_hybrid_single_condition_batch tests.test_hybrid_exact_z_compatibility tests.test_hybrid_variance_trainer tests.test_hybrid_checkpoint_runner tests.test_gfn_loss tests.test_no_anchor_config tests.test_no_anchor_foundations tests.test_conditional_normalizer tests.test_no_anchor_trainer tests.test_checkpoint_no_anchor -v
```

Result:

```text
Ran 103 tests in 101.690s
OK
```

Combined Step 9 result:

```text
Synthetic E2E smoke: 1/1 passed
Accumulated focused regression: 103/103 passed
Total: 104/104 passed
```

Coverage included:

- frozen 5/15 hybrid config and configurable K;
- direct LPV value, derivative scale, detach, and PF autograd contracts;
- transactional cycle scheduling and resume;
- fixed-N collection and retry behavior;
- legacy 6/20 to hybrid 5/15 exact N=1/2 reuse;
- objective routing and policy-only optimization;
- objective-specific diagnostics and trajectory counters;
- distinct hybrid checkpoint/runner schemas and deterministic resume;
- legacy scalar TB, no-anchor config, exact/learned conditional normalizer, registry foundations, Trainer, and checkpoint compatibility.

### Static checks

Commands:

```powershell
.\.venv\python.exe -m py_compile tests/test_hybrid_end_to_end_smoke.py
git diff --check -- STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md factor_gfn/gfn/__init__.py factor_gfn/gfn/complexity_scheduler.py factor_gfn/gfn/loss.py factor_gfn/gfn/trainer.py tests/test_complexity_scheduler.py tests/test_conditional_normalizer.py tests/test_no_anchor_trainer.py
```

Result: passed. Git emitted only existing Windows LF-to-CRLF working-copy warnings; no whitespace error was reported.

### Not executed

```text
Full unittest discovery suite: not run
Real Reward: not run
Real data loading/evaluation: not run
Production exhaustive registry or exact-Z asset generation: not run
Formal K=16 hybrid training: not run
CUDA hybrid training: not run
Stage 6 evaluation: not run
```

The smoke used only temporary files and removed its registry, checkpoint, runner directory, and diagnostics at test teardown. It did not access or mutate project data, production registries, checkpoints, caches, or run artifacts.

### Deviations and unresolved issues

```text
Frozen-contract deviations: none
Unresolved Step 9 issues: none
```

### Milestone gate

Steps 1–9 of the frozen hybrid-variance implementation are complete. Stop here for user review. Do not start real Reward evaluation, formal K=16 training, CUDA execution, or Stage 6 work without a new explicit instruction.

## Entry 010 — Phase A full regression suite

Date: 2026-08-16  
Status: **Not clean — 3 unrelated Notebook cleanliness failures**

### Scope and safeguards

- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` and this development log before execution.
- Inspected the dirty worktree and preserved all existing production, Notebook, data, registry, checkpoint, cache, and run artifacts.
- Did not modify production code, execute real Reward/data evaluation, or start real/formal training.
- Ran only the repository's complete unittest discovery suite and read-only failure attribution checks.

### Full regression command

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

Result:

```text
Ran 497 tests in 251.088s
494 passed
3 failed
0 skipped
```

### Failure attribution

All three failures came from `tests/test_no_anchor_notebooks.py`, whose shared `_read()` helper requires every code cell in an active Notebook to have `execution_count is None` and `outputs == []`.

```text
test_formal_stage5_notebook_is_locked_segmented_and_compact
  notebooks/run_stage5_no_anchor_formal_6_20.ipynb
  observed non-null execution counts: 1, 2, 3

test_logz_adam_lr2e2_experiment_a_is_safe_and_single_variable
  notebooks/run_stage5_logz_adam_lr2e2_ab.ipynb
  observed non-null execution counts: 3, 4, 5

test_logz_sgd_lr1e1_b1_has_two_stage_successful_update_gate
  notebooks/run_stage5_logz_sgd_lr1e1_b1.ipynb
  observed non-null execution counts: 9, 10, 11, 12, 13
```

Classification:

```text
Hybrid-caused regressions: none observed
Legacy regression: none observed in production behavior
Historical/unrelated failures: 3 Notebook cleanliness failures
Environment/dependency failures: none observed
```

Evidence for the unrelated/historical classification:

- The failing files are legacy/no-anchor formal or logZ A/B1 experiment Notebooks, not Hybrid modules or tests.
- Their filesystem last-write times are 2026-08-14 or 2026-08-15, before the 2026-08-16 Hybrid implementation work.
- The formal Notebook is an already-modified tracked file; the two A/B1 Notebooks are existing untracked files.
- Their code cells contain stored outputs as well as execution counts, consistent with prior interactive execution state.
- All Hybrid tests discovered by the full suite passed, including the synthetic E2E smoke, checkpoint/resume, objective routing, LPV, exact-Z compatibility, diagnostics, config, scheduler, and single-N collection tests.

### Freeze decision

The complete suite is not clean, so the requested Hybrid V1 frozen marker was intentionally withheld.

No Notebook was cleaned or rewritten automatically because those files contain user run outputs and unrelated working-tree changes. Phase B and Phase C were not started.

### Next gate

Stop here and wait for explicit user direction. To complete Phase A, the three active Notebooks must either be explicitly authorized for output/execution-count cleanup or the cleanliness contract must be deliberately changed; neither action is inferred from the regression request.

## Entry 011 — Phase A authorized Notebook execution-trace cleanup

Date: 2026-08-16  
Status: **Execution traces cleaned; two pre-existing Notebook control-contract failures remain**

### Authorized scope

The user explicitly authorized cleaning execution traces from exactly these three Notebooks:

```text
notebooks/run_stage5_no_anchor_formal_6_20.ipynb
notebooks/run_stage5_logz_adam_lr2e2_ab.ipynb
notebooks/run_stage5_logz_sgd_lr1e1_b1.ipynb
```

For every code cell, the cleanup changed only:

```text
execution_count -> null
outputs -> []
```

No source, Markdown, Notebook metadata, cell identity, training configuration, production code, data, registry, checkpoint, cache, or run artifact was changed. A semantic comparison against pre-cleanup copies confirmed that removing the two allowed execution fields makes every cleaned Notebook exactly equal to its pre-cleanup structure.

Pre-cleanup copies remain temporarily recoverable at:

```text
C:\Users\qiuyu\AppData\Local\Temp\codex_phase_a_notebook_cleanup_20260816
```

### Cleanup implementation note

The first `nbconvert` attempt emitted a missing-cell-ID normalization warning and added cell IDs to two old Notebooks. That result was not accepted. All three files were restored from the temporary copies and cleaned again with a JSON transformation that does not normalize the Notebook schema. Final semantic verification passed for all three files.

### Focused test

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_no_anchor_notebooks -v
```

Result:

```text
Ran 7 tests in 0.021s
5 passed
2 failed
0 skipped
```

The previously failing formal no-anchor Notebook cleanliness test now passes. The two A/B1 tests passed their cleanliness checks and then exposed pre-existing source-level run-control differences that had previously been masked by the earlier `execution_count` assertion:

```text
run_stage5_logz_adam_lr2e2_ab.ipynb
  expected contract marker: RUN_EXPERIMENT_A=False
  current Notebook source:   RUN_EXPERIMENT_A=True

run_stage5_logz_sgd_lr1e1_b1.ipynb
  expected contract markers: RUN_EXPERIMENT_B1=False
                             MODE='new'
                             CONTINUE_AFTER_SAFETY_GATE=False
  current Notebook source:   RUN_EXPERIMENT_B1=True
                             MODE='resume'
                             CONTINUE_AFTER_SAFETY_GATE=True
```

The B1 Notebook also contains a concrete resume run directory, consistent with an intentionally executed continuation configuration. These source-level controls were not changed because the user authorized execution-trace cleanup only.

### Phase A decision

- Hybrid-caused regression remains unobserved.
- The full suite was not rerun after this focused failure because the known focused prerequisite is not green.
- The Hybrid V1 frozen marker remains withheld.
- Phase B and Phase C were not started.

### Next gate

Stop and wait for explicit user direction on whether the two experiment Notebooks should be returned to their safety-locked source contracts. If authorized, change only the identified run-control fields, rerun the Notebook focused suite, then rerun full unittest discovery before recording the freeze marker.

## Entry 012 — Phase A completion and Hybrid V1 implementation freeze

Date: 2026-08-16  
Status: **Complete**

```text
HYBRID V1 IMPLEMENTATION FROZEN
```

### Authorized safety-lock restoration

The user authorized completing Phase A and restoring the two experiment Notebooks to their test-defined safe entry state.

Changed only these source-level run controls:

```text
notebooks/run_stage5_logz_adam_lr2e2_ab.ipynb
  RUN_EXPERIMENT_A=False

notebooks/run_stage5_logz_sgd_lr1e1_b1.ipynb
  RUN_EXPERIMENT_B1=False
  MODE='new'
  RESUME_RUN_DIR=None
  CONTINUE_AFTER_SAFETY_GATE=False
```

The execution-trace cleanup from Entry 011 remains in place for all three authorized Notebooks. No production code, Hybrid training logic, Reward, data, registry, checkpoint, cache, or run artifact was modified or executed.

### Focused Notebook regression

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_no_anchor_notebooks -v
```

Result:

```text
Ran 7 tests in 0.023s
OK
```

Additional static verification confirmed that all three Notebooks parse as JSON, every code cell parses as Python, every code-cell `execution_count` is null, and every code-cell `outputs` array is empty. `git diff --check` reported no whitespace errors.

### Complete regression suite

Command:

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

Result:

```text
Ran 497 tests in 236.051s
497 passed
0 failed
0 skipped
```

No Hybrid regression, legacy regression, historical failure, or environment/dependency failure remained in this run.

### Frozen Hybrid V1 contract

```text
max_depth = 5
max_nodes = 15
N = 1..15
K = 16 configurable
N=1/2 exact TB
N=3..15 direct LPV
shared conditional policy
15 single-N batches / cycle
15 optimizer updates / cycle
240 successful trajectories / cycle when K=16
policy Adam lr=1e-4
clip=5
no learned logZ for N>=3
```

### Execution boundary

```text
Real Reward: not run
Real data loading/evaluation: not run
Production exact-Z generation: not run
Formal K=16 training: not run
CUDA training: not run
Stage 6 execution: not run
Phase B: not started
Phase C: not started
```

### Next gate

Phase A is complete and Hybrid V1 implementation is frozen. Stop here. Do not begin Phase B or Phase C without a new explicit user instruction.

## Entry 013 — Phase B design freeze and B1 start

Date: 2026-08-16  
Status: **In progress**

### Authorization and pre-Step gate

- The user explicitly authorized Phase B and limited this turn to B1.
- Re-read `STAGE5_HYBRID_VARIANCE_DESIGN.md` and this development log.
- Inspected the dirty worktree and the existing diffs for the B1 target files.
- Added the frozen Phase B contracts and B1 artifact schema to the design
  source of truth before modifying production code.
- Confirmed that B2 candidate freeze, B3 Stage 6 overlay integration, B4
  Validation/equivalence completion, real Reward, real data, and real training
  remain out of scope for this Step.

### Frozen B1 boundary

```text
Persist only already-computed Train evaluation information.
Use canonical structural_hash as the only candidate identity.
Do not add Validation fields or new Train calculations.
Place persistence at the Hybrid runner/output boundary.
Keep HybridVarianceTrainer, LPV, exact TB, policy, optimizer, Reward formula,
Stage 6 selection, and Validation definitions unchanged.
Make writes atomic, recoverable, idempotent, and fail closed on divergence.
Do not freeze the candidate universe; that belongs to B2.
```

### Implementation completed

- Extended `RewardResult` only with the already-computed Train long-excess
  dates/values needed by Stage 6. The existing Reward calculation, validity,
  flooring, LPV, and exact-TB inputs are unchanged.
- Added incremental read-only access to `RealRewardProvider` evaluation records;
  no new interpretation or Train metric computation was introduced.
- Added `train_candidate_artifact.json` with a versioned Train-only schema,
  provider/context/implementation provenance, canonical `structural_hash`
  identity, the frozen Train fields, and first/last/visit provenance.
- Integrated atomic artifact commits at the Hybrid runner boundary after the
  committed checkpoint. Resume is idempotent and fails closed if artifact and
  checkpoint optimizer steps or evaluation contracts diverge.
- Preserved pre-B1 Hybrid run compatibility: a run manifest without an artifact
  declaration resumes as a legacy artifact-disabled run.
- Did not implement candidate-universe freeze, Validation, Stage 6 loading, or
  selection changes.

### Changed files

Production:

```text
factor_gfn/gfn/reward.py
factor_gfn/gfn/real_reward.py
factor_gfn/gfn/train_candidate_artifact.py
factor_gfn/gfn/hybrid_search_runner.py
factor_gfn/gfn/__init__.py
```

Focused tests:

```text
tests/test_train_candidate_artifact.py
```

Source-of-truth records:

```text
STAGE5_HYBRID_VARIANCE_DESIGN.md
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md
```

### Focused verification

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_train_candidate_artifact -v
```

Result:

```text
Ran 5 tests in 1.959s
OK
```

The tests cover the exact Train field contract, absence of Validation and
Stage-6-unused Reward fields, no second `FactorInterpreter` evaluation,
canonical duplicate collapse and visit provenance, runner persistence without
objective changes, atomic/idempotent resume behavior, fail-closed step
divergence, and unchanged Reward formula output.

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_gfn_reward tests.test_gfn_real_reward tests.test_hybrid_checkpoint_runner tests.test_hybrid_variance_trainer tests.test_log_partition_variance tests.test_gfn_loss tests.test_hybrid_exact_z_compatibility tests.test_backtest_stage6_train_reuse tests.test_backtest_stage6_mixed_evaluation -v
```

Result:

```text
Ran 69 tests in 52.712s
OK
```

Additional checks:

```text
py_compile passed for all B1 production modules and the focused test module.
git diff --check passed with no whitespace errors.
No real Reward, real data evaluation, real training, CUDA training, or Stage 6
workflow was run.
The full regression suite was not rerun; the focused set above covers B1 and
the directly adjacent legacy/Phase A contracts.
```

### Diff review

- No changes were made to `HybridVarianceTrainer`, LPV, exact TB, policy,
  optimizer, scheduler, Reward formula, Stage 6 screening thresholds, or
  Validation definitions.
- The artifact exposes no `reward_config` or `reward_floor`; only the two
  Barra/neutralization settings required to identify Train metric semantics are
  projected, while the full provider is bound by fingerprint.
- Existing unrelated dirty-worktree changes and generated/run artifacts were
  preserved.

## PHASE B — B1 COMPLETE

B1 is complete and verified. Stop here. B2 candidate freeze is not started and
requires a new explicit user instruction.

## Entry 014 — B1 final schema audit and implementation freeze

Date: 2026-08-16  
Status: **Frozen**

The user requested a final code-level audit before freezing B1. The audit
confirmed:

- the actual top-level and per-record fields now documented in Section 9.4 of
  `STAGE5_HYBRID_VARIANCE_DESIGN.md` match the writer implementation;
- `train_long_excess_dates` and `train_long_excess_values` are passed from the
  already-computed Reward evaluation long-excess series into `RewardResult`;
- artifact persistence consumes recorded provider evaluation results and does
  not call `FactorInterpreter` or recompute Train metrics;
- no file under `factor_gfn/backtest/` imports or reads the new Train candidate
  artifact, so current Stage 6 behavior is unchanged;
- B2 candidate freeze and B3 Stage 6 integration remain unimplemented.

Verification command:

```powershell
.\.venv\python.exe -m unittest tests.test_train_candidate_artifact -v
```

Result:

```text
Ran 5 tests in 1.834s
OK
```

Static search result:

```text
Stage 6 references to train_candidate_artifact / TRAIN_CANDIDATE_ARTIFACT: 0
```

```text
PHASE B — B1 IMPLEMENTATION FROZEN
```

Stop here. Do not enter B2 without a new explicit user instruction.

## Entry 015 — defer B2/B3/B4 and start real-training Notebook preflight

Date: 2026-08-16  
Status: **In progress**

The user froze B1 and explicitly deferred B2 candidate freeze, B3 Stage 6
artifact integration, and B4 Validation/screening integration until the formal
Stage 5 real training run is complete.

```text
B2 / B3 / B4 — DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE
```

The approved current work package is limited to:

1. creating a thin real Hybrid Stage 5 training-control Notebook;
2. using the frozen 5/15, K=16 production code and existing exact N=1/2 assets;
3. performing a no-optimizer-step readiness/preflight;
4. reporting readiness before any real one-cycle training run.

No B2/B3/B4 implementation, Stage 6 modification, Validation calculation,
candidate-universe freeze, or real optimizer step is authorized in this
preflight sub-step.

### Notebook created

```text
notebooks/run_stage5_hybrid_variance_real_5_15.ipynb
```

The Notebook is a thin orchestration layer with clean execution state. Its
cells cover:

1. environment/CUDA reporting;
2. the frozen 5/15, K=16 contract;
3. no-training real-data/provider/exact-registry/artifact preflight;
4. manually gated new/resume runner construction;
5. exactly one current condition cycle;
6. objective-specific and runtime/CUDA diagnostics;
7. Train artifact inspection;
8. checkpoint/resume verification without continuation;
9. an uncalled optional continuation function.

The manual gate is frozen at:

```text
RUN_REAL_ONE_CYCLE = False
MODE = 'new'
RESUME_RUN_DIR = None
```

### Structural focused test

Command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_real_training_notebook -v
```

Result:

```text
Ran 5 tests in 0.005s
OK
```

The test verifies clean Notebook execution state, Python parsing, the frozen
configuration, the disabled real-training gate, production API usage, absence
of training calls from preflight, required diagnostics/artifact/resume cells,
and absence of Stage 6/Validation/candidate-freeze integration.

### Project-level real preflight

The exact Cell 1–3 source was executed sequentially in the project interpreter
after the Jupyter kernel issue described below. It performed no Trainer update.

```text
Python: 3.12.13
PyTorch: 2.6.0+cu124
CUDA: ready
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
device: cuda:0
seed: 42

max_depth/max_nodes: 5/15
K: 16
conditions: 1..15
exact TB: 1/2
direct LPV: 3..15
optimizer steps/cycle: 15
trajectories/cycle: 240
config fingerprint: c18bc89438b6570ef4f528790da9a58cb3004f588e063ee63d774149d13ab9ec

Train data: ready
requested range: 2010-01-01..2018-12-31
actual range: 2010-01-04..2018-12-28
evaluation shape: [2187, 5424]
rebalance periods: 386

RealRewardProvider: ready
provider fingerprint: dce89ac7b21afdbeeab1da9d6e29d61094e2d927efa5946f964c51ce19ec6096
Validation/OOS loaded: false
FactorInterpreter probe: valid
FactorInterpreter evaluations: 1

exact registry: read-only
N=1 proof: ready
N=2 proof: ready
new exact Reward evaluations: 0

Train artifact writer: ready in temporary step-0 runner
initial artifact candidate_count: 0
initial checkpoint: ready
formal output root: runs/stage5_hybrid_variance_real_5_15
optimizer_step_after_preflight: 0
real training executed: false
```

The temporary preflight runner was automatically removed with its temporary
directory. No production run directory, checkpoint, artifact, data, registry,
or cache was changed.

### Jupyter runtime blocker

An in-memory nbclient execution attempt failed before Cell 1 because the
project virtual environment's IPython/kernel dependency set is incomplete:

```text
ipykernel 7.3.0 requires matplotlib-inline, which is not installed
ipython 9.16.1 requires jedi, which is not installed
ipython 9.16.1 requires matplotlib-inline, which is not installed
ipython 9.16.1 requires stack-data, which is not installed
```

The immediate kernel traceback was:

```text
ModuleNotFoundError: No module named 'stack_data'
Kernel died before replying to kernel_info
```

No dependency was installed or configuration changed automatically. The formal
one-cycle run remains blocked until the Notebook kernel dependency set is
repaired and the preflight is rerun successfully through the actual kernel.

### Changed files

```text
STAGE5_HYBRID_VARIANCE_DESIGN.md
STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md
notebooks/run_stage5_hybrid_variance_real_5_15.ipynb
tests/test_hybrid_real_training_notebook.py
```

No production Python module or Stage 6 file was modified.

### Stop gate

```text
B1: FROZEN
B2/B3/B4: DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE
REAL K=16 ONE-CYCLE TRAINING: NOT STARTED
```

Stop here and report the Jupyter dependency blocker. Do not install packages or
run the real one-cycle training without the user's next instruction.

## Entry 016 — repair project Notebook kernel and rerun Cell 1–3 preflight

Date: 2026-08-16  
Status: **Complete — no training executed**

The user authorized only the project virtual-environment repair and actual
Notebook-kernel preflight. The interpreter and kernelspec were confirmed before
installation:

```text
Python executable: D:\实习\Gflownet因子挖掘\.venv\python.exe
Python prefix: D:\实习\Gflownet因子挖掘\.venv
kernelspec: python3
kernelspec resource: .venv\share\jupyter\kernels\python3
kernel module: ipykernel_launcher
```

Installed with the project interpreter, without an upgrade flag:

```powershell
.\.venv\python.exe -m pip install matplotlib-inline jedi stack-data
```

Installed versions:

```text
matplotlib-inline 0.2.2
jedi 0.20.0
stack-data 0.6.3
asttokens 3.0.2  # required dependency of stack-data
```

No unrelated package was upgraded. Post-install `pip check` reported:

```text
No broken requirements found.
```

The actual `python3` Jupyter kernel then started successfully through nbclient
and executed only the Notebook prefix ending at Cell 3 (`real-preflight`). The
kernel emitted non-blocking Windows/sandbox warnings about the Proactor selector
thread and IPython profile permissions, but executed all requested cells and
shut down normally.

Cell 1–3 result:

```text
PREFLIGHT_READY: true
REAL_TRAINING_EXECUTED: false
CUDA: ready
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Train data: ready
RealRewardProvider: ready
FactorInterpreter probe: valid
exact N=1/2 read-only reuse: ready
Train artifact writer step-0 check: ready
optimizer_step_after_preflight: 0
```

The temporary step-0 runner was removed automatically. The formal output path
`runs/stage5_hybrid_variance_real_5_15` does not exist, confirming that no
formal run directory or checkpoint was created.

Notebook structural tests were rerun:

```text
Ran 5 tests in 0.005s
OK
```

Safety boundary retained:

```text
RUN_REAL_ONE_CYCLE = False
real K=16 training: not run
optimizer: not stepped
production code: not modified
Stage 6 / Validation / B2 / B3 / B4: not touched
```

Stop here. Real training remains a manual Notebook action controlled by the
user.

## Entry 017 — first real Hybrid cycle audit and continuation-budget finding

Date: 2026-08-16  
Status: **First cycle healthy; continuation requires an explicit budget decision**

Read-only audit target:

```text
runs/stage5_hybrid_variance_real_5_15/hybrid_5_15_k16_seed42_20260816T012450Z
```

Verified results:

```text
completed conditions: 15/15, exactly one each for N=1..15
global optimizer step: 15
total committed trajectories: 240
diagnostic records: 15
all diagnostic scalars finite: true
retry exhausted count: 0
invalid/retried evaluations: 60
unique candidate records: 230
artifact committed optimizer step: 15
checkpoint schema: factor_gfn.checkpoint.hybrid_variance.v1
artifact schema: factor_gfn.stage5_train_candidate_artifact.v1
N>=3 records missing train_long_excess: 0
Validation fields in candidate records: 0
full-cycle runtime: 1460.726 seconds
CUDA peak allocated bytes: 264873984
```

All LPV conditions had `unique_terminal_fraction=1.0`. No NaN, infinity,
sampling collapse, incomplete batch, retry exhaustion, checkpoint/artifact
step divergence, or missing Train series was observed. High-N invalid retries
and runtime increased with complexity but every batch completed. The reported
`policy_grad_norm` is the value returned by `clip_grad_norm_` before clipping;
the actual update still used the frozen maximum norm of 5.

The current run cannot safely continue by editing one Notebook number:

```text
FORMAL_MAX_CYCLES = 1
runner_state.complete = true
configured total optimizer steps = 15
```

Changing `FORMAL_MAX_CYCLES` to 6 changes the frozen config fingerprint. Both
the runner manifest and checkpoint loader reject such a resume before trainer
mutation. The existing `run_additional_cycles(runner, 5)` also returns without
updates because `runner.complete` is already true.

No in-memory config mutation, checkpoint rewrite, artifact rewrite, Notebook
change, or additional training was performed. The protected one-cycle run is
unchanged.

Two safe choices remain and require user selection:

1. recommended: preserve the audited run and start a new planned six-cycle run,
   using the first cycle as the same safety gate and then five more cycles;
2. implement and test an explicit cycle-budget extension/migration contract for
   the current checkpoint/run/artifact before resuming it for five cycles.

Stop before either choice because they differ in model-state provenance and
checkpoint/public-interface semantics.

## Entry 018 — configure a protected fresh 1+5 cycle run

Date: 2026-08-16  
Status: **Ready for user-controlled execution; no training executed**

The user selected the recommended continuation strategy:

```text
preserve the audited one-cycle run unchanged
create a fresh run with a planned total budget of 6 cycles
execute the first cycle as a separate safety gate
execute the remaining 5 cycles only through a second manual gate
```

Notebook changes:

```text
FORMAL_MAX_CYCLES = 6
planned optimizer steps = 90
planned successful trajectories = 1440

RUN_REAL_ONE_CYCLE = False
ADDITIONAL_CYCLES = 5
RUN_ADDITIONAL_FIVE_CYCLES = False
```

The first-cycle cell now requires an optimizer-step-zero runner, preventing an
accidental second execution of that cell. The continuation cell requires
exactly one completed cycle (step 15), prints one progress record per successful
condition update, runs exactly 75 additional optimizer steps, and verifies the
final step-90 checkpoint/artifact boundary and 1,440 trajectory count.

The original audited run remains unchanged and complete at step 15:

```text
runs/stage5_hybrid_variance_real_5_15/hybrid_5_15_k16_seed42_20260816T012450Z
```

Focused command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_real_training_notebook tests.test_hybrid_variance_config tests.test_hybrid_checkpoint_runner -v
```

Result:

```text
Ran 21 tests in 10.278s
OK
```

The actual project Jupyter kernel executed only Cell 1–3 after the change:

```text
PREFLIGHT_READY: true
REAL_TRAINING_EXECUTED: false
CUDA: ready
planned_total_optimizer_steps: 90
planned_total_trajectories: 1440
new config fingerprint: 42c41f92b4b6772db96ffec974d750122eb5bc4a5fa44d491d23e20a25124e42
optimizer_step_after_preflight: 0
```

The preflight step-zero runner used a temporary directory and was removed. No
new formal run directory, checkpoint, or Train artifact was created. No
production module, existing run artifact, Stage 6 code, Validation path, or
B2/B3/B4 implementation was changed.

Stop here. The user controls both real-training switches manually in the
Notebook.

## Entry 019 — simplify Notebook budget to one 100-cycle run with cumulative targets

Date: 2026-08-16  
Status: **Complete; no training executed**

The user replaced the temporary planned-six-cycle workflow with one formal run
whose immutable budget is fixed before creation:

```text
FORMAL_MAX_CYCLES = 100
planned optimizer steps = 1500
planned successful trajectories = 24000
```

This is an upper bound bound to the run config/fingerprint, not an instruction
to train immediately. Cell 5 remains restricted to a fresh step-zero runner and
still performs exactly one complete cycle (15 successful optimizer updates and
240 accepted trajectories).

Cell 9 now exposes the cumulative-target helper:

```python
run_until_cycle(active_runner, target_cycle=6)
run_until_cycle(active_runner, target_cycle=100)
```

The first call advances an already completed 1-cycle run by five more cycles to
six total. The second advances the same 6-cycle run by 94 more cycles to 100
total. The helper rejects booleans/non-positive values, targets above the frozen
run budget, targets behind the current completed cycle, and starts from a
mid-cycle boundary. It checks the cumulative optimizer-step and trajectory
counts plus artifact committed-step alignment after reaching the target. The
saved Notebook keeps `RUN_TO_TARGET_CYCLE = False`. After a continuation, the
Notebook rebinds `runner` to the active resumed runner so rerunning Cells 6–8
inspects the latest diagnostics, artifact, checkpoint, and resume state rather
than the pre-continuation in-memory object.

All Markdown instruction cells were rewritten in Chinese. No production
trainer, runner, scheduler, objective, Reward, policy, optimizer, checkpoint,
artifact, or fingerprint implementation was changed.

Focused command:

```powershell
.\.venv\python.exe -m unittest tests.test_hybrid_real_training_notebook -v
```

Result:

```text
Ran 8 tests
OK
```

Static checks also confirmed that the Notebook remains clean (all execution
counts null and outputs empty), all code cells parse, the real-training and
continuation gates remain false, and no legacy fixed-additional-cycle control is
present. A behavior-level fake runner test verified that the same run advances
from cumulative cycle 1 to 6 (five cycles in the call) and then from 6 to 100
(94 cycles in the call). `git diff --check` passed for the scoped files.

A read-only run-directory check found an additional pre-existing partial run:

```text
runs/stage5_hybrid_variance_real_5_15/hybrid_5_15_k16_seed42_20260816T020503Z
global optimizer step: 4
total trajectories: 64
config fingerprint: c18bc89438b6570ef4f528790da9a58cb3004f588e063ee63d774149d13ab9ec
```

That fingerprint is the one-cycle config fingerprint, not the new 100-cycle
fingerprint `0fe6af55b7b6ab0078df051f76ef6d342478ce35850f541af078033907f16243`.
The partial run was not modified and must not be resumed under the 100-cycle
config. A fresh Notebook kernel and a new run are required for the new formal
workflow.

Stop here. Real training remains entirely user-controlled in the Notebook;
B2/B3/B4 remain deferred.

## Entry 020 — formal 100-cycle run completed and reporting v1 frozen

Date: 2026-08-17  
Status: **Complete**

The user completed the formal run:

```text
runs/stage5_hybrid_variance_real_5_15/
  hybrid_5_15_k16_seed42_20260816T025559Z
```

Final aligned state:

```text
complete = true
pending_assignment = none
optimizer steps = 1500
successful trajectories = 24000
unique candidates = 21261
config fingerprint = 0fe6af55b7b6ab0078df051f76ef6d342478ce35850f541af078033907f16243
```

Checkpoint, runner state, diagnostics and candidate artifact all terminate at
step 1500. The completed run passed the Stage 5 completion audit and was accepted
as the Raw Daily Baseline Stage 5 source. Training health was accepted with the
explicit clipping caveat: 90.1% of updates had pre-clip norm above 5 and the
trigger rate for N=4..15 was approximately 100%. No learning rate, clipping,
Reward, objective or Conditional-N contract was changed.

The real B2 report was rendered and presentation-reviewed, then frozen as:

```text
Raw Daily Baseline / Stage 5 Reporting v1
outputs/stage5_reporting/report_manifest.json
15 figures + 18 tables
```

Future supplementary figures remain allowed, but they may not overwrite v1 or
alter the training result/statistical definitions.

## Entry 021 — downstream authority chain completed through verified OOS

Date: 2026-08-18  
Status: **Complete; no Stage 5 contract change**

The completed Stage 5 run became the only formal Stage 6 source. The real
selection funnel was 21261 source candidates, 6011 Train-prefilter/Validation
candidates, 2815 six-item hard-filter passes and 1610 retained factors after
Train long-excess decorrelation.

The downstream Baseline authority chain is now complete:

```text
full Baseline Factor Pool = 1610
StrategyInput = frozen-order Top100 prefix
strategies = Equal Weight / Fixed ICIR / LightGBM
OOS evaluation status = complete_verified_oos
OOS rebalance periods = 241
invalid periods = 0
```

Exact fingerprints and artifact paths are recorded in
`BASELINE_DEVELOPMENT_LOG.md`. These downstream results do not revise the Stage 5
design or training contract.

## Entry 022 — Baseline workspace and Legacy boundary cleanup

Date: 2026-08-18  
Status: **Complete with two ACL-protected non-authoritative remnants**

The retained Legacy boundary is deliberately narrow:

- `d521789d86de425794a9e871b42db586`: grammar-hierarchical Primary evidence and
  the only retained Legacy training entry;
- `8778d49870c244a6996e31aa49f40e45`: flat-policy Secondary evidence only;
- `legacy_gflownet_conditional_motivation.ipynb`: dual-run explanation/report.

Old arity/no-anchor/AB/resource-limited/failed Hybrid Notebook entries and run
directories are retired from the current workspace. The current Hybrid Notebook
still reads the N=1/2 Exact-TB registry under
`runs/complexity_diagnostic_6_20/manual_diagnostic_6_20_seed42/`; that directory
is therefore protected in full until a separately validated dependency
migration is completed.

The cleanup removed the approved obsolete Notebook entries, six unrelated
`real_search` runs, three failed Hybrid runs, old no-anchor/diagnostic/A-B run
roots, the legacy Stage 6 archive and the flat-policy checkpoint. Windows ACLs
prevented the Codex process from fully removing two non-authoritative remnants:
`runs/stage6/provisional` and the failed empty Development Matrix directory
`f86a739...`. Neither is referenced by the frozen Baseline authority chain; they
must only be removed manually with appropriate local ownership/administrator
rights.
