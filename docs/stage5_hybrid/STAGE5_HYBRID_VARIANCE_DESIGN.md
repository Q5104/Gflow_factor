# Stage 5 Hybrid Variance Training Design

Status: **FROZEN — implementation source of truth**  
Frozen on: 2026-08-16  
Applies to: the new Stage 5 `5/15` hybrid-variance training mode only

## 1. Authority and change control

This document is the source of truth for implementing the Stage 5 hybrid-variance training mode.

- Implementation must follow the contracts in this document.
- Core contracts must not be changed implicitly during implementation, refactoring, testing, or debugging.
- If implementation reveals a genuine incompatibility, stop the current Step, record the evidence in `STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md`, and ask the user for a decision.
- A core-contract change requires explicit user approval and a documented amendment to this file before implementation continues.
- Historical `6/20` no-anchor, learned-logZ, A/B, and B1 paths remain legacy/baseline contracts and must not be deleted or silently redefined.

Before every implementation Step:

1. Re-read this document in full.
2. Re-read `STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md` in full.
3. Inspect the current git status and the exact diff of files in that Step.
4. Implement only the requested Step.

After every implementation Step:

1. Run only that Step's focused tests and necessary static checks.
2. Update the development log with changed files, exact tests, results, and remaining issues.
3. Stop and wait for the user's next instruction. Do not automatically start the next Step.

## 2. Frozen training contract

### 2.1 Search and condition space

```text
max_depth = 5
max_nodes = 15
conditions = N = 1, 2, ..., 15
```

All generated expressions must satisfy:

```text
depth <= 5
nodes <= 15
```

No complexity-boundary training diagnostics are required. In particular, do not add:

```text
mean_expression_depth
mean_node_count
fraction_depth_eq_5
fraction_nodes_eq_15
```

Boundary checks remain correctness tests only.

### 2.2 Shared conditional policy

There is exactly one shared conditional GFlowNet forward policy with parameters `theta`.

```text
P_F(action | state, N; theta)
```

All conditions `N=1..15` update the same policy object and parameters. Do not create per-condition models.

The existing condition path must remain active:

```text
target N
-> ExactNodeGrammarState
-> state/action masks
-> StateAdapter condition features
-> shared ForwardPolicyNetwork
```

The current backward probability remains the fixed uniform parent probability. Do not add a trainable PB network.

### 2.3 Single-N batch and K

Each training batch contains one fixed condition and `K` trajectories:

```text
batch = one N + K trajectories whose target_node_count is N
```

`K` is a validated configuration parameter and must not be hard-coded in the loss, Trainer, scheduler, or sampler.

First formal configuration:

```text
K = 16
```

The same K is used for exact-TB conditions `N=1/2` and LPV conditions `N=3..15`. Dynamic K and automatic K adaptation are out of scope.

### 2.4 Condition cycle

Every cycle independently shuffles all conditions using the project's seeded RNG:

```text
conditions = shuffle([1, 2, ..., 15])

for N in conditions:
    collect one K-trajectory single-N batch
    compute the objective for N
    loss.backward()
    clip shared-policy gradients
    policy_optimizer.step()
    scheduler.commit()
```

Therefore, with the first formal K:

```text
1 cycle = 15 single-N batches
1 cycle = 15 policy optimizer steps
1 cycle = 240 successful training trajectories
```

The scheduler uses transactional `peek`/`commit` semantics:

- `peek` returns the pending `(cycle_index, condition_position, N)` without advancing.
- `commit` is called only after a successful policy optimizer step.
- Sampling or training failure does not advance the pending N.
- Every completed cycle gives every N exactly one successful optimizer update.
- A fixed seed must reproduce condition order, including after resume.

### 2.5 Objective routing

#### N = 1 and N = 2

Use the existing fixed exact logZ and existing exact trajectory-balance objective:

```text
fixed exact logZ_N
+ sum_log_pf
- log_reward
- sum_log_pb
```

The exact logZ values:

- remain fixed buffers;
- have no gradient;
- are not optimizer parameters;
- are loaded through the existing exhaustive-registry/exact-mass reuse path;
- may retain the existing small canonical-terminal/structural-hash compatibility verification;
- must not trigger real Reward recomputation or generation of new exact Z assets.

Do not redesign the exact infrastructure unless focused compatibility tests reveal a concrete incompatibility with the new `5/15` config. If one is found, stop and report it before expanding the change.

#### N = 3 through N = 15

Use direct log-partition variance (LPV), with no learned logZ:

```text
zeta_i = log_reward_i + sum_log_pb_i - sum_log_pf_i

zeta_mean = mean(zeta)
centered = zeta - zeta_mean
loss = mean(centered ** 2)
```

Equivalently:

```text
L_LPV = (1 / K) * sum_i (zeta_i - mean(zeta))^2
```

This first implementation intentionally uses the direct Robust Scheduling minibatch empirical variance scale. Do not add the `1 / [2(K-1)]` VarGrad normalization and do not implement two formal LPV variants.

The expected derivative with respect to each zeta element is:

```text
dL/dzeta_i = 2 * (zeta_i - zeta_mean) / K
```

Different variance normalizations can share the same optimum while producing different loss and gradient scales. The scale above is part of the frozen first-version contract.

### 2.6 Autograd and detach contract

- `log_reward` is an environment-provided constant and must be detached from data, IC, Reward parameters, and other external calculations.
- `sum_log_pb` is a constant because the current PB is the fixed uniform parent probability.
- `sum_log_pf` must retain its complete autograd graph.
- Sampled discrete actions and trajectory structure are stop-gradient samples.
- The log-probability of each sampled action under the shared PF retains gradient.
- `zeta_mean = zeta.mean()` participates normally in the expression and is not explicitly detached.
- Do not detach `sum_log_pf`, zeta, or the complete LPV expression.
- Do not implement an explicit leave-one-out loop or additional VarGrad machinery.

### 2.7 No persistent logZ for LPV conditions

For `N=3..15`, hybrid mode must not contain:

```text
trainable logZ parameters
logZ Adam or SGD
logZ learning rate
logZ gradient clipping
EMA logZ
rolling logZ
online normalizer tracker
```

`zeta_mean` is only a diagnostic estimate of the current batch's implied logZ. It is not a learned, selected, rolling, EMA, or checkpointed normalizer for future batches.

Legacy learned-logZ TB code remains available in the legacy/baseline mode for historical reproduction and A/B comparison.

### 2.8 Policy optimizer

Hybrid mode uses only the shared policy parameters:

```text
optimizer = Adam
policy learning rate = 1e-4
policy gradient clip = 5
```

Do not change optimizer type, learning rate, clipping, policy architecture, hidden sizes, sampling temperature, exploration, Reward, operators, PF/PB semantics, data, or labels as part of this work.

## 3. Frozen diagnostics and counting contract

### 3.1 Common per-update fields

At minimum record:

```text
cycle_index
condition_position_in_cycle
condition_N
objective_kind
global_optimizer_step
trajectories_in_batch
total_trajectories_seen
requested / accepted / invalid / retry counts
reward_mean / reward_std
sum_log_pf_mean
sum_log_pb_mean
trajectory_length
terminal_success_rate
policy_grad_norm
```

`total_trajectories_seen` counts trajectories used by successful optimizer updates. With K=16, it increases by 16 per successful batch and by 240 per completed cycle. Rejected or retry attempts remain separate diagnostics.

### 3.2 Exact-TB fields for N=1/2

```text
objective_kind = exact_tb
exact_log_z
tb_loss
tb_delta_mean
tb_delta_std
tb_delta_rms
```

### 3.3 LPV fields for N=3..15

```text
objective_kind = log_partition_variance
zeta_mean
zeta_std
zeta_variance
variance_loss
centered_zeta_rms
unique_terminal_count
unique_terminal_fraction
```

Terminal diversity must reuse the project's stable canonical structural hash or terminal ID:

```text
unique_terminal_fraction = unique terminal identities / K
```

It is diagnostic only. Do not add it to the loss, Reward, penalty, scheduler, or exploration mechanism.

For `N>=3`, do not output or synthesize:

```text
selected_logZ
learned_logZ
TB delta mean/std/RMS
```

Do not rename `zeta_mean` as a legacy logZ field and do not construct a fake zero-mean TB delta from it.

## 4. Checkpoint and resume contract

The hybrid mode uses a distinct checkpoint schema and must support deterministic mid-cycle resume.

The checkpoint contains at least:

```text
shared policy state
policy-only optimizer state and contract
fixed exact N=1/2 buffers and reuse proof/manifest
objective mode
hybrid config fingerprint

cycle index
current cycle permutation
condition position
condition RNG state

global optimizer step
total trajectories seen
history and diagnostic counters

Python RNG
NumPy RNG
Torch CPU RNG
CUDA RNG
```

Successful update order is fixed:

```text
optimizer.step()
-> scheduler.commit()
-> update counters
-> atomic checkpoint save
```

Hybrid and legacy checkpoints must reject cross-loading. The hybrid checkpoint must not contain learned-logZ optimizer state, EMA/rolling logZ, or a persistent LPV normalizer.

## 5. Out of scope

This implementation must not:

- change Reward or Reward parameters;
- change data, labels, IC calculation, or factor evaluation;
- change operators or grammar semantics;
- change policy architecture, hidden size, embedding size, temperature, or exploration;
- add a trainable PB network;
- add dynamic K;
- add complexity-boundary diagnostics or penalties;
- redesign exhaustive/exact infrastructure preemptively;
- delete legacy learned-logZ/TB code;
- modify Stage 6 contracts;
- start real Reward evaluation or real training without separate explicit approval.

## 6. Frozen implementation Steps

### Step 1 — Hybrid configuration

Add an isolated `5/15` hybrid config with configurable K and first-version K=16. Preserve legacy fingerprints. Focused tests cover config validation, K configurability, legacy stability, and hard 5/15 correctness boundaries.

### Step 2 — Direct LPV objective

Add a standalone direct-LPV implementation using `centered.square().mean()`. Focused tests cover values, the analytic `2 * centered / K` gradient, detach behavior, PF gradient, and K<2 failure.

### Step 3 — Transactional condition scheduler

Extend `BalancedNodeCountScheduler` with `peek`/`commit` while preserving legacy APIs. Focused tests cover exact cycle coverage, shuffle reproducibility, failure without advance, and state round-trip.

### Step 4 — Single-N batch collection

Add a Trainer helper that requests `(N,) * K` through the existing sampler. Focused tests cover fixed N, configurable K, retry behavior, and no optimizer/scheduler advance on incomplete batches.

### Step 5 — Existing exact-Z compatibility

Verify the existing exact-registry loading and reuse path under the hybrid `5/15` config. Make only the narrowest adapter if an actual type/manifest incompatibility is proven. Do not redesign exact infrastructure.

### Step 6 — Hybrid routing and policy-only optimizer

Route `N=1/2` to existing exact TB and `N=3..15` to direct LPV. Hybrid mode contains one shared policy optimizer and no learned-logZ parameter or optimizer. Preserve the complete legacy path.

### Step 7 — Diagnostics and counters

Add objective-specific diagnostics, explicit cycle/optimizer/trajectory units, and canonical terminal diversity. Do not add complexity-boundary diagnostics or fake TB fields.

### Step 8 — Hybrid checkpoint and runner

Add a distinct hybrid checkpoint schema and runner with deterministic mid-cycle resume and the fixed commit/save ordering.

### Step 9 — Focused regression and synthetic smoke

Run the accumulated focused tests and a synthetic-only end-to-end smoke. Do not run real Reward, real training, or the full regression suite without separate approval.

## 7. Step dependency order

```text
Step 1 -> Steps 3, 4, 5
Step 2 -> Step 6
Steps 3, 4, 5 -> Step 6
Step 6 -> Step 7
Steps 3, 6, 7 -> Step 8
Step 8 -> Step 9
```

Steps 2 and 3 can be prepared independently. Steps 4 and 5 can be prepared independently after Step 1. Regardless of dependency, no Step starts automatically: each Step requires a new user instruction.

## 8. Completion gate

The core training design is frozen. Implementation may begin only with an explicit instruction to start Step 1.

A Step is complete only when:

1. its approved scope is implemented;
2. its focused tests pass;
3. the exact tests and results are recorded in the development log;
4. changed files and any deviations are recorded;
5. work stops for user review.

## 9. Phase B — Stage 5 Train artifact and Stage 6 reuse

Status: **FROZEN — 2026-08-16**

Phase B only makes Train evaluation results that Stage 5 already computed
durable and reusable by Stage 6. It does not change training or selection
mathematics.

### 9.1 Frozen boundaries

```text
Stage 5 already-computed Train metrics should be reused when their contract matches.
Validation never enters Reward, LPV, exact TB, policy update, or candidate generation.
candidate identity = existing canonical structural_hash; no second candidate ID.
candidate universe freezes only after training completes.
accepted candidates -> Train prefilter -> frozen train-pass manifest
                    -> Validation only for train-pass candidates.
complete new Hybrid artifact -> no repeated Train FactorInterpreter in Stage 6.
legacy or incomplete artifact -> explicit fresh fallback is allowed.
missing N=1/2 train_long_excess -> never fabricate; only a survivor may use enrichment fallback.
```

Phase B must not refactor `HybridVarianceTrainer`, LPV, exact TB, policy,
optimizer, Reward formula, Stage 6 thresholds/selection, or Validation
definitions. It must not add a database or a general cache framework.

### 9.2 Frozen implementation order

```text
B1 minimal Train artifact
B2 candidate freeze gate — DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE
B3 Stage 6 verified overlay integration — DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE
B4 Validation and equivalence regression — DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE
```

Each step is separately approved, tested, logged, and stopped. A later step
must not start automatically.

### 9.3 B1 minimal artifact contract

B1 persists only Train information already produced by Stage 5 evaluation:

```text
structural_hash
train_evaluation_contract_fingerprint
train_ic
train_ic_valid_periods
train_direction
train_long_ir
train_long_valid_periods
train_long_excess_dates
train_long_excess_values
train_barra_ts_corr
train_barra_correlations
train_barra_valid_periods_by_style
neutralization_diagnostics
```

The artifact also binds its stable schema/contract version, source run,
Reward Provider, Train data/context, and implementation fingerprints. Existing
provenance must be referenced rather than duplicated when possible.

The artifact must not contain a full factor matrix, raw full-panel factor
values, Validation metrics, Reward ranking, or Stage 6-unused Reward-derived
fields. `valid`, Reward, and Reward floor remain training/audit values and must
not become Stage 6 selection substitutes.

Repeated visits align by `structural_hash`; one Train evaluation record is
kept per identity, with optional first/last-seen provenance. Persistence belongs
to the Hybrid runner/output boundary, is atomic, recoverable, and idempotent,
and fails closed on checkpoint/artifact divergence. Candidate-universe freeze
is B2 and is not part of B1.

### 9.4 B1 implementation freeze

Status: **B1 IMPLEMENTATION FROZEN — 2026-08-16**

The verified artifact has these top-level fields:

```text
schema
source_run
train_evaluation_contract
train_evaluation_contract_fingerprint
committed_optimizer_step
candidate_count
records
```

Each candidate record has these fields:

```text
schema
structural_hash
formula
prefix_token_ids
node_count
depth
train_evaluation_contract_fingerprint
train_ic
train_ic_valid_periods
train_direction
train_long_ir
train_long_valid_periods
train_long_excess_dates
train_long_excess_values
train_barra_ts_corr
train_barra_correlations
train_barra_valid_periods_by_style
neutralization_diagnostics
first_seen
last_seen
visit_count
```

`train_long_excess_dates` and `train_long_excess_values` are serialized from
the same long-excess result already produced inside Reward evaluation. Artifact
persistence reads the recorded `RewardResult`; it does not invoke
`FactorInterpreter` or recompute factor values, portfolio returns, IC, IR, or
Barra metrics.

No Stage 6 module loads or imports this artifact in B1. Candidate freeze and
Stage 6 integration remain B2/B3 work and require separate authorization.

### 9.5 Deferred Phase B continuation

Status: **DEFERRED UNTIL STAGE 5 REAL TRAINING IS COMPLETE**

B2, B3, and B4 remain part of the approved Phase B design, but their
implementation is intentionally postponed until the formal Stage 5 run has
completed and the project is ready to enter Stage 6. This deferral does not
delete or redefine their contracts.

The immediate next work package is a thin real-training Notebook that only
orchestrates the frozen production config, preflight, diagnostics,
checkpoint/resume, and Train-artifact inspection. It must not reimplement any
training objective, scheduler, optimizer, Reward, persistence, or checkpoint
logic.
