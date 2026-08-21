# Daily-Derived v1 Feature Contract

Status: **FROZEN — implementation source of truth**  
Frozen on: 2026-08-18  
Amended on: 2026-08-18 — Step 3C requires every feature-dependent price input to be strictly positive  
Feature space ID: `daily_derived_v1`  
Applies to: the Daily-Derived experiment only

## 1. Authority and scope

This document is the source of truth for the Daily-Derived v1 feature space. The 16 names are reported by the research report, but the exact formulas were not disclosed. Every mathematical definition below is therefore an **engineering reconstruction**, not a report-confirmed formula.

Daily-Derived v1 is a new experiment alongside the frozen Raw Daily Baseline v1:

```text
One shared GFlowNet framework
        |
        +-- Raw Daily Baseline v1: 6 raw leaves
        |
        +-- Daily-Derived v1: 16 engineered daily leaves
```

This contract freezes feature names and order, formulas, raw dependencies, adjustment conventions, PIT/lag behavior, validity rules, dtype/layout, mask semantics, feature-space identity, and experiment isolation.

Implementation must not silently change this contract. If implementation reveals a genuine incompatibility or numerical blocker, stop and request an explicit contract amendment before proceeding. Raw Daily Baseline v1 must not be modified to accommodate this feature space.

## 2. Frozen notation and source fields

```text
O_t    = adjusted open
H_t    = adjusted high
L_t    = adjusted low
C_t    = adjusted close
VWAP_t = adjusted VWAP
V_t    = raw trading volume in shares
A_t    = raw actual trading amount in RMB
S_t    = point-in-time list_a_shares in shares
```

Adjusted price fields use the existing post-adjusted price system:

```text
O/H/L/C    <- market_data with adjust_type=2
adj_factor = close_adjusted / close_raw
VWAP       = amount_raw * adj_factor / volume
```

All price inputs within one formula must use the same adjusted scale. Adjusted OHLC must never be mixed with an unadjusted price.

Raw non-price fields retain their actual economic units:

```text
volume        = raw shares traded; not adjusted
amount        = raw actual RMB trading amount; not adjusted
list_a_shares = shares; not adjusted
```

`amount_raw * adj_factor` is permitted only as the intermediate numerator used to construct adjusted VWAP. It is forbidden as the amount input to `illiq` or `ret_amt_chg5`.

## 3. Frozen feature order and formulas

The following order is immutable.

| Index | Feature | Frozen engineering-reconstruction formula | Warm-up |
|---:|---|---|---|
| 1 | `ret_gap` | `O_t / C_{t-1} - 1` | lag 1 |
| 2 | `ret_cc1` | `C_t / C_{t-1} - 1` | lag 1 |
| 3 | `ret_co` | `C_t / O_t - 1` | none |
| 4 | `ret_hl` | `H_t / L_t - 1` | none |
| 5 | `ret_range` | `(H_t - L_t) / C_{t-1}` | lag 1 |
| 6 | `ret_body` | `abs(C_t - O_t) / C_{t-1}` | lag 1 |
| 7 | `ret_upper_shadow` | `(H_t - max(O_t, C_t)) / C_{t-1}` | lag 1 |
| 8 | `ret_lower_shadow` | `(min(O_t, C_t) - L_t) / C_{t-1}` | lag 1 |
| 9 | `ret_close_vwap` | `C_t / VWAP_t - 1` | none |
| 10 | `ret_open_vwap` | `O_t / VWAP_t - 1` | none |
| 11 | `ret_vol_chg1` | `V_t / V_{t-1} - 1` | lag 1 |
| 12 | `ret_vol_chg5` | `V_t / V_{t-5} - 1` | lag 5 |
| 13 | `turnover` | `V_t / S_t` | none |
| 14 | `illiq` | `1e8 * abs(ret_cc1_t) / A_t` | lag 1 |
| 15 | `ret_amt_chg5` | `A_t / A_{t-5} - 1` | lag 5 |
| 16 | `clv` | `(2*C_t - H_t - L_t) / (H_t - L_t)` | none |

Additional frozen semantics:

- `ret_hl` and `ret_range` remain distinct and must not be merged.
- `ret_body` is a magnitude; direction remains available through `ret_co`.
- Under valid OHLC geometry, `ret_range = ret_body + ret_upper_shadow + ret_lower_shadow`, subject only to floating-point rounding.
- `ret_vol_chg1`, `ret_vol_chg5`, and `ret_amt_chg5` remain simple percentage changes. They are not log-ratios and are not rolling-mean ratios.
- Finite extreme values in the three change features are retained. The builder must not winsorize them.
- `illiq` is a daily leaf. It is not automatically converted to a 20-day or 60-day mean; multi-day forms may be constructed by the shared grammar's existing time-series operators.
- The `1e8` multiplier in `illiq` is a fixed numerical scale constant and is part of the feature contract and fingerprint.
- `turnover` uses shares divided by shares and is not multiplied or divided by 100.

## 4. Frozen PIT and lag contract

Every `t-1` and `t-5` reference is an exact positional offset on the single shared market `date_list`:

```text
t-k = date_index(t) - k
```

The implementation must not:

- search backward for the same stock's most recent non-missing observation;
- move a feature to another date;
- cross a suspension by substituting another date;
- forward-fill or backfill any lag input.

If the input for that stock is invalid on the exact target lag date, the dependent feature is NaN. Warm-up absence never moves the feature date.

The point-in-time share definition is frozen as:

```text
S_t = same-stock latest list_a_shares with change_date <= trade_date t
```

If no qualifying record exists, `turnover_t` is NaN. This is an engineering PIT standard based on the effective date of historical share changes. The current source has no announcement timestamp, database vintage, or revision history and therefore must not be described as strict publication-time PIT. A future publication-time implementation requires a new data contract and experiment version; it must not rewrite Daily-Derived v1.

The feature value at date `t` is an end-of-day value whenever it depends on same-day high, low, close, volume, amount, or VWAP. It is available for the existing evaluation label beginning at `open[t+1]`, not for a trade executed before those `t`-day inputs are known.

## 5. Frozen numerical validity rules

Validity is evaluated independently for each feature and only from that feature's dependencies.

### 5.1 Common rules

- If any required input is NaN, `Inf`, or `-Inf`, the feature is NaN.
- Every adjusted `open`, `high`, `low`, `close`, or `VWAP` actually required by a feature must be strictly greater than zero; otherwise that feature is NaN. This rule applies to price numerators as well as denominators.
- If any ratio denominator is `<= 0`, the feature is NaN.
- No epsilon is added to force a denominator valid.
- Missing values are not replaced with zero.
- Inputs are not forward-filled or backfilled.
- The builder performs no general winsorization or clipping.

These rules imply positive denominators for adjusted prices, VWAP, volume lags, amount, amount lags, and `list_a_shares` wherever they appear.

### 5.2 OHLC geometry

The geometry-based features are:

```text
ret_range
ret_body
ret_upper_shadow
ret_lower_shadow
clv
```

Their same-day OHLC inputs must all be finite and strictly positive, and must satisfy:

```text
H_t >= max(O_t, C_t)
L_t <= min(O_t, C_t)
H_t >= L_t
```

If the relevant geometry is violated, the corresponding geometry-based feature is NaN. The builder must not repair Raw data or silently clamp materially negative shadows.

For `clv`, `H_t - L_t <= 0` produces NaN. General clipping is forbidden. The only permitted boundary correction is:

```text
CLV_TOL = 1e-12

if 1 < abs(clv) <= 1 + CLV_TOL:
    clamp to the corresponding -1 or +1 boundary

if abs(clv) > 1 + CLV_TOL:
    clv = NaN
```

## 6. Frozen tensor and schema contract

```text
feature_space_id = "daily_derived_v1"
schema identity  = "factor_gfn.daily_derived.v1"
layout           = (date, feature, stock)
feature_count    = 16
compute dtype    = float64
artifact dtype   = float32
```

The date axis must be exactly equal, in values and order, to the Raw Daily `date_list`. The stock axis must be exactly equal, in values and order, to the Raw Daily `stock_list`. The feature axis must use the immutable order in Section 3.

The feature-space fingerprint must bind at least the feature-space ID, schema identity, ordered feature names, formulas and constants, raw dependencies and units, adjustment convention, PIT/lag rules, numerical validity rules, tensor layout, and artifact dtype. The action/token fingerprint must independently reflect the 16-leaf vocabulary.

Changing any frozen item requires a new contract/version and must not reuse Daily-Derived v1 artifacts or fingerprints.

## 7. Frozen mask semantics

Daily-Derived v1 does not create a new complete-case stock mask.

```text
Raw universe_mask
    -> determines whether a stock belongs to the evaluation universe

Derived tensor NaN
    -> means only that one feature is unavailable for one date/stock
```

It is valid for one date/stock to have, for example, finite `ret_co` and NaN `ret_vol_chg5`. One missing derived feature must not invalidate all 16 features and must not modify the Raw `universe_mask`.

A feature-validity summary may be saved later as a diagnostic artifact only. It must not become a new research screen or evaluation-universe condition.

## 8. Expression features and evaluation inputs remain separate

The Daily-Derived tensor is used only for:

- GFlowNet expression leaves;
- `FactorInterpreter` expression evaluation.

Future assembly must logically separate:

```text
expression_feature_tensor
```

from the frozen Raw market context used for labels, Raw adjusted open, Barra, universe, PIT industry, shares, benchmark, and all other evaluation inputs.

The forward-return label remains:

```text
forward_return_5d[t]
= raw-baseline adjusted open[t+6]
  / raw-baseline adjusted open[t+1]
  - 1
```

Daily-Derived code must not search for or reconstruct `open` from the 16-feature tensor.

## 9. Experimental isolation and frozen controls

Relative to Raw Daily Baseline v1, the only main experimental variable is the leaf feature space:

```text
6 Raw Daily leaves -> 16 Daily-Derived leaves
```

The following Derived identities and artifacts must be independent:

- expression feature tensor and schema;
- feature vocabulary and token/action-space fingerprint;
- feature-space fingerprint;
- N=1/2 exhaustive registry and exact logZ;
- checkpoints, run identity, Stage 5 outputs, downstream artifacts, and reporting.

Raw registries, checkpoints, runs, and downstream artifacts remain read-only and must never be overwritten or resumed by Daily-Derived v1.

The shared framework and comparison controls remain frozen:

- grammar operator semantics and window set;
- grammar-hierarchical policy and Conditional `N`;
- Hybrid Exact-TB / LPV and Reward;
- candidate cleaning and Barra definitions;
- training allocation/budget, first-version LR, and grad clipping;
- Stage 6 hard filter, sorting, decorrelation, and Top100 rule;
- Equal Weight, Fixed ICIR, Static LightGBM, OOS dates, benchmark, evaluation metrics, forward-return definition, and 5-day rebalance.

If implementation proves that any supposedly frozen shared component must change, stop and request a decision. Do not silently expand the experiment variable.

The research question is fixed as:

> With the other main research contracts held constant, does replacing six Raw Daily leaves with sixteen manually structured Daily-Derived leaves improve GFlowNet Alpha discovery and subsequent OOS performance?

## 10. Rejected or deferred alternatives

- Log volume/amount changes: deferred; v1 uses simple percentage changes.
- Strict publication-time shares PIT: deferred; a future version requires a new source and contract.
- Combined Raw + Derived leaf space: deferred; v1 replaces the six Raw leaves rather than adding to them.
- Builder-level winsorization or rolling illiquidity: rejected for v1.

## 11. Implementation boundary after this freeze

This document freeze does not authorize implementation. In particular, it does not create or modify a builder, production preprocessing, Derived tensor, grammar leaves, token registry, interpreter, model, trainer, Reward, Exact-TB registry, Stage 5, Stage 6, Strategy, or OOS artifact.

The next implementation Step must be separately approved, remain focused, and use synthetic/focused tests before any user-run real data processing.
