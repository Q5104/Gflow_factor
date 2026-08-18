# Repository Guidelines

## Project Structure & Sources of Truth

Production code lives in `factor_gfn/`: `data/` handles download and preprocessing, `grammar/` defines the partial-AST DAG, `evaluator/` computes factors and metrics, `barra/` builds style exposures, and `gfn/` contains policy, reward, TB loss, and training code. Tests mirror these modules under `tests/`; manually run workflows live in `notebooks/`. Treat `DEVELOPMENT_SPEC.md` as the decision log, while the actual code and tests are the executable contract.

## Collaboration Rules

Classify each task using the development workflow below before acting. For core-architecture or otherwise substantial changes, first inspect the relevant files and present a short 3–4 step plan with a verifiable result for each step. Wait for approval, then implement only the approved step. If an unresolved choice could alter data semantics, formulas, model state, or public interfaces, pause and ask before continuing. Do not silently freeze research parameters or describe engineering assumptions as paper-confirmed facts.

The user runs long downloads, data-processing notebooks, and real training jobs. Agents may write and statically validate these entry points, but must not launch them unless explicitly asked. Preserve raw data, resumable parts, processed arrays, checkpoints, and unrelated local changes; never delete or regenerate them without explicit approval.

## Development Workflow

### Classify the Task Before Acting

Use the lightest of these three execution modes that preserves correctness. When uncertain between two levels, start with the lighter level and escalate only if the actual code dependency requires it.

#### Lightweight Patch

Examples include notebooks, display text, documentation, small configuration changes, and local UI or control-flow edits.

- Make the smallest local change.
- Do not re-audit the whole repository or redesign architecture.
- Run only a focused existing test, syntax check, or similarly small validation unless broader testing is explicitly requested.
- Do not update development logs unless the change is materially important.
- If the task unexpectedly requires substantially more files or broad architectural changes, stop and explain before expanding scope.

#### Medium Feature

Examples include a small module, artifact, persistence layer, or isolated interface.

- Inspect only the relevant code.
- Implement the minimum feature.
- Run focused tests and adjacent regression tests.
- Avoid unrelated refactors.

#### Core Architecture

Examples include trainer behavior, loss or Reward semantics, checkpoint contracts, schedulers, and model architecture.

- Do not modify code immediately.
- First inspect the current implementation and produce a stepwise plan that freezes what may and may not change.
- For a substantial independent redesign, create a dedicated `*_DESIGN.md` and `*_DEVELOPMENT_LOG.md`.
- Implement step by step with focused tests and reviewable diffs, stopping at the approved boundaries.

### Prevent Scope Expansion

Never silently broaden a task. If a small task appears to require substantially more files, changed semantics, or architectural work than expected, stop and report why before proceeding.

### Documentation Separation

- Put project-wide stable rules in the project specification.
- Give an independent major redesign its own `*_DESIGN.md`.
- Record that redesign's implementation history in its own `*_DEVELOPMENT_LOG.md`.
- Minor patches normally need no dedicated log.
- Treat the current dedicated design file as the source of truth for its redesign; a historical design must not override a newer frozen contract.

### Testing Proportional to Risk

- Lightweight patch: focused existing test, syntax check, or equivalent small validation.
- Medium feature: focused tests plus adjacent regression tests.
- Core architecture: focused tests, then synthetic smoke, broader regression when appropriate, and a small real-world run only when explicitly authorized.
- Do not run expensive tests automatically for a low-risk patch.

### Experimental Integrity

- Resume only when the frozen configuration is unchanged.
- For material configuration changes, preserve the old run and create a new run.
- Do not bypass checkpoint or configuration fingerprints for convenience.
- Never silently change unrelated experimental variables.

## Development Commands

Use the project interpreter on Windows:

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe -m jupyter lab
```

Run focused tests while developing. Reserve the full suite for important contract changes, milestone handoffs, or an explicit user request. Report exactly what was and was not executed; synthetic tests do not prove real-data readiness.

## Execution Efficiency

- For a local change, default to focused tests and the smallest necessary static checks.
- When a traceback already identifies the failing path, diagnose that path locally instead of reopening the broader design.
- Do not repeatedly scan the repository or reread unchanged files already inspected in the current work package.
- Treat frozen contracts as fixed; confirm only the proposed difference unless the user explicitly requests a broader review.
- Run the full regression suite only for important contract changes, milestone handoffs, or an explicit request.
- Use lower reasoning effort for simple, well-scoped edits and diagnostics.
- Recommend a new thread for each independent work package, with a short handoff of current state and boundaries.

## Style & Testing

Use four-space indentation, type hints, `snake_case` functions, `PascalCase` classes, and immutable dataclasses where practical. Keep numerical conventions explicit: tensor axes, NaN behavior, `ddof`, window boundaries, and look-ahead constraints require tests. Add regression tests for every changed operator, grammar transition, reward term, or checkpoint contract.

## Git & Review

Keep data, reports, model files, local paths, secrets, and generated outputs out of Git. Prefer small commits such as `feat: add TB loss` or `test: cover parent enumeration`. Before committing, review `git status --short` and `git diff --cached --stat`. A handoff should summarize changed files, tests run, remaining assumptions, and any manual Notebook step.
