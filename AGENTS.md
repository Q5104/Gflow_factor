# Repository Guidelines

## Project Structure & Sources of Truth

Production code lives in `factor_gfn/`: `data/` handles download and preprocessing, `grammar/` defines the partial-AST DAG, `evaluator/` computes factors and metrics, `barra/` builds style exposures, and `gfn/` contains policy, reward, TB loss, and training code. Tests mirror these modules under `tests/`; manually run workflows live in `notebooks/`. Treat `DEVELOPMENT_SPEC.md` as the decision log, while the actual code and tests are the executable contract.

## Collaboration Rules

For non-trivial changes, first inspect the relevant files and present a short 3–4 step plan with a verifiable result for each step. Wait for approval, then implement only the approved step. If an unresolved choice could alter data semantics, formulas, model state, or public interfaces, pause and ask before continuing. Do not silently freeze research parameters or describe engineering assumptions as paper-confirmed facts.

The user runs long downloads, data-processing notebooks, and real training jobs. Agents may write and statically validate these entry points, but must not launch them unless explicitly asked. Preserve raw data, resumable parts, processed arrays, checkpoints, and unrelated local changes; never delete or regenerate them without explicit approval.

## Development Commands

Use the project interpreter on Windows:

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe -m jupyter lab
```

Run focused tests while developing, then the full suite before handoff. Report exactly what was and was not executed; synthetic tests do not prove real-data readiness.

## Style & Testing

Use four-space indentation, type hints, `snake_case` functions, `PascalCase` classes, and immutable dataclasses where practical. Keep numerical conventions explicit: tensor axes, NaN behavior, `ddof`, window boundaries, and look-ahead constraints require tests. Add regression tests for every changed operator, grammar transition, reward term, or checkpoint contract.

## Git & Review

Keep data, reports, model files, local paths, secrets, and generated outputs out of Git. Prefer small commits such as `feat: add TB loss` or `test: cover parent enumeration`. Before committing, review `git status --short` and `git diff --cached --stat`. A handoff should summarize changed files, tests run, remaining assumptions, and any manual Notebook step.
