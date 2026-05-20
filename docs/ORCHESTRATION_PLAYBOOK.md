# Orchestration Playbook

## Purpose
Enable agent-based execution with clear human control points and reproducible checkpoints.

## Roles
1. Operator: approves direction, risk decisions, and release.
2. Engineer-Agent: implements, validates, documents.
3. Auditor-Agent: validates determinism and requirement traceability.

## Standard Execution Sequence
1. Confirm requirement set and constraints.
2. Update `orchestration/TASK_BOARD.md`.
3. Implement code changes.
4. Run required validation commands.
5. Write evidence into `runs/` and update engineering log.
6. Prepare release summary with residual risks.

## Hard Gates
1. No Python runtime handoff in training path.
2. No hidden fallback behavior.
3. No release without deterministic parity check.

## Command Matrix
```powershell
# Build
& "C:\Users\Op-Prime\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\cargo.exe" build --release

# Smoke
& .\target\release\ashira_tokenizer_v2.exe --corpus <dir> --output <dir> --vocab-size 320 --min-freq 2 --accelerator cpu
```

## Handoff Template
Use `orchestration/HANDOFF_TEMPLATE.md`.

