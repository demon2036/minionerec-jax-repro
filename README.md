# MiniOneRec JAX Scaffold

This repository contains the initial scaffold for reproducing MiniOneRec in JAX.
It focuses on reproducible artifact pins, checkpoint conversion probes, and offline
metric parity before generation parity work.

## Scope of This Scaffold

- Project packaging and CLI entrypoints
- Pinned artifact configuration in one dataclass
- Artifact manifest schema plus JSON load/save helpers
- Checkpoint snapshot download helper (`snapshot_download` wrapper)
- EasyDeL `from_torch` load probe from local checkpoint subfolder
- Constrained logits masking component with deterministic parity tests
- MiniOneRec-style offline metric computation (`HR@K` and `NDCG@K`)

## Pinned Artifacts

- Official source commit: `8e03e354033fc81f830580f01c102bd7fbaa262a`
- Hugging Face model sha: `365a03fc6601ed36abd69cc9a0a59025a3d31cdc`
- Default checkpoint subfolder: `Industrial_ckpt`

## Quickstart

```bash
cd /home/john/workdir3/minionerec-jax
bash scripts/bootstrap_env.sh
source .venv/bin/activate
```

## CLI Commands

```bash
PYTHONPATH=src python -m minionerec_jax.cli --help
PYTHONPATH=src python -m minionerec_jax.cli init-manifest --output artifacts/manifest.json
PYTHONPATH=src python -m minionerec_jax.cli print-config
PYTHONPATH=src python -m minionerec_jax.cli smoke
PYTHONPATH=src python -m minionerec_jax.cli probe-constraint-mask
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics --dry-run
```

### Download checkpoint snapshot (subfolder-focused)

```bash
PYTHONPATH=src python -m minionerec_jax.cli download-checkpoint \
  --local-dir artifacts/hf_snapshot \
  --allow-pattern Industrial_ckpt/*
```

Dry-run (no network/download):

```bash
PYTHONPATH=src python -m minionerec_jax.cli download-checkpoint \
  --local-dir artifacts/hf_snapshot \
  --allow-pattern Industrial_ckpt/* \
  --dry-run
```

### Probe EasyDeL torch->JAX load (local subfolder workaround)

Use the local checkpoint subfolder path (for example
`artifacts/hf_snapshot/Industrial_ckpt`) instead of passing an HF subfolder to
EasyDeL directly.

```bash
PYTHONPATH=src python -m minionerec_jax.cli probe-load \
  --checkpoint-dir artifacts/hf_snapshot/Industrial_ckpt
```

Dry-run (path/dependency checks only):

```bash
PYTHONPATH=src python -m minionerec_jax.cli probe-load \
  --checkpoint-dir artifacts/hf_snapshot/Industrial_ckpt \
  --dry-run
```

### Probe constrained logits mask parity

Runs a tiny deterministic example and prints the selected `prefix_index`,
step-wise `count`, allowed-token sets, and EOS fallback diagnostics.

```bash
PYTHONPATH=src python -m minionerec_jax.cli probe-constraint-mask
PYTHONPATH=src python -m minionerec_jax.cli probe-constraint-mask --base-model llama-3 --eos-token-id 9
```

### Compute offline MiniOneRec metrics (`HR@K`, `NDCG@K`)

Expected prediction JSON schema:

```json
[
  {
    "predict": ["candidate-1", "candidate-2", "candidate-3"],
    "output": "candidate-2"
  },
  {
    "predict": ["candidate-1", "candidate-2", "candidate-3"],
    "output": ["candidate-3"]
  }
]
```

`output` may be a string or list; for lists, the first element is used as the
target item. Metrics are computed for top-k values `[1, 3, 5, 10, 20, 50]`
filtered to `k <= n_beam` (`n_beam` is inferred from `len(sample["predict"])`
of the first sample).

Real-file mode:

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics \
  --predictions-json artifacts/eval/predictions.json \
  --item-info artifacts/datasets/Industrial_and_Scientific/new_item_info
```

Dry-run mode (embedded fixture, no external files):

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics --dry-run
```

CLI prints machine-readable lines like `hr@10=...` and `ndcg@10=...`.

### Deterministic unit tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python -m compileall -q src tests && echo COMPILE_OK
```

## Dependency Notes

- `huggingface-hub` is required for real `download-checkpoint` execution.
- `probe-load --dry-run` does not require heavy runtime dependencies.
- Non-dry-run `probe-load` requires `easydel` + `jax`.

## Layout

- `src/minionerec_jax/config.py`: pinned config dataclass
- `src/minionerec_jax/checkpoint.py`: checkpoint download/load wrappers
- `src/minionerec_jax/beam_constraints.py`: constrained logits processor + prefix map helpers
- `src/minionerec_jax/eval_metrics.py`: MiniOneRec-style offline HR/NDCG metric computation
- `src/minionerec_jax/data/artifact_manifest.py`: manifest schema + IO helpers
- `src/minionerec_jax/cli.py`: CLI commands including checkpoint/load probes, constraints, and metrics
- `scripts/bootstrap_env.sh`: local virtualenv bootstrap
- `scripts/run_local_smoke.sh`: local smoke and dry-run probe sequence
- `tests/test_beam_constraints.py`: deterministic constrained-mask parity coverage
- `tests/test_eval_metrics.py`: offline metric parity coverage

## Known Limitations

- A successful `probe-load` confirms checkpoint conversion path only.
- Generation remains unstable due to mesh-context and generation-config compatibility drift.
- Full constrained beam generation loop parity is intentionally deferred; current scope is logits masking parity + offline metric parity only.

## GitHub Publish Automation

Use these scripts to validate GitHub CLI prerequisites and publish the current
project without force-pushing or implicitly pushing tags.

Preflight checks:

```bash
cd /home/john/workdir3/minionerec-jax
bash scripts/check_github_prereqs.sh
```

Dry-run publish preview (recommended first):

```bash
cd /home/john/workdir3/minionerec-jax
bash scripts/publish_to_github.sh \
  --repo minionerec-jax-repro \
  --visibility public \
  --branch main \
  --remote origin \
  --dry-run
```

Real publish (no `--dry-run`):

```bash
cd /home/john/workdir3/minionerec-jax
git add -A
bash scripts/publish_to_github.sh \
  --repo minionerec-jax-repro \
  --visibility public \
  --branch main \
  --remote origin
```

Notes:
- The script creates a repo with `gh repo create` only when the selected remote is missing.
- The script commits only already staged changes (`git add ...` remains explicit).
- Push command is branch-only (`git push --set-upstream <remote> <branch>`), no force and no tags.

## TPU Execution

For full TPU orchestration instructions (publish -> create -> setup -> run -> collect logs -> delete), see:

- `docs/tpu_runbook.md`

### Shortest dry-run path

```bash
cd /home/john/workdir3/minionerec-jax
bash scripts/tpu/create_tpu_vm.sh --name minionerec-jax-dryrun --zone us-central2-b --accelerator-type v4-8 --version tpu-ubuntu2204-base --preemptible --dry-run
bash scripts/tpu/setup_tpu_runtime.sh --name minionerec-jax-dryrun --zone us-central2-b --remote-project-dir /tmp/minionerec-jax --dry-run
bash scripts/tpu/run_remote_pipeline.sh --name minionerec-jax-dryrun --zone us-central2-b --project-dir /home/john/workdir3/minionerec-jax --dry-run
bash scripts/tpu/delete_tpu_vm.sh --name minionerec-jax-dryrun --zone us-central2-b --yes --dry-run
```

### Real-run flow

```bash
cd /home/john/workdir3/minionerec-jax
bash scripts/tpu/create_tpu_vm.sh --name minionerec-jax-v4 --zone us-central2-b --accelerator-type v4-8 --version tpu-ubuntu2204-base --preemptible
bash scripts/tpu/setup_tpu_runtime.sh --name minionerec-jax-v4 --zone us-central2-b --remote-project-dir /tmp/minionerec-jax
bash scripts/tpu/run_remote_pipeline.sh --name minionerec-jax-v4 --zone us-central2-b --project-dir /home/john/workdir3/minionerec-jax
bash scripts/tpu/delete_tpu_vm.sh --name minionerec-jax-v4 --zone us-central2-b --yes
```

Note: generation path is currently unstable; TPU orchestration here targets reproducible smoke checks, not a final paper-level parity claim.
