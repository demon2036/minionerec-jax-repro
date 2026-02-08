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
PYTHONPATH=src python -m minionerec_jax.cli eval-official-parity \
  --dry-run \
  --checkpoint-dir /tmp/ckpt \
  --info-file /tmp/info.txt \
  --test-csv /tmp/test.csv \
  --result-json /tmp/out.json \
  --category Industrial_and_Scientific
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

### Full parity eval (`official evaluate.py` behavior)

This command runs the full constrained-beam generation path against official
checkpoint/info/test inputs and writes a JSON that can be fed directly into
`eval-metrics`.

Dry-run mode (prints config/runtime/path diagnostics, no heavy model load):

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-official-parity \
  --dry-run \
  --checkpoint-dir /tmp/ckpt \
  --info-file /tmp/info.txt \
  --test-csv /tmp/test.csv \
  --result-json /tmp/out.json \
  --category Industrial_and_Scientific
```

Real run example:

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-official-parity \
  --checkpoint-dir artifacts/hf_snapshot/Industrial_ckpt \
  --info-file ../official-minionerec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --test-csv ../official-minionerec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --result-json artifacts/eval/final_result_Industrial_and_Scientific.json \
  --category Industrial_and_Scientific \
  --batch-size 4 \
  --num-beams 50 \
  --max-new-tokens 256 \
  --length-penalty 0.0 \
  --seed 42
```

Then compute metrics using the generated JSON:

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics \
  --predictions-json artifacts/eval/final_result_Industrial_and_Scientific.json \
  --item-info ../official-minionerec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11
```

Notes:
- Constrained decoding uses official-style SID prefix dictionary + EOS fallback.
- Tokenizer left-padding and decode post-processing follows `split("Response:\n")[-1].strip()`.
- Default sharding axis dims are `1,1,1,-1,1` and can be overridden by `--sharding-axis-dims`.

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
- `src/minionerec_jax/official_eval_parity.py`: official evaluate.py parity generation pipeline (EasyDeL/JAX)
- `src/minionerec_jax/data/artifact_manifest.py`: manifest schema + IO helpers
- `src/minionerec_jax/cli.py`: CLI commands including checkpoint/load probes, constraints, metrics, and parity eval
- `scripts/bootstrap_env.sh`: local virtualenv bootstrap
- `scripts/run_local_smoke.sh`: local smoke and dry-run probe sequence
- `tests/test_beam_constraints.py`: deterministic constrained-mask parity coverage
- `tests/test_eval_metrics.py`: offline metric parity coverage
- `tests/test_official_eval_parity.py`: offline tests for prompt/prefix/postprocess parity helpers

## Known Limitations

- `probe-load` confirms checkpoint conversion path, but does not guarantee final ranking quality.
- End-to-end parity still depends on pinned runtime versions (`jax`, `easydel`, `transformers`, `huggingface_hub`) and TPU topology.
- `official-minionerec/evaluate.py` defaults to `num_beams=50`; for paper-level alignment, run the paper profile (`num_beams=16`) shown below.


## Paper Alignment (TPU Evidence)

The following **full Office** run aligns with the paper target when using beam width 16:

- Artifact: `artifacts/eval/office_full_beam16_batch16.json`
- Sample count: `4866`
- Metrics (`eval-metrics`):
  - `hr@10=0.164611590629` (paper `0.1634`, Δ `+0.001211590629`)
  - `ndcg@10=0.127101142105` (paper `0.1242`, Δ `+0.002901142105`)
- Full per-metric comparison table: `docs/paper_alignment.md`

Reproduce on TPU (from clean GitHub `main`):

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-official-parity   --checkpoint-dir /tmp/minionerec-assets/Office_ckpt   --info-file /tmp/official-minionerec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt   --test-csv /tmp/official-minionerec/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv   --result-json artifacts/eval/office_full_beam16_batch16.json   --category Office_Products   --batch-size 16   --num-beams 16   --max-new-tokens 256   --length-penalty 0.0   --seed 42   --start-index 0   --model-dtype bfloat16   --param-dtype bfloat16   --sharding-axis-dims 1,1,1,-1,1   --early-stopping never
```

Then compute metrics:

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics   --predictions-json artifacts/eval/office_full_beam16_batch16.json   --item-info ../official-minionerec/data/Amazon/info/Office_Products_5_2016-10-2018-11
```

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

Note: for paper-level comparison, prefer the full `eval-official-parity` + `eval-metrics` flow above with the documented paper profile.
