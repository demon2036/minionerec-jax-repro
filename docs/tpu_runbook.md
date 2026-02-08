# TPU Runbook (MiniOneRec-JAX)

This runbook documents the reproducible TPU orchestration flow for this repository,
from GitHub publish through TPU create/setup/run/log collection/teardown.

## Scope and Current Limitations

- This runbook covers end-to-end TPU orchestration and full `eval-official-parity` execution.
- Paper-level alignment is reproducible with the documented **paper profile** (`num_beams=16` on Office).
- Exact values can still drift across runtime versions; pin `jax`, `easydel`, `transformers`, and `huggingface_hub`.
- EasyDeL checkpoint loading uses a **local checkpoint subdirectory path**
  (for example `/tmp/minionerec-assets/Office_ckpt`).

## Prerequisites

1. Local machine has these tools installed and available in `PATH`:
   - `gcloud`
   - `git`
   - `gh`
   - `bash`
2. `gcloud` authenticated and target project selected:
   - `gcloud auth login`
   - `gcloud config set project <YOUR_GCP_PROJECT>`
3. TPU API enabled for the project and zone capacity available (examples use `us-central2-b`).
4. Repository has been published/pushed to GitHub already (see `scripts/publish_to_github.sh`).
5. Run from repo root:

```bash
cd /home/john/workdir3/minionerec-jax
```

## Recommended Execution Order

1. Create TPU VM
2. Setup TPU runtime dependencies
3. Sync code + run remote pipeline
4. Collect remote logs/artifacts
5. Delete TPU VM (cost control)

---

## 1) Create TPU VM

### Dry-run (safe preview)

```bash
bash scripts/tpu/create_tpu_vm.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --accelerator-type v4-8 \
  --version tpu-ubuntu2204-base \
  --preemptible \
  --dry-run
```

### Real run

```bash
bash scripts/tpu/create_tpu_vm.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --accelerator-type v4-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

Notes:
- Script is idempotent-aware; if TPU already exists, it emits `resource_exists=true`
  and exits safely.
- Optional: add `--project <YOUR_GCP_PROJECT>` to pin project explicitly.

---

## 2) Setup TPU Runtime

Installs runtime prerequisites on TPU VM:
- upgrades `pip`
- installs `jax[tpu]`
- optionally installs project deps with `pip install -e <remote-project-dir>`

### Dry-run

```bash
bash scripts/tpu/setup_tpu_runtime.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --remote-project-dir /tmp/minionerec-jax \
  --dry-run
```

### Real run

```bash
bash scripts/tpu/setup_tpu_runtime.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --remote-project-dir /tmp/minionerec-jax
```

Optional flags:
- `--skip-project-deps` to skip editable install.
- `--jax-wheel-url <URL>` to override TPU wheel index source.

---

## 3) Run Remote Pipeline

`scripts/tpu/run_remote_pipeline.sh` supports two code-sync modes:
- `scp` mode (local project copy)
- `git` mode (remote clone/pull from repo URL)

### A. Dry-run with local project copy (`scp`)

```bash
bash scripts/tpu/run_remote_pipeline.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --project-dir /home/john/workdir3/minionerec-jax \
  --dry-run
```

### B. Real run with local project copy (`scp`)

```bash
bash scripts/tpu/run_remote_pipeline.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --project-dir /home/john/workdir3/minionerec-jax \
  --run-id smoke-$(date -u +%Y%m%dT%H%M%SZ)
```

### C. Real run with Git sync (`git`)

```bash
bash scripts/tpu/run_remote_pipeline.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --sync-mode git \
  --repo-url https://github.com/<owner>/minionerec-jax-repro.git \
  --branch main \
  --remote-project-dir /tmp/minionerec-jax
```

The script prints machine-readable lines (for example `status=`, `run=`, `run_result=`,
`remote_log_file=`). Save terminal output if you want a full execution trace.

---


## 3.5) Paper-Profile Full Eval Command (Office)

Run this directly on TPU for the paper-comparison profile:

```bash
cd /tmp/minionerec-jax
PYTHONPATH=src python -m minionerec_jax.cli eval-official-parity   --checkpoint-dir /tmp/minionerec-assets/Office_ckpt   --info-file /tmp/official-minionerec/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt   --test-csv /tmp/official-minionerec/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv   --result-json artifacts/eval/office_full_beam16_batch16.json   --category Office_Products   --batch-size 16   --num-beams 16   --max-new-tokens 256   --length-penalty 0.0   --seed 42   --start-index 0   --model-dtype bfloat16   --param-dtype bfloat16   --sharding-axis-dims 1,1,1,-1,1   --early-stopping never
```

Then compute metrics:

```bash
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics   --predictions-json artifacts/eval/office_full_beam16_batch16.json   --item-info /tmp/official-minionerec/data/Amazon/info/Office_Products_5_2016-10-2018-11
```

## 4) Collect Logs and Results

The remote log file path defaults to:
- `/tmp/minionerec-jax-logs/pipeline_<RUN_ID>.log`

You can collect logs by either:

```bash
# Option 1: view on TPU

gcloud compute tpus tpu-vm ssh minionerec-jax-v4 \
  --zone us-central2-b \
  --command 'tail -n 200 /tmp/minionerec-jax-logs/pipeline_<RUN_ID>.log'
```

```bash
# Option 2: copy log locally

gcloud compute tpus tpu-vm scp \
  minionerec-jax-v4:/tmp/minionerec-jax-logs/pipeline_<RUN_ID>.log \
  ./artifacts/pipeline_<RUN_ID>.log \
  --zone us-central2-b
```

---

## 5) Delete TPU VM (Mandatory Cost Control)

### Dry-run

```bash
bash scripts/tpu/delete_tpu_vm.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --yes \
  --dry-run
```

### Real run

```bash
bash scripts/tpu/delete_tpu_vm.sh \
  --name minionerec-jax-v4 \
  --zone us-central2-b \
  --yes
```

**Always delete TPU resources when finished** to avoid unnecessary charges.

---

## Common Failures and Troubleshooting

### 1) TPU VM not found

Symptoms:
- script emits `message=tpu_vm_not_found_use_create_tpu_vm_first`

Actions:
- run create step first
- verify `--name` and `--zone`
- verify active project (`gcloud config get-value project`)

### 2) Permission or API errors

Symptoms:
- gcloud returns `PERMISSION_DENIED`, API not enabled, or quota errors

Actions:
- ensure correct IAM roles for TPU VM operations
- ensure TPU API is enabled in target project
- retry with explicit `--project <project-id>`

### 3) Preemptible TPU interruption

Symptoms:
- instance disappears or run is interrupted unexpectedly

Actions:
- rerun create/setup/run sequence
- for longer jobs, consider non-preemptible TPU configuration

### 4) Runtime dependency install failure

Symptoms:
- `setup_tpu_runtime.sh` fails during pip install

Actions:
- rerun setup and inspect pip output
- verify network egress/package index access from TPU VM
- optionally rerun with `--skip-project-deps` then install project deps manually

### 5) Remote pipeline fails after sync

Symptoms:
- `run_remote_pipeline.sh` completes sync but remote commands fail

Actions:
- inspect `remote_log_file=...` from script output
- validate `PYTHONPATH=src` command execution from remote project dir
- verify the remote project directory path (`--remote-project-dir`)

## Quick Command Checklist

```bash
bash scripts/tpu/create_tpu_vm.sh --name <name> --zone us-central2-b --accelerator-type v4-8 --version tpu-ubuntu2204-base --preemptible --dry-run
bash scripts/tpu/setup_tpu_runtime.sh --name <name> --zone us-central2-b --remote-project-dir /tmp/minionerec-jax --dry-run
bash scripts/tpu/run_remote_pipeline.sh --name <name> --zone us-central2-b --project-dir /home/john/workdir3/minionerec-jax --dry-run
bash scripts/tpu/delete_tpu_vm.sh --name <name> --zone us-central2-b --yes --dry-run
```
