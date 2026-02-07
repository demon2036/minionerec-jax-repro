#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="false"
NAME=""
ZONE="us-central2-b"
PROJECT=""
PROJECT_DIR=""
REMOTE_PROJECT_DIR="/tmp/minionerec-jax"
REPO_URL=""
BRANCH="main"
SYNC_MODE="auto"
REMOTE_LOG_DIR="/tmp/minionerec-jax-logs"
RUN_ID=""

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tpu/run_remote_pipeline.sh --name <tpu-name> [options]

Options:
  --name <value>               TPU VM name. (required)
  --zone <value>               TPU zone (default: us-central2-b)
  --project <value>            GCP project id (optional; otherwise gcloud default)
  --project-dir <path>         Local project dir for scp sync (required for sync-mode=scp)
  --remote-project-dir <path>  Remote project dir (default: /tmp/minionerec-jax)
  --repo-url <value>           Git URL for remote clone/update (required for sync-mode=git)
  --branch <value>             Git branch to checkout in git sync mode (default: main)
  --sync-mode <value>          auto|git|scp (default: auto)
  --remote-log-dir <path>      Remote directory for pipeline logs (default: /tmp/minionerec-jax-logs)
  --run-id <value>             Stable run identifier (default: UTC timestamp)
  --dry-run                    Print planned commands without mutating remote state
  -h, --help                   Show this help message

Examples:
  bash scripts/tpu/run_remote_pipeline.sh \
    --name minionerec-jax-dryrun \
    --zone us-central2-b \
    --project-dir /home/john/workdir3/minionerec-jax \
    --dry-run

  bash scripts/tpu/run_remote_pipeline.sh \
    --name minionerec-jax-v4 \
    --zone us-central2-b \
    --sync-mode git \
    --repo-url https://github.com/<owner>/minionerec-jax-repro.git \
    --branch main
USAGE
}

fail() {
  kv "status" "error"
  kv "message" "$1"
  exit 1
}

command_string() {
  local formatted=""
  local token=""
  for token in "$@"; do
    formatted+="$(printf '%q' "${token}") "
  done
  printf '%s' "${formatted% }"
}

run_cmd() {
  local cmd=""
  cmd="$(command_string "$@")"
  kv "run" "${cmd}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    kv "run_result" "dry_run_skip"
    return 0
  fi
  "$@"
  kv "run_result" "ok"
}

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    fail "${cmd}_not_found_${hint}"
  fi
  kv "check_${cmd}_installed" "pass"
}

require_flag_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    fail "missing_value_for_${flag#--}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      require_flag_value "--name" "${2:-}"
      NAME="$2"
      shift 2
      ;;
    --zone)
      require_flag_value "--zone" "${2:-}"
      ZONE="$2"
      shift 2
      ;;
    --project)
      require_flag_value "--project" "${2:-}"
      PROJECT="$2"
      shift 2
      ;;
    --project-dir)
      require_flag_value "--project-dir" "${2:-}"
      PROJECT_DIR="$2"
      shift 2
      ;;
    --remote-project-dir)
      require_flag_value "--remote-project-dir" "${2:-}"
      REMOTE_PROJECT_DIR="$2"
      shift 2
      ;;
    --repo-url)
      require_flag_value "--repo-url" "${2:-}"
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      require_flag_value "--branch" "${2:-}"
      BRANCH="$2"
      shift 2
      ;;
    --sync-mode)
      require_flag_value "--sync-mode" "${2:-}"
      SYNC_MODE="$2"
      shift 2
      ;;
    --remote-log-dir)
      require_flag_value "--remote-log-dir" "${2:-}"
      REMOTE_LOG_DIR="$2"
      shift 2
      ;;
    --run-id)
      require_flag_value "--run-id" "${2:-}"
      RUN_ID="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      fail "unknown_argument_${1}"
      ;;
  esac
done

if [[ -z "${NAME}" ]]; then
  usage
  fail "missing_required_argument_name"
fi

if [[ -z "${ZONE}" ]]; then
  fail "zone_empty"
fi

if [[ -z "${REMOTE_PROJECT_DIR}" ]]; then
  fail "remote_project_dir_empty"
fi

if [[ -z "${REMOTE_LOG_DIR}" ]]; then
  fail "remote_log_dir_empty"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ "${SYNC_MODE}" != "auto" && "${SYNC_MODE}" != "git" && "${SYNC_MODE}" != "scp" ]]; then
  fail "invalid_sync_mode_use_auto_git_or_scp"
fi

if [[ "${SYNC_MODE}" == "auto" ]]; then
  if [[ -n "${REPO_URL}" ]]; then
    SYNC_MODE="git"
  else
    SYNC_MODE="scp"
  fi
fi

if [[ "${SYNC_MODE}" == "git" && -z "${REPO_URL}" ]]; then
  fail "sync_mode_git_requires_repo_url"
fi

if [[ "${SYNC_MODE}" == "scp" ]]; then
  if [[ -z "${PROJECT_DIR}" ]]; then
    fail "sync_mode_scp_requires_project_dir"
  fi
  if [[ ! -d "${PROJECT_DIR}" ]]; then
    fail "project_dir_not_found"
  fi
fi

REMOTE_LOG_FILE="${REMOTE_LOG_DIR}/pipeline_${RUN_ID}.log"

kv "status" "start"
kv "script" "run_remote_pipeline"
kv "name" "${NAME}"
kv "zone" "${ZONE}"
kv "project" "${PROJECT:-gcloud_default}"
kv "sync_mode" "${SYNC_MODE}"
kv "project_dir" "${PROJECT_DIR:-not_set}"
kv "remote_project_dir" "${REMOTE_PROJECT_DIR}"
kv "repo_url" "${REPO_URL:-not_set}"
kv "branch" "${BRANCH}"
kv "run_id" "${RUN_ID}"
kv "remote_log_file" "${REMOTE_LOG_FILE}"
kv "dry_run" "${DRY_RUN}"

require_cmd "gcloud" "install_google_cloud_sdk"

GCLOUD_ARGS=()
if [[ -n "${PROJECT}" ]]; then
  GCLOUD_ARGS+=(--project "${PROJECT}")
fi

DESCRIBE_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm describe "${NAME}"
  --zone "${ZONE}"
  "--format=value(state)"
)

resource_state=""
if resource_state="$("${DESCRIBE_CMD[@]}" 2>/dev/null)" && [[ -n "${resource_state}" ]]; then
  kv "resource_exists" "true"
  kv "resource_state" "${resource_state}"
else
  if [[ "${DRY_RUN}" == "true" ]]; then
    kv "resource_exists" "unknown"
    kv "resource_check" "dry_run_not_enforced"
  else
    fail "tpu_vm_not_found_use_create_tpu_vm_first"
  fi
fi

if [[ "${SYNC_MODE}" == "git" ]]; then
  kv "sync_action" "git_clone_or_update"
  SYNC_GIT_CMD=(
    gcloud
    "${GCLOUD_ARGS[@]}"
    compute tpus tpu-vm ssh "${NAME}"
    --zone "${ZONE}"
    --command "set -euo pipefail; if [ -d $(printf '%q' "${REMOTE_PROJECT_DIR}")/.git ]; then echo sync_mode=git_update; git -C $(printf '%q' "${REMOTE_PROJECT_DIR}") fetch --depth=1 origin $(printf '%q' "${BRANCH}"); git -C $(printf '%q' "${REMOTE_PROJECT_DIR}") checkout -B $(printf '%q' "${BRANCH}") origin/$(printf '%q' "${BRANCH}"); else echo sync_mode=git_clone; rm -rf $(printf '%q' "${REMOTE_PROJECT_DIR}"); git clone --depth=1 --branch $(printf '%q' "${BRANCH}") $(printf '%q' "${REPO_URL}") $(printf '%q' "${REMOTE_PROJECT_DIR}"); fi"
  )
  run_cmd "${SYNC_GIT_CMD[@]}"
else
  kv "sync_action" "scp_local_project"
  LOCAL_PROJECT_DIR_ABS="$(cd "${PROJECT_DIR}" && pwd)"
  LOCAL_PARENT_DIR="$(dirname "${LOCAL_PROJECT_DIR_ABS}")"
  LOCAL_BASENAME="$(basename "${LOCAL_PROJECT_DIR_ABS}")"
  REMOTE_PARENT_DIR="$(dirname "${REMOTE_PROJECT_DIR}")"
  REMOTE_BASENAME="$(basename "${REMOTE_PROJECT_DIR}")"

  SSH_PREP_CMD=(
    gcloud
    "${GCLOUD_ARGS[@]}"
    compute tpus tpu-vm ssh "${NAME}"
    --zone "${ZONE}"
    --command "set -euo pipefail; mkdir -p $(printf '%q' "${REMOTE_PARENT_DIR}"); if [ -d $(printf '%q' "${REMOTE_PROJECT_DIR}") ]; then echo sync_existing=true; rm -rf $(printf '%q' "${REMOTE_PROJECT_DIR}"); else echo sync_existing=false; fi"
  )
  run_cmd "${SSH_PREP_CMD[@]}"

  SCP_CMD=(
    gcloud
    "${GCLOUD_ARGS[@]}"
    compute tpus tpu-vm scp
    --recurse
    "${LOCAL_PROJECT_DIR_ABS}"
    "${NAME}:${REMOTE_PARENT_DIR}"
    --zone "${ZONE}"
  )
  run_cmd "${SCP_CMD[@]}"

  if [[ "${LOCAL_BASENAME}" != "${REMOTE_BASENAME}" ]]; then
    SSH_RENAME_CMD=(
      gcloud
      "${GCLOUD_ARGS[@]}"
      compute tpus tpu-vm ssh "${NAME}"
      --zone "${ZONE}"
      --command "set -euo pipefail; rm -rf $(printf '%q' "${REMOTE_PROJECT_DIR}"); mv $(printf '%q' "${REMOTE_PARENT_DIR}/${LOCAL_BASENAME}") $(printf '%q' "${REMOTE_PROJECT_DIR}")"
    )
    run_cmd "${SSH_RENAME_CMD[@]}"
  fi
fi

LOCAL_TMP_PIPELINE_SCRIPT="$(mktemp -t minionerec-tpu-pipeline-XXXXXX.sh)"
trap 'rm -f "${LOCAL_TMP_PIPELINE_SCRIPT}"' EXIT

cat >"${LOCAL_TMP_PIPELINE_SCRIPT}" <<'REMOTE_PIPELINE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

REMOTE_PROJECT_DIR="$1"
REMOTE_LOG_FILE="$2"

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

run_logged() {
  local cmd=""
  cmd="$1"
  kv "run" "${cmd}"
  bash -lc "${cmd}"
  kv "run_result" "ok"
}

mkdir -p "$(dirname "${REMOTE_LOG_FILE}")"

{
  kv "status" "remote_start"
  kv "remote_project_dir" "${REMOTE_PROJECT_DIR}"
  kv "remote_log_file" "${REMOTE_LOG_FILE}"

  if [[ ! -d "${REMOTE_PROJECT_DIR}" ]]; then
    kv "status" "error"
    kv "message" "remote_project_dir_not_found"
    exit 1
  fi

  cd "${REMOTE_PROJECT_DIR}"

  run_logged "PYTHONPATH=src python -m minionerec_jax.cli print-config"
  run_logged "PYTHONPATH=src python -m minionerec_jax.cli init-manifest --output artifacts/manifest.json"
  run_logged "PYTHONPATH=src python -m minionerec_jax.cli probe-constraint-mask"
  run_logged "PYTHONPATH=src python -m minionerec_jax.cli download-checkpoint --local-dir artifacts/hf_snapshot --allow-pattern 'Industrial_ckpt/*' --dry-run"
  run_logged "PYTHONPATH=src python -m minionerec_jax.cli probe-load --checkpoint-dir artifacts/hf_snapshot/Industrial_ckpt --dry-run"
  run_logged "PYTHONPATH=src python -m minionerec_jax.cli eval-metrics --dry-run"

  kv "status" "remote_ok"
} 2>&1 | tee "${REMOTE_LOG_FILE}"

kv "remote_log_file" "${REMOTE_LOG_FILE}"
REMOTE_PIPELINE_SCRIPT

chmod +x "${LOCAL_TMP_PIPELINE_SCRIPT}"
REMOTE_PIPELINE_SCRIPT_PATH="/tmp/minionerec_run_remote_pipeline.sh"

SCP_PIPELINE_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm scp
  "${LOCAL_TMP_PIPELINE_SCRIPT}"
  "${NAME}:${REMOTE_PIPELINE_SCRIPT_PATH}"
  --zone "${ZONE}"
)
run_cmd "${SCP_PIPELINE_CMD[@]}"

SSH_PIPELINE_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm ssh "${NAME}"
  --zone "${ZONE}"
  --command "bash ${REMOTE_PIPELINE_SCRIPT_PATH} $(printf '%q' "${REMOTE_PROJECT_DIR}") $(printf '%q' "${REMOTE_LOG_FILE}")"
)
run_cmd "${SSH_PIPELINE_CMD[@]}"

SSH_CLEANUP_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm ssh "${NAME}"
  --zone "${ZONE}"
  --command "rm -f ${REMOTE_PIPELINE_SCRIPT_PATH}"
)
run_cmd "${SSH_CLEANUP_CMD[@]}"

kv "status" "ok"
