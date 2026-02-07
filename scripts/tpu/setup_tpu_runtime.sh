#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="false"
NAME=""
ZONE="us-central2-b"
PROJECT=""
REMOTE_PROJECT_DIR="/tmp/minionerec-jax"
JAX_WHEEL_URL="https://storage.googleapis.com/jax-releases/libtpu_releases.html"
INSTALL_PROJECT_DEPS="true"

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tpu/setup_tpu_runtime.sh --name <tpu-name> [options]

Options:
  --name <value>               TPU VM name. (required)
  --zone <value>               TPU zone (default: us-central2-b)
  --project <value>            GCP project id (optional; otherwise gcloud default)
  --remote-project-dir <path>  Remote project path for editable install (default: /tmp/minionerec-jax)
  --jax-wheel-url <value>      JAX TPU wheel index URL
                                (default: https://storage.googleapis.com/jax-releases/libtpu_releases.html)
  --skip-project-deps          Skip pip install -e <remote-project-dir>
  --dry-run                    Print planned gcloud commands without remote mutation
  -h, --help                   Show this help message

Example:
  bash scripts/tpu/setup_tpu_runtime.sh \
    --name minionerec-jax-v4 \
    --zone us-central2-b \
    --remote-project-dir /tmp/minionerec-jax \
    --dry-run
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
    --remote-project-dir)
      require_flag_value "--remote-project-dir" "${2:-}"
      REMOTE_PROJECT_DIR="$2"
      shift 2
      ;;
    --jax-wheel-url)
      require_flag_value "--jax-wheel-url" "${2:-}"
      JAX_WHEEL_URL="$2"
      shift 2
      ;;
    --skip-project-deps)
      INSTALL_PROJECT_DEPS="false"
      shift
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

if [[ -z "${JAX_WHEEL_URL}" ]]; then
  fail "jax_wheel_url_empty"
fi

kv "status" "start"
kv "script" "setup_tpu_runtime"
kv "name" "${NAME}"
kv "zone" "${ZONE}"
kv "project" "${PROJECT:-gcloud_default}"
kv "remote_project_dir" "${REMOTE_PROJECT_DIR}"
kv "jax_wheel_url" "${JAX_WHEEL_URL}"
kv "install_project_deps" "${INSTALL_PROJECT_DEPS}"
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

LOCAL_TMP_SCRIPT="$(mktemp -t minionerec-tpu-setup-XXXXXX.sh)"
trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat >"${LOCAL_TMP_SCRIPT}" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

REMOTE_PROJECT_DIR="$1"
JAX_WHEEL_URL="$2"
INSTALL_PROJECT_DEPS="$3"

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  kv "remote_status" "error"
  kv "remote_message" "python_not_found"
  exit 1
fi

if "${PYTHON_BIN}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("jax") else 1)' >/dev/null 2>&1; then
  kv "remote_jax_present_before" "true"
else
  kv "remote_jax_present_before" "false"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
kv "remote_pip_upgrade" "ok"

"${PYTHON_BIN}" -m pip install --upgrade "jax[tpu]" -f "${JAX_WHEEL_URL}"
kv "remote_jax_tpu_install" "ok"

if [[ "${INSTALL_PROJECT_DEPS}" == "true" ]]; then
  if [[ -d "${REMOTE_PROJECT_DIR}" ]]; then
    "${PYTHON_BIN}" -m pip install -e "${REMOTE_PROJECT_DIR}"
    kv "remote_project_deps" "installed"
  else
    kv "remote_project_deps" "skipped_project_dir_missing"
  fi
else
  kv "remote_project_deps" "skipped_by_flag"
fi

kv "remote_status" "ok"
REMOTE_SCRIPT

chmod +x "${LOCAL_TMP_SCRIPT}"
REMOTE_SCRIPT_PATH="/tmp/minionerec_setup_tpu_runtime.sh"

SCP_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm scp
  "${LOCAL_TMP_SCRIPT}"
  "${NAME}:${REMOTE_SCRIPT_PATH}"
  --zone "${ZONE}"
)

SSH_SETUP_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm ssh "${NAME}"
  --zone "${ZONE}"
  --command "bash ${REMOTE_SCRIPT_PATH} $(printf '%q' "${REMOTE_PROJECT_DIR}") $(printf '%q' "${JAX_WHEEL_URL}") $(printf '%q' "${INSTALL_PROJECT_DEPS}")"
)

SSH_CLEANUP_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm ssh "${NAME}"
  --zone "${ZONE}"
  --command "rm -f ${REMOTE_SCRIPT_PATH}"
)

kv "runtime_action" "copy_remote_setup_script"
run_cmd "${SCP_CMD[@]}"
kv "runtime_action" "execute_remote_setup"
run_cmd "${SSH_SETUP_CMD[@]}"
kv "runtime_action" "cleanup_remote_setup_script"
run_cmd "${SSH_CLEANUP_CMD[@]}"

kv "status" "ok"
