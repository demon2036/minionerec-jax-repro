#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="false"
NAME=""
ZONE="us-central2-b"
ACCELERATOR_TYPE="v4-8"
TPU_VERSION="tpu-ubuntu2204-base"
PREEMPTIBLE="false"
PROJECT=""

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tpu/create_tpu_vm.sh --name <tpu-name> [options]

Options:
  --name <value>              TPU VM name. (required)
  --zone <value>              TPU zone (default: us-central2-b)
  --accelerator-type <value>  TPU accelerator type (default: v4-8)
  --version <value>           TPU runtime version (default: tpu-ubuntu2204-base)
  --project <value>           GCP project id (optional; otherwise gcloud default)
  --preemptible               Create TPU VM as preemptible
  --dry-run                   Print planned gcloud command without creating resources
  -h, --help                  Show this help message

Example:
  bash scripts/tpu/create_tpu_vm.sh \
    --name minionerec-jax-dryrun \
    --zone us-central2-b \
    --accelerator-type v4-8 \
    --version tpu-ubuntu2204-base \
    --preemptible \
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
    --accelerator-type)
      require_flag_value "--accelerator-type" "${2:-}"
      ACCELERATOR_TYPE="$2"
      shift 2
      ;;
    --version)
      require_flag_value "--version" "${2:-}"
      TPU_VERSION="$2"
      shift 2
      ;;
    --project)
      require_flag_value "--project" "${2:-}"
      PROJECT="$2"
      shift 2
      ;;
    --preemptible)
      PREEMPTIBLE="true"
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

if [[ -z "${ACCELERATOR_TYPE}" ]]; then
  fail "accelerator_type_empty"
fi

if [[ -z "${TPU_VERSION}" ]]; then
  fail "version_empty"
fi

kv "status" "start"
kv "script" "create_tpu_vm"
kv "name" "${NAME}"
kv "zone" "${ZONE}"
kv "accelerator_type" "${ACCELERATOR_TYPE}"
kv "version" "${TPU_VERSION}"
kv "preemptible" "${PREEMPTIBLE}"
kv "project" "${PROJECT:-gcloud_default}"
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

CREATE_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm create "${NAME}"
  --zone "${ZONE}"
  --accelerator-type "${ACCELERATOR_TYPE}"
  --version "${TPU_VERSION}"
)
if [[ "${PREEMPTIBLE}" == "true" ]]; then
  CREATE_CMD+=(--preemptible)
fi

existing_state=""
if existing_state="$("${DESCRIBE_CMD[@]}" 2>/dev/null)" && [[ -n "${existing_state}" ]]; then
  kv "resource_exists" "true"
  kv "resource_state" "${existing_state}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    kv "resource_action" "would_skip_already_exists"
    kv "status" "ok"
    exit 0
  fi

  kv "resource_action" "skip_already_exists"
  kv "status" "ok"
  exit 0
fi

kv "resource_exists" "false"
kv "resource_action" "create"
run_cmd "${CREATE_CMD[@]}"
kv "status" "ok"
