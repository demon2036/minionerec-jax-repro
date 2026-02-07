#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="false"
YES="false"
NAME=""
ZONE="us-central2-b"
PROJECT=""

kv() {
  printf '%s=%q\n' "$1" "${2:-}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tpu/delete_tpu_vm.sh --name <tpu-name> [options]

Options:
  --name <value>     TPU VM name. (required)
  --zone <value>     TPU zone (default: us-central2-b)
  --project <value>  GCP project id (optional; otherwise gcloud default)
  --yes              Confirm deletion without interactive prompt
  --dry-run          Print planned delete command without deleting resources
  -h, --help         Show this help message

Examples:
  bash scripts/tpu/delete_tpu_vm.sh --name minionerec-jax-v4 --zone us-central2-b --dry-run
  bash scripts/tpu/delete_tpu_vm.sh --name minionerec-jax-v4 --zone us-central2-b --yes
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
    --yes)
      YES="true"
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

kv "status" "start"
kv "script" "delete_tpu_vm"
kv "name" "${NAME}"
kv "zone" "${ZONE}"
kv "project" "${PROJECT:-gcloud_default}"
kv "confirm_yes" "${YES}"
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

DELETE_CMD=(
  gcloud
  "${GCLOUD_ARGS[@]}"
  compute tpus tpu-vm delete "${NAME}"
  --zone "${ZONE}"
  --quiet
)

resource_state=""
if resource_state="$("${DESCRIBE_CMD[@]}" 2>/dev/null)" && [[ -n "${resource_state}" ]]; then
  kv "resource_exists" "true"
  kv "resource_state" "${resource_state}"
else
  if [[ "${DRY_RUN}" == "true" ]]; then
    kv "resource_exists" "unknown"
    kv "resource_action" "dry_run_no_delete"
    kv "status" "ok"
    exit 0
  fi
  kv "resource_exists" "false"
  kv "resource_action" "skip_not_found"
  kv "status" "ok"
  exit 0
fi

if [[ "${YES}" != "true" ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    kv "resource_action" "would_delete_with_yes"
    kv "status" "ok"
    exit 0
  fi
  fail "confirmation_required_pass_yes_to_delete"
fi

kv "resource_action" "delete"
run_cmd "${DELETE_CMD[@]}"
kv "status" "ok"
