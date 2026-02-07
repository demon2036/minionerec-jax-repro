#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN="false"
REPO_NAME=""
VISIBILITY="private"
BRANCH_NAME="main"
REMOTE_NAME="origin"

kv() {
  printf '%s=%q\n' "$1" "$2"
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/publish_to_github.sh --repo <name|owner/name> [options]

Options:
  --repo <value>         Repository name (e.g. minionerec-jax-repro or owner/repo). (required)
  --visibility <value>   public|private (default: private)
  --branch <value>       Branch name to publish (default: main)
  --remote <value>       Git remote name (default: origin)
  --dry-run              Print planned actions without changing local or remote state
  -h, --help             Show this help message

Examples:
  bash scripts/publish_to_github.sh --repo minionerec-jax-repro --visibility public --branch main --remote origin --dry-run
  bash scripts/publish_to_github.sh --repo minionerec-jax-repro --visibility public --branch main --remote origin
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
}

require_cmd() {
  local cmd="$1"
  local install_hint="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    fail "${cmd}_not_found_${install_hint}"
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
    --repo)
      require_flag_value "--repo" "${2:-}"
      REPO_NAME="$2"
      shift 2
      ;;
    --visibility)
      require_flag_value "--visibility" "${2:-}"
      VISIBILITY="$2"
      shift 2
      ;;
    --branch)
      require_flag_value "--branch" "${2:-}"
      BRANCH_NAME="$2"
      shift 2
      ;;
    --remote)
      require_flag_value "--remote" "${2:-}"
      REMOTE_NAME="$2"
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

if [[ -z "${REPO_NAME}" ]]; then
  usage
  fail "missing_required_argument_repo"
fi

if [[ "${VISIBILITY}" != "public" && "${VISIBILITY}" != "private" ]]; then
  fail "invalid_visibility_use_public_or_private"
fi

if [[ -z "${BRANCH_NAME}" ]]; then
  fail "branch_name_empty"
fi

if [[ -z "${REMOTE_NAME}" ]]; then
  fail "remote_name_empty"
fi

kv "status" "start"
kv "script" "publish_to_github"
kv "root_dir" "${ROOT_DIR}"
kv "dry_run" "${DRY_RUN}"
kv "repo_input" "${REPO_NAME}"
kv "visibility" "${VISIBILITY}"
kv "branch" "${BRANCH_NAME}"
kv "remote" "${REMOTE_NAME}"
kv "safeguard_force_push" "disabled"
kv "safeguard_push_tags" "disabled"

require_cmd "git" "install_git"
require_cmd "gh" "install_from_https://cli.github.com/"

if ! gh_auth_output="$(gh auth status 2>&1)"; then
  printf '%s\n' "${gh_auth_output}" >&2
  fail "gh_auth_status_failed_run_gh_auth_login"
fi
kv "check_gh_auth_status" "pass"

active_account="$(gh api user --jq .login 2>/dev/null || true)"
if [[ -z "${active_account}" ]]; then
  fail "gh_active_account_not_detected"
fi
kv "gh_active_account" "${active_account}"

if [[ "${REPO_NAME}" == */* ]]; then
  REPO_SLUG="${REPO_NAME}"
else
  REPO_SLUG="${active_account}/${REPO_NAME}"
fi
REMOTE_URL="https://github.com/${REPO_SLUG}.git"

kv "repo_slug" "${REPO_SLUG}"
kv "target_remote_url" "${REMOTE_URL}"

repo_query_available="true"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  kv "git_repo_present" "true"
else
  kv "git_repo_present" "false"
  kv "git_init_action" "initialize_repository"
  run_cmd git init
  if [[ "${DRY_RUN}" == "true" ]]; then
    repo_query_available="false"
    kv "git_repo_query_mode" "simulated_dry_run_without_init"
  fi
fi
kv "git_repo_query_available" "${repo_query_available}"

git_has_commits="false"
current_branch=""
if [[ "${repo_query_available}" == "true" ]]; then
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    git_has_commits="true"
  fi
  current_branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
fi
kv "git_has_commits_before" "${git_has_commits}"
kv "git_current_branch_before" "${current_branch:-DETACHED_OR_UNBORN}"

if [[ "${current_branch}" != "${BRANCH_NAME}" ]]; then
  if [[ "${repo_query_available}" == "true" ]] && git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
    kv "branch_action" "checkout_existing"
    run_cmd git checkout "${BRANCH_NAME}"
  elif [[ "${git_has_commits}" == "true" ]]; then
    kv "branch_action" "create_from_current"
    run_cmd git checkout -b "${BRANCH_NAME}"
  else
    kv "branch_action" "create_unborn_branch"
    run_cmd git checkout -B "${BRANCH_NAME}"
  fi
else
  kv "branch_action" "already_selected"
fi

if [[ "${repo_query_available}" == "true" ]] && existing_remote_url="$(git remote get-url "${REMOTE_NAME}" 2>/dev/null)"; then
  kv "git_remote_present" "true"
  kv "git_remote_current_url" "${existing_remote_url}"
  if [[ "${existing_remote_url}" != "${REMOTE_URL}" ]]; then
    kv "git_remote_action" "set_url"
    run_cmd git remote set-url "${REMOTE_NAME}" "${REMOTE_URL}"
  else
    kv "git_remote_action" "no_change"
  fi
else
  kv "git_remote_present" "false"

  if gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
    kv "gh_repo_exists" "true"
  else
    kv "gh_repo_exists" "false"
    kv "gh_repo_action" "create_repository"
    run_cmd gh repo create "${REPO_SLUG}" "--${VISIBILITY}"
  fi

  kv "git_remote_action" "add_remote"
  run_cmd git remote add "${REMOTE_NAME}" "${REMOTE_URL}"
fi

if [[ "${repo_query_available}" == "true" ]]; then
  if git diff --cached --quiet; then
    kv "staged_changes" "false"
    if ! git diff --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
      kv "hint" "stage_files_with_git_add_before_rerun_for_commit"
    fi
  else
    kv "staged_changes" "true"
    COMMIT_MESSAGE="chore: publish snapshot via automation"
    kv "commit_message" "${COMMIT_MESSAGE}"
    run_cmd git commit -m "${COMMIT_MESSAGE}"
  fi
else
  kv "staged_changes" "unknown_no_repo_context"
  kv "hint" "dry_run_without_repo_skips_index_checks"
fi

if [[ "${repo_query_available}" == "true" ]]; then
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    kv "git_has_commits_after" "true"
    kv "push_action" "push_branch_only"
    run_cmd git push --set-upstream "${REMOTE_NAME}" "${BRANCH_NAME}"
  else
    if [[ "${DRY_RUN}" == "true" ]]; then
      kv "git_has_commits_after" "false"
      kv "push_action" "dry_run_skip_no_commits"
    else
      fail "no_commit_available_stage_and_commit_before_push"
    fi
  fi
else
  kv "git_has_commits_after" "false"
  kv "push_action" "dry_run_skip_no_repo_context"
fi

kv "status" "ok"
