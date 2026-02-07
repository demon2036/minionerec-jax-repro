#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

kv() {
  printf '%s=%q\n' "$1" "$2"
}

fail() {
  kv "status" "error"
  kv "message" "$1"
  exit 1
}

kv "status" "start"
kv "script" "check_github_prereqs"
kv "root_dir" "${ROOT_DIR}"

if ! command -v git >/dev/null 2>&1; then
  fail "git_not_found_install_git"
fi
kv "check_git_installed" "pass"

if ! command -v gh >/dev/null 2>&1; then
  fail "gh_not_found_install_from_https://cli.github.com/"
fi
kv "check_gh_installed" "pass"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  repo_root="$(git rev-parse --show-toplevel)"
  branch_name="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "DETACHED_OR_UNBORN")"

  if git diff --quiet && git diff --cached --quiet; then
    git_dirty="false"
  else
    git_dirty="true"
  fi

  untracked_count="$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')"

  kv "git_repo_present" "true"
  kv "git_repo_root" "${repo_root}"
  kv "git_branch" "${branch_name}"
  kv "git_dirty" "${git_dirty}"
  kv "git_untracked_count" "${untracked_count}"
else
  kv "git_repo_present" "false"
  kv "git_repo_status" "not_initialized"
fi

auth_output=""
if ! auth_output="$(gh auth status 2>&1)"; then
  printf '%s\n' "${auth_output}" >&2
  fail "gh_auth_status_failed_run_gh_auth_login"
fi
kv "check_gh_auth_status" "pass"

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  kv "gh_auth_detail" "${line}"
done <<< "${auth_output}"

active_account="$(gh api user --jq .login 2>/dev/null || true)"
if [[ -z "${active_account}" ]]; then
  fail "gh_active_account_not_detected"
fi

kv "gh_active_account" "${active_account}"
kv "status" "ok"
