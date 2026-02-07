#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHONPATH=src python -m minionerec_jax.cli print-config
PYTHONPATH=src python -m minionerec_jax.cli init-manifest --output artifacts/manifest.json
PYTHONPATH=src python -m minionerec_jax.cli smoke
PYTHONPATH=src python -m minionerec_jax.cli probe-constraint-mask
PYTHONPATH=src python -m minionerec_jax.cli download-checkpoint --local-dir artifacts/hf_snapshot --allow-pattern Industrial_ckpt/* --dry-run
PYTHONPATH=src python -m minionerec_jax.cli probe-load --checkpoint-dir artifacts/hf_snapshot/Industrial_ckpt --dry-run
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics --dry-run
