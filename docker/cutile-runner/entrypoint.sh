#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_REPO:?RUNNER_REPO is required}"
: "${RUNNER_TOKEN:?RUNNER_TOKEN is required}"
: "${RUNNER_NAME:?RUNNER_NAME is required}"
: "${RUNNER_LABELS:=gpu,cutile-rtx4060}"

./config.sh \
  --unattended \
  --url "$RUNNER_REPO" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work _work \
  --disableupdate

unset RUNNER_TOKEN
exec ./run.sh
