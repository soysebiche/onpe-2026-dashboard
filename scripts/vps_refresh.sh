#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/onpe-2026-dashboard}"
BRANCH="${BRANCH:-main}"

mkdir -p "$(dirname "$REPO_DIR")"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" git@github.com:soysebiche/onpe-2026-dashboard.git "$REPO_DIR"
fi

cd "$REPO_DIR"

git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null

python onpe_dashboard.py

TRACKED_FILES="dashboard.html dashboard_data.json output/foreign_geo_names.json"

if ! git diff --quiet -- $TRACKED_FILES; then
  git config user.name "npe-vps-bot"
  git config user.email "npe-vps-bot@local"
  git add $TRACKED_FILES
  git commit -m "Auto-update dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git push origin "$BRANCH"
else
  echo "No hubo cambios en los archivos rastreados"
fi
