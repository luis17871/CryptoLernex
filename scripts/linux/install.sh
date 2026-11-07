#!/usr/bin/env bash
# Install or update Kronos Lab with Docker on a Linux box (e.g. the one running Frigate).
#   curl -fsSL https://raw.githubusercontent.com/luis17871/CryptoLernex/claude/lucid-rubin-ltu6sf/scripts/linux/install.sh | bash
# or, from a clone:  bash scripts/linux/install.sh
set -euo pipefail
DIR="${KRONOS_DIR:-$HOME/kronos-lab}"
BRANCH="claude/lucid-rubin-ltu6sf"
command -v docker >/dev/null || { echo "Docker not found. Install it first: https://docs.docker.com/engine/install/"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose plugin not found (docker compose)."; exit 1; }
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone -b "$BRANCH" https://github.com/luis17871/CryptoLernex.git "$DIR"
fi
cd "$DIR"
docker compose -f deploy/docker-compose.yml up -d --build
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Kronos Lab is starting. Dashboard: http://${IP:-localhost}:8010"
echo "First start downloads the model (~110 MB). Logs: docker logs -f kronos-lab"
