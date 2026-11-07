#!/usr/bin/env bash
# Install Kronos Lab on a Mac that stays on 24/7 and run it as a background service (launchd).
# The live dashboard then runs locally (http://localhost:8000, or http://<mac-name>.local:8000 on
# your network): one Kronos round per closed candle, live scoring and the baseline benchmark.
#
#   bash scripts/mac/setup.sh            # install or update, start the service
#   bash scripts/mac/setup.sh uninstall  # stop and remove the service
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
LABEL="com.cryptolernex.kronoslab"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"; echo "service removed"; exit 0
fi

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10+ required: brew install python@3.12"; exit 1; }

git pull --ff-only || true
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
[ -f weights/kronos-small/model/config.json ] || .venv/bin/python scripts/download_weights.py kronos-small ./weights
mkdir -p data "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>uvicorn</string>
    <string>app.server:app</string><string>--host</string><string>0.0.0.0</string><string>--port</string><string>8000</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>KRONOS_ENGINE</key><string>kronos</string>
    <key>KRONOS_MODEL</key><string>kronos-small</string>
    <key>KRONOS_WEIGHTS_DIR</key><string>$ROOT/weights/kronos-small</string>
    <key>KRONOS_SAMPLES</key><string>100</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/data/kronoslab.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/kronoslab.log</string>
</dict></plist>
PL

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Kronos Lab running: http://localhost:8000  (log: $ROOT/data/kronoslab.log)"
echo "Tip: keep the Mac awake with: sudo pmset -a sleep 0 disksleep 0"
