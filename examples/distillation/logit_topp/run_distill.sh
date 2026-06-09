#!/bin/bash
# Launch the 3 distillation roles (broker + teacher worker + student client) as
# separate processes. Ctrl+C tears them all down. Extra args go to the client
# (e.g. training.steps=500).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
trap 'kill $(jobs -p) 2>/dev/null; wait' EXIT

# Clean up any stale IPC sockets from previous runs.
rm -f /tmp/softs_distill_*.sock

"$PYTHON" broker.py &
sleep 2
"$PYTHON" supplier.py &
sleep 2

# Client runs in the foreground; forward extra args (e.g. training.steps=500).
"$PYTHON" client.py "$@"
