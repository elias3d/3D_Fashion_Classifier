#!/usr/bin/env bash
# ============================================================
#  Fashion3D - run on Mac / Linux:   bash run_mac_linux.sh
#  First run: creates .venv and installs packages (2-5 min).
#  Then opens http://localhost:8517 in your browser. Ctrl+C to stop.
# ============================================================
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/installed.ok ]; then
  PY=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10, 3.11 or 3.12 is needed: https://www.python.org/downloads/"
    exit 1
  fi
  # NVIDIA GPU on Linux -> CUDA PyTorch; Mac or no GPU -> CPU PyTorch. Override: FASHION3D_DEVICE=cpu|gpu
  REQ=requirements.txt; KIND=CPU
  if [ "${FASHION3D_DEVICE:-}" = "gpu" ] || { [ "${FASHION3D_DEVICE:-}" != "cpu" ] && [ "$(uname)" = "Linux" ] \
       && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
    REQ=requirements-gpu.txt; KIND="GPU (NVIDIA CUDA)"
  fi
  echo "Creating the Python environment with $PY ..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  echo "Installing packages for $KIND (first run only, this can take several minutes) ..."
  if ! .venv/bin/python -m pip install -r "$REQ"; then
    [ "$REQ" = requirements.txt ] && exit 1
    echo "The GPU install failed - falling back to the CPU version ..."
    .venv/bin/python -m pip install -r requirements.txt
    KIND=CPU
  fi
  echo "$KIND" > .venv/installed.ok
fi

echo
echo " Fashion3D is starting on http://localhost:8517  (Ctrl+C to stop)"
echo
( sleep 6; (command -v open >/dev/null && open http://localhost:8517) || \
           (command -v xdg-open >/dev/null && xdg-open http://localhost:8517) ) >/dev/null 2>&1 &
.venv/bin/python -m streamlit run app.py --server.port 8517 --server.headless true
