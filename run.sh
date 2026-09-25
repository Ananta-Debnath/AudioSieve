#!/usr/bin/env bash
# Start SPECTRA on http://127.0.0.1:5000 (macOS / Linux / Git Bash).
#
#   ./run.sh           set up .venv if needed, then run the app
#   ./run.sh --clean   also delete old uploads and results first
#
# The first run needs the internet (pip installs requirements.txt into
# .venv). Later runs skip pip while requirements.txt is unchanged, so
# they work offline.
set -e
cd "$(dirname "$0")"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (use --clean or --help)"; exit 2 ;;
  esac
done

# A Python 3.12+ that really runs (on Windows, "python3" can be a Store stub).
PYTHON=""
for candidate in python3 python; do
  if "$candidate" -c "import sys; sys.exit(sys.version_info < (3, 12))" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ] && [ ! -d .venv ]; then
  echo "SPECTRA needs Python 3.12 or newer on the PATH (python3 or python)."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating the virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
fi

# Activate it (bin/ on macOS and Linux, Scripts/ for a Windows Python).
if [ -f .venv/bin/activate ]; then
  . .venv/bin/activate
else
  . .venv/Scripts/activate
fi

STAMP=.venv/requirements.installed
if ! cmp -s requirements.txt "$STAMP"; then
  echo "Installing requirements ..."
  python -m pip install --disable-pip-version-check -r requirements.txt
  cp requirements.txt "$STAMP"
fi

if [ "$CLEAN" = 1 ]; then
  python scripts/clean_runs.py
fi

echo
echo "  SPECTRA: open http://127.0.0.1:5000"
echo
exec python app.py
