import sys
from pathlib import Path

# Project modules (stft, utils, effects) live in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
