import sys
from pathlib import Path

# Project modules (stft, utils, effects) live in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running performance tests (skip with -m 'not slow')"
    )
