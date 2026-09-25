"""Delete old uploads and results, so a demo starts with an empty run
list. run.sh / run.bat call this for their --clean flag.

Only the files inside uploads/ and processed/ are removed (the folders
stay). Run from anywhere:
    python scripts/clean_runs.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = (ROOT / "uploads", ROOT / "processed")


def clean(folders=FOLDERS):
    """Empty each folder (creating it if missing). Returns the number of
    entries removed."""
    removed = 0
    for folder in folders:
        folder.mkdir(exist_ok=True)
        for entry in folder.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
    return removed


def main():
    removed = clean()
    names = " and ".join(f"{folder.name}/" for folder in FOLDERS)
    print(f"Cleaned {names}: {removed} old file(s) removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
