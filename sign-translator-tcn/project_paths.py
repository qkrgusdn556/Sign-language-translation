"""Shared project location and local Python module search paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in ('backend', 'training', 'tools'):
    location = str(ROOT / folder)
    if location not in sys.path:
        sys.path.insert(0, location)
