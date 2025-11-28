import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide dummy env vars for settings during tests
os.environ.setdefault("RUNPOD_API_KEY", "test-key")
os.environ.setdefault("YEHIA_ENDPOINT_ID", "test-endpoint")
