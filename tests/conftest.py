import sys
from pathlib import Path

# Make scripts/hf_configs importable in tests without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
