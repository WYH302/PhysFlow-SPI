from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"
THIRD_PARTY = ROOT / "third_party"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
WHEELS = ROOT / "wheels"

for path in (OUTPUTS, DATA, ASSETS, OUTPUTS / "checkpoints"):
    path.mkdir(parents=True, exist_ok=True)
