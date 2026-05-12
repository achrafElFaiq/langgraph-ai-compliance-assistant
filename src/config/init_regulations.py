from pathlib import Path
import yaml

_REGULATIONS_FILE = Path(__file__).resolve().parents[2] / "configs" / "regulations.yaml"


def load_regulations() -> dict:
    """Load regulation metadata from configs/regulations.yaml."""
    with open(_REGULATIONS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["regulations"]


REGULATIONS = load_regulations()

