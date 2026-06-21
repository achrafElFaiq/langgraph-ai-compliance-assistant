from pathlib import Path
import yaml

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def _load(filename: str) -> str:
    with open(_PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system"]


def load_answer_prompt() -> str:
    return _load("answer.yaml")


def load_synthesis_prompt() -> str:
    return _load("synthesis.yaml")

def load_ground_prompt() -> str:
    return _load("ground.yaml")


def load_apply_prompt() -> str:
    return _load("apply.yaml")