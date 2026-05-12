from pathlib import Path
import yaml

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def _load(filename: str) -> str:
    with open(_PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system"]


def load_question_generation_prompt() -> str:
    return _load("question_generation.yaml")


def load_needs_research_prompt() -> str:
    return _load("needs_research.yaml")


def load_answer_prompt() -> str:
    return _load("answer.yaml")


def load_critic_prompt() -> str:
    return _load("critic.yaml")


def load_synthesis_prompt() -> str:
    return _load("synthesis.yaml")

