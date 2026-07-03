"""classify_intent node — routes incoming messages to research, followup, chitchat, or synthesis."""
import logging
import joblib
import numpy as np
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from src.application.agent.state import State
from src.config.init_llm import llm

logger = logging.getLogger(__name__)


async def classify_intent(state: State) -> dict:
    """Classify message intent → research | followup | chitchat to route the graph
    (with a shortcut to synthesis for the report trigger)."""
    logger.info("classify_intent | started")

    if state.input_text == "Generate synthesis":
        logger.info("classify_intent | route=synthesis (shortcut)")
        return {"route": "synthesis"}

    logger.debug("classify_intent | history_length=%d", len(state.messages))

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Classifiez l'intention du message en UN seul mot.\n"
            "- research  : nouvelle question de conformité nécessitant une recherche réglementaire\n"
            "- followup  : demande de clarification, reformulation ou précision de la réponse précédente\n"
            "- chitchat  : salutation, remerciement, ou message sans rapport avec la conformité\n"
            "Répondez uniquement par : research, followup, ou chitchat"
        )),
        *state.messages,
        HumanMessage(content=state.input_text)
    ])

    route = response.content.strip().lower()
    if route not in ("research", "followup", "chitchat"):
        route = "research"

    logger.info("classify_intent | route=%s", route)

    reset = {"route": route}
    if route == "research":
        reset.update({
            "retry_count": 0,
            "critic_opinion": "",
            "fallback_attempted": False,
        })

    return reset


# Load model artifacts once at module level (not on every call)
_classifier = joblib.load("models/classifier.joblib")
_vectorizer  = joblib.load("models/vectorizer.joblib")
_mlb         = joblib.load("models/mlb.joblib")
_thresholds  = joblib.load("models/thresholds.joblib")


def classify(state: State) -> dict:
    """Predict which regulations the query concerns via the trained multi-label classifier."""
    logger.info("classify | started")
    time = datetime.now()

    X = _vectorizer.transform([state.input_text])
    probas = _classifier.predict_proba(X)

    regulations = [
        reg
        for i, reg in enumerate(_mlb.classes_)
        if probas[i][0][1] >= _thresholds[i]
    ]

    # Fallback: if nothing passes threshold take the highest scoring one
    if not regulations:
        scores = [probas[i][0][1] for i in range(len(_mlb.classes_))]
        regulations = [_mlb.classes_[int(np.argmax(scores))]]

    duration = (datetime.now() - time).total_seconds()
    logger.info("classify | regulations=%s duration=%.2fs", regulations, duration)
    logger.debug(
        "classify | probas=%s thresholds=%s",
        {reg: round(float(probas[i][0][1]), 3) for i, reg in enumerate(_mlb.classes_)},
        {reg: round(float(_thresholds[i]), 3) for i, reg in enumerate(_mlb.classes_)},
    )
    return {"regulations": regulations}
