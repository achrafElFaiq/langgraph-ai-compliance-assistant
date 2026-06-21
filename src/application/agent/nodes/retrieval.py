import joblib
import numpy as np
import json
from datetime import datetime
from src.application.agent.state import State
from src.config.init_embedder import embedder
from src.config.init_store import store


# Load model artifacts once at module level (not on every call)
_classifier = joblib.load("datasets/classifier/model/classifier.joblib")
_vectorizer  = joblib.load("datasets/classifier/model/vectorizer.joblib")
_mlb         = joblib.load("datasets/classifier/model/mlb.joblib")
_thresholds  = joblib.load("datasets/classifier/model/thresholds.joblib")



# Node Classify: Classify the input text into relevant regulations using a pre-trained classifier.
def classify(state: State) -> dict:
    print("[Started Node] classify")
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

    print(f"[Finished Node] classify In: {(datetime.now() - time).total_seconds()} seconds")
    return {"regulations": regulations}


# Node: Performs retrieval on the classified label
# What happens in case the classifier doesn't do well ? Not treated
async def retrieve_articles(state: State) -> dict:
    print("[Started Node] retrieve articles")
    time = datetime.now()

    query = state.input_text
    top_k = 5 * len(state.regulations)

    embedding = await embedder.embed_query(query)
    articles = await store.retrieve(
        embedding=embedding,
        query=query,
        top_k=top_k,
        regulations=state.regulations
    )

    skeleton = {
        a.breadcrumb: {"relevant": None, "excerpts": []}
        for a in articles
    }

    print(f"[Finished Node] retrieve Ended In: {(datetime.now() - time).total_seconds()} seconds, retrieved={len(articles)}")
    return {
        "retrieved_articles": articles,
        "grounded_skeleton": json.dumps(skeleton, ensure_ascii=False, indent=2)
    }


# Node: In case the gounding doesn't find anything relevant we perform more general retrieval not taking in consideration the classifier decision
async def retrieve_fallback(state: State) -> dict:
    print("[Started Node] retrieve_fallback")
    time = datetime.now()

    embedding = await embedder.embed_query(state.input_text)
    articles = await store.retrieve(
        embedding=embedding,
        query=state.input_text,
        top_k=5
    )

    skeleton = {
        a.breadcrumb: {"relevant": None, "excerpts": []}
        for a in articles
    }

    print(f"[Finished Node] retrieve_fallback In: {(datetime.now() - time).total_seconds()} seconds, no filter, retrieved={len(articles)}")
    return {
        "retrieved_articles": articles,
        "grounded_skeleton": json.dumps(skeleton, ensure_ascii=False, indent=2),
        "fallback_attempted": True
    }