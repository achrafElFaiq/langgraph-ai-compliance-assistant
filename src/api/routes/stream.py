import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from uuid import uuid4

from src.api.schemas.chat import ChatRequest
from src.application.agent.graph import compiled_graph

router = APIRouter()

REG_NAMES = {
    "mica": "MiCA",
    "dora": "DORA",
    "ai_act": "AI Act",
    "gdpr": "GDPR",
}

NODE_LABELS_ACTIVE: dict[str, str] = {
    "classify_intent":   "Analyse de votre question",
    "classify":          "Identification des régulations applicables",
    "retrieve_articles": "Récupération des articles réglementaires",
    "retrieve_fallback": "Élargissement de la recherche",
    "ground":            "Ancrage des articles sur votre question",
    "apply":             "Application du droit à votre contexte",
    "answer":            "Rédaction de la réponse",
    "critic_answer":     "Vérification de l'exactitude",
    "synthesize":        "Génération du rapport de synthèse",
    "direct_answer":     "Génération de la réponse",
}

NODE_LABELS_DONE: dict[str, str] = {
    "classify_intent":   "Intention classifiée",
    "classify":          "Régulations identifiées",
    "retrieve_articles": "Articles récupérés",
    "retrieve_fallback": "Recherche élargie",
    "ground":            "Articles ancrés",
    "apply":             "Droit appliqué",
    "answer":            "Réponse rédigée",
    "critic_answer":     "Réponse vérifiée",
    "synthesize":        "Rapport synthétisé",
    "direct_answer":     "Réponse générée",
}


def _done_label(node: str, output: dict) -> str:
    if node == "classify":
        regs = output.get("regulations", [])
        if regs:
            names = [REG_NAMES.get(r, r.upper()) for r in regs]
            return f"Identifiées : {', '.join(names)}"

    elif node in ("retrieve_articles", "retrieve_fallback"):
        articles = output.get("retrieved_articles", [])
        count = len(articles) if hasattr(articles, "__len__") else 0
        prefix = "Élargi : " if node == "retrieve_fallback" else ""
        if count:
            return f"{prefix}{count} article{'s' if count != 1 else ''} récupéré{'s' if count != 1 else ''}"

    elif node == "ground":
        skeleton = output.get("grounded_skeleton", "")
        if skeleton:
            try:
                parsed = json.loads(skeleton)
                relevant = sum(1 for v in parsed.values() if v.get("relevant") is True)
                total = len(parsed)
                return f"{relevant}/{total} articles pertinents"
            except (json.JSONDecodeError, AttributeError):
                pass

    elif node == "critic_answer":
        if output.get("critic_opinion", "x") == "":
            return "Réponse vérifiée"
        return "Problèmes détectés, affinage en cours"

    return NODE_LABELS_DONE.get(node, node)


@router.post(
    "/chat/stream",
    summary="Stream a compliance query",
    description="Runs the full RAG pipeline and streams progress events (node start/end) followed by the final answer as Server-Sent Events (SSE).",
    tags=["chat"],
)
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    async def generate():
        async for chunk in compiled_graph.astream(
            {"input_text": request.input_text},
            config=config,
            stream_mode="debug",
        ):
            chunk_type = chunk.get("type")
            payload = chunk.get("payload", {})
            node = payload.get("name", "")

            if not node or node not in NODE_LABELS_ACTIVE:
                continue

            if chunk_type == "task":
                data = {
                    "type": "node_start",
                    "node": node,
                    "label": NODE_LABELS_ACTIVE[node],
                }
                yield f"data: {json.dumps(data)}\n\n"

            elif chunk_type == "task_result":
                try:
                    output = dict(payload.get("result", []))
                except (TypeError, ValueError):
                    output = {}
                data = {
                    "type": "node_end",
                    "node": node,
                    "label": _done_label(node, output),
                }
                yield f"data: {json.dumps(data)}\n\n"

        # Final state from checkpointer
        state = await compiled_graph.aget_state(config)
        values = state.values

        def _get(key, default=None):
            if isinstance(values, dict):
                return values.get(key, default)
            return getattr(values, key, default)

        citations = []
        grounded = _get("grounded_skeleton", "")
        if grounded:
            try:
                for breadcrumb, info in json.loads(grounded).items():
                    citations.append({
                        "breadcrumb": breadcrumb,
                        "relevant": info.get("relevant", False),
                        "excerpts": info.get("excerpts", []),
                    })
            except (json.JSONDecodeError, AttributeError):
                pass

        yield f"data: {json.dumps({
            'type': 'done',
            'answer': _get('answer', ''),
            'thread_id': thread_id,
            'regulations': _get('regulations', []),
            'route': _get('route', ''),
            'retry_count': _get('retry_count', 0),
            'fallback_attempted': _get('fallback_attempted', False),
            'citations': citations,
            'final_report': _get('final_report', ''),
        })}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
