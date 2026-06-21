from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.application.agent.state import State
from src.config.init_llm import llm



# Classifies the user input either reseacrh, followup, or chitchat
# In case of generate synthesis it goes to synthesis

async def classify_intent(state: State) -> dict:
    print("[Started Node] classify_intent")

    if state.input_text == "Generate synthesis":
        return {"route": "synthesis"}


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

    print(f"[Finished Node] classify_intent, route={route}")

    reset = {"route": route}
    if route == "research":
        reset.update({
            "retry_count": 0,
            "critic_opinion": "",
            "fallback_attempted": False,
        })

    return reset