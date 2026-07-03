"""Integration test for the /chat/stream SSE endpoint.

The agent is mocked to emit a known node sequence; we assert the endpoint's SSE
*output contract*: progress events surface classify → ground → answer in order.
Parsing accepts either an SSE ``event:`` label or a ``data:`` JSON ``node`` field,
since the exact SSE encoding of the node id is not part of the ordering contract.
"""
import json

import pytest

pytestmark = pytest.mark.integration


def _node_sequence(sse_text: str):
    """Extract the ordered list of node ids announced as they start."""
    seq = []
    for raw in sse_text.splitlines():
        line = raw.strip()
        if line.startswith("event:"):
            seq.append(line.split(":", 1)[1].strip())
        elif line.startswith("data:"):
            try:
                obj = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "node_start" and obj.get("node"):
                seq.append(obj["node"])
    return seq


class _FakeState:
    values = {
        "answer": "réponse finale",
        "final_report": "",
        "route": "research",
        "regulations": ["MiCA"],
        "grounded_skeleton": "",
        "retry_count": 0,
        "fallback_attempted": False,
    }


class _FakeGraph:
    async def astream(self, inputs, config=None, stream_mode=None):
        for name in ("classify", "ground", "answer"):
            yield {"type": "task", "payload": {"name": name}}
            yield {"type": "task_result", "payload": {"name": name, "result": []}}

    async def aget_state(self, config):
        return _FakeState()


async def test_stream_emits_nodes_in_pipeline_order(client, monkeypatch):
    """/chat/stream announces classify → ground → answer, in that order.

    Matters because the UI renders these as live progress steps; out-of-order or
    missing events make the pipeline's state unreadable to the user.
    """
    import src.api.routes.stream as stream_mod
    monkeypatch.setattr(stream_mod, "compiled_graph", _FakeGraph())

    body = ""
    async with client.stream("POST", "/chat/stream", json={"input_text": "q"}) as resp:
        assert resp.status_code == 200
        async for chunk in resp.aiter_text():
            body += chunk

    seq = _node_sequence(body)
    for node in ("classify", "ground", "answer"):
        assert node in seq, f"expected node {node!r} in stream; got {seq}"
    assert seq.index("classify") < seq.index("ground") < seq.index("answer")


async def test_stream_terminates_with_done_event(client, monkeypatch):
    """The stream ends with a 'done' event carrying the final answer.

    Matters because the client relies on 'done' to stop reading and render the answer.
    """
    import src.api.routes.stream as stream_mod
    monkeypatch.setattr(stream_mod, "compiled_graph", _FakeGraph())

    body = ""
    async with client.stream("POST", "/chat/stream", json={"input_text": "q"}) as resp:
        async for chunk in resp.aiter_text():
            body += chunk

    done_events = [
        json.loads(line[len("data:"):].strip())
        for line in body.splitlines()
        if line.strip().startswith("data:") and '"done"' in line
    ]
    assert done_events, "no 'done' event emitted"
    assert done_events[-1].get("type") == "done"
    assert "answer" in done_events[-1]
