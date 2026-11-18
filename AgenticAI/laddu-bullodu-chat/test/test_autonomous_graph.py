import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.autonomous_graph as autonomous
import src.graph_nodes as graph_nodes
import src.intent_router as intent_router
from src.llm_provider import OLLAMA_MODEL, OPENROUTER_FREE_MODEL


MIXED_SITE_SELECTION_GOAL = (
    "Explain about this chat assistant feature followed by data center site "
    "selection in West Indies"
)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"next_agent": "delegate_to_web_search"}, "worker"),
        ({"next_agent": "finish_task"}, "finalize"),
    ],
)
def test_route_after_supervisor(state, expected):
    """Route supervisor tool calls to a specialist or final synthesis."""
    assert autonomous.route_after_supervisor(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"is_complete": True, "rounds": 1}, "finalize"),
        ({"is_complete": False, "rounds": autonomous.MAX_WORKER_ROUNDS}, "finalize"),
        ({"is_complete": False, "rounds": 1, "supervisor_turns": 1}, "supervisor"),
    ],
)
def test_route_after_critic_is_bounded(state, expected):
    """Continue only when review identifies gaps and worker budget remains."""
    assert autonomous.route_after_critic(state) == expected


def test_parse_critique_requires_valid_decision():
    """Accept the critic JSON contract and retry on malformed model output."""
    assert autonomous._parse_critique('{"complete": true, "feedback": "Enough evidence."}') == (
        True,
        "Enough evidence.",
    )
    complete, feedback = autonomous._parse_critique("not valid JSON")
    assert complete is False
    assert "not valid JSON" in feedback


def test_required_specialists_cover_mixed_goal_in_order():
    """Require the dedicated site-selection worker and project RAG worker."""
    assert autonomous._required_specialist_tasks(MIXED_SITE_SELECTION_GOAL) == [
        (
            "delegate_to_site_selection",
            "Perform the requested data-center or commercial site-selection "
            "analysis for the locations or region in this goal. Preserve any "
            "requested geography and criteria. Original goal: "
            + MIXED_SITE_SELECTION_GOAL,
        ),
        (
            "delegate_to_rag",
            "Use internal project documentation to explain the features and "
            "capabilities of this chat assistant. Original goal: "
            + MIXED_SITE_SELECTION_GOAL,
        ),
    ]


def test_supervisor_forces_pending_site_selection_then_rag(monkeypatch):
    """Do not let generic web search replace explicitly requested specialists."""
    monkeypatch.setattr(
        autonomous,
        "get_chat_model",
        lambda _: pytest.fail("Mandatory specialist routing should not call the LLM"),
    )
    initial = {
        "goal": MIXED_SITE_SELECTION_GOAL,
        "worker_results": [],
        "rounds": 0,
        "supervisor_turns": 0,
    }

    site_task = autonomous.supervisor_node(initial)
    assert site_task["next_agent"] == "delegate_to_site_selection"
    assert "West Indies" in site_task["worker_query"]
    assert autonomous.route_after_critic(
        {
            **initial,
            "is_complete": True,
            "worker_results": [
                {
                    "agent": "delegate_to_site_selection",
                    "query": site_task["worker_query"],
                    "result": "Site report",
                }
            ],
            "rounds": 1,
            "supervisor_turns": 1,
        }
    ) == "supervisor"

    after_site_selection = {
        **initial,
        "worker_results": [{"agent": "delegate_to_site_selection", "query": "", "result": "Site report"}],
        "rounds": 1,
        "supervisor_turns": 1,
    }
    rag_task = autonomous.supervisor_node(after_site_selection)
    assert rag_task["next_agent"] == "delegate_to_rag"
    assert "internal project documentation" in rag_task["worker_query"]


def test_worker_executes_selected_site_selection_specialist(monkeypatch):
    """Verify deterministic routing reaches the existing site-selection workflow."""
    executions = []

    def fake_domain_worker(intent, query, selected_model):
        executions.append((intent, query, selected_model))
        return "Site-selection report"

    monkeypatch.setattr(autonomous, "_run_domain_worker", fake_domain_worker)
    result = autonomous.worker_node(
        {
            "next_agent": "delegate_to_site_selection",
            "worker_query": "Compare West Indies data-center hubs",
            "selected_model": "test-model",
            "worker_results": [],
            "rounds": 0,
        }
    )

    assert executions == [
        ("SITE_SELECTION", "Compare West Indies data-center hubs", "test-model")
    ]
    assert result["worker_results"][0]["agent"] == "delegate_to_site_selection"
    assert result["rounds"] == 1


def test_west_indies_site_selection_uses_caribbean_hubs(monkeypatch):
    """Avoid falling back to United States locations for the West Indies."""
    monkeypatch.setattr(
        intent_router,
        "get_chat_model",
        lambda _: pytest.fail("Regional defaults should not require model extraction"),
    )

    country, locations = intent_router.extract_site_parameters(
        "Data center site selection in the West Indies"
    )

    assert country == "Caribbean (West Indies)"
    assert locations == [
        "San Juan Puerto Rico",
        "Santo Domingo Dominican Republic",
        "Port of Spain Trinidad and Tobago",
    ]


@pytest.mark.parametrize(
    ("selected_model", "expected_local_embeddings"),
    [
        (OPENROUTER_FREE_MODEL, True),
        (OLLAMA_MODEL, False),
    ],
)
def test_retrieve_node_selects_embedding_backend(
    monkeypatch, selected_model, expected_local_embeddings
):
    """Keep cloud-model retrieval independent of the Ollama embedding server."""
    embedding_options = []

    class FakeVectorStore:
        def similarity_search(self, query, k):
            return []

    def fake_load_embeddings_and_vector_store(use_local_embeddings):
        embedding_options.append(use_local_embeddings)
        return FakeVectorStore()

    monkeypatch.setattr(
        graph_nodes,
        "load_embeddings_and_vector_store",
        fake_load_embeddings_and_vector_store,
    )
    monkeypatch.setattr(graph_nodes, "get_or_init_hybrid_retriever", lambda _: None)

    graph_nodes.retrieve_node(
        {
            "messages": [HumanMessage(content="Explain this feature")],
            "selected_model": selected_model,
        }
    )

    assert embedding_options == [expected_local_embeddings]


def test_autonomous_graph_delegates_and_synthesizes(monkeypatch):
    """Exercise supervisor, specialist, critic, finalizer, and guardrail with fakes."""
    class FakeBoundSupervisor:
        def invoke(self, messages):
            assert messages
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_to_web_search",
                        "args": {"query": "Search current AI agent news"},
                        "id": "call_web",
                        "type": "tool_call",
                    }
                ],
            )

    class FakeModel:
        def bind_tools(self, tools, tool_choice):
            assert tool_choice == "any"
            assert "delegate_to_web_search" in {tool.name for tool in tools}
            return FakeBoundSupervisor()

        def invoke(self, messages):
            prompt = str(messages[-1].content)
            if "You are a skeptical quality reviewer." in str(messages[0].content):
                return AIMessage(
                    content='{"complete": true, "feedback": "The search results answer the goal."}'
                )
            assert "Specialist findings:" in prompt
            return AIMessage(content="Synthesized answer based on current search results.")

    def fake_get_chat_model(model_name):
        assert model_name == "test-model"
        return FakeModel()

    def fake_domain_worker(intent, query, selected_model):
        assert intent == "WEB"
        assert query == "Search current AI agent news"
        assert selected_model == "test-model"
        return "Current search evidence."

    def fake_log_execution(**kwargs):
        assert kwargs["status"] == "SUCCESS"

    monkeypatch.setattr(autonomous, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(autonomous, "_run_domain_worker", fake_domain_worker)
    monkeypatch.setattr("src.graph_nodes.log_execution", fake_log_execution)

    graph = autonomous.autonomous_workflow.compile()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What is current AI agent news?")],
            "selected_model": "test-model",
            "thread_id": "autonomous-test",
        }
    )

    assert result["rounds"] == 1
    assert result["worker_results"][0]["agent"] == "delegate_to_web_search"
    assert result["is_complete"] is True
    assert result["messages"][-1].content == (
        "[AUTONOMOUS] Synthesized answer based on current search results."
    )
