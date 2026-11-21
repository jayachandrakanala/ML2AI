import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.autonomous_graph as autonomous
import src.graph_nodes as graph_nodes
import src.intent_router as intent_router
import src.logger as logger
import src.reranker as reranker
import src.site_selector as site_selector
import src.weather as weather
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


def test_route_after_critic_does_not_repeat_site_selection():
    """Finalize site-selection evidence instead of rerunning its fixed pipeline."""
    state = {
        "goal": "Compare data center site selection in Dubai",
        "is_complete": False,
        "rounds": 1,
        "supervisor_turns": 1,
        "worker_results": [
            {
                "agent": "delegate_to_site_selection",
                "query": "Compare data center sites in Dubai",
                "result": "Site-selection report with evidence gaps.",
            }
        ],
    }

    assert autonomous.route_after_critic(state) == "finalize"


def test_route_after_critic_runs_other_required_specialists_before_stopping():
    """A site-selection critique must not skip another mandatory subtask."""
    state = {
        "goal": MIXED_SITE_SELECTION_GOAL,
        "is_complete": False,
        "rounds": 1,
        "supervisor_turns": 1,
        "worker_results": [
            {
                "agent": "delegate_to_site_selection",
                "query": "Compare West Indies data-center hubs",
                "result": "Site-selection report with evidence gaps.",
            }
        ],
    }

    assert autonomous.route_after_critic(state) == "supervisor"


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
    """Require dedicated site-selection and project-document workers."""
    assert autonomous._required_specialist_tasks(MIXED_SITE_SELECTION_GOAL) == [
        (
            "delegate_to_site_selection",
            "Perform the requested data-center or commercial site-selection "
            "analysis for the locations or region in this goal. Preserve any "
            "requested geography and criteria. Original goal: "
            + MIXED_SITE_SELECTION_GOAL,
        ),
        (
            "delegate_to_project_docs",
            "Use the local project architecture documentation to explain the "
            "implemented chat assistant features and any specifically "
            "requested workflow details. Do not invent capabilities not "
            "listed in the docs. "
            "Original goal: "
            + MIXED_SITE_SELECTION_GOAL,
        ),
    ]


def test_required_specialists_cover_weather_and_autonomous_explanation():
    """Route current weather to web and project mechanics to direct local docs."""
    goal = (
        "What is today's weather report in Dubai, and how does autonomous "
        "multi-agent mode route and review work?"
    )

    tasks = autonomous._required_specialist_tasks(goal)

    assert [agent for agent, _ in tasks] == [
        "delegate_to_web_search",
        "delegate_to_project_docs",
    ]
    assert "Dubai" in tasks[0][1]
    assert "local project architecture documentation" in tasks[1][1]


def test_required_specialists_cover_features_and_dubai_water_supply():
    """Use direct project docs and require water-specific Dubai site research."""
    goal = (
        "Explain about chat assistant features and how water supply will be "
        "provided for data center setup in Dubai"
    )

    tasks = autonomous._required_specialist_tasks(goal)

    assert [agent for agent, _ in tasks] == [
        "delegate_to_site_selection",
        "delegate_to_project_docs",
    ]
    assert "water source" in tasks[0][1]
    assert "Dubai" in tasks[0][1]
    assert "Do not invent capabilities" in tasks[1][1]


def test_supervisor_forces_project_docs_specialist(monkeypatch):
    """Do not use low-relevance RAG/web search for the project's own workflow."""
    monkeypatch.setattr(
        autonomous,
        "get_chat_model",
        lambda _: pytest.fail("The project documentation specialist is mandatory."),
    )

    decision = autonomous.supervisor_node(
        {
            "goal": "How does autonomous multi-agent mode route and review work?",
            "worker_results": [],
            "rounds": 0,
            "supervisor_turns": 0,
        }
    )

    assert decision["next_agent"] == "delegate_to_project_docs"
    assert "local project architecture documentation" in decision["worker_query"]


def test_critic_reviews_all_worker_findings_as_a_whole(monkeypatch):
    """A specialist need not answer subtasks covered by earlier specialist results."""
    prompts = []

    class FakeReviewer:
        def invoke(self, messages):
            prompts.append(str(messages[-1].content))
            return AIMessage(
                content='{"complete": true, "feedback": "Both findings answer the goal."}'
            )

    monkeypatch.setattr(autonomous, "get_chat_model", lambda _: FakeReviewer())
    state = {
        "goal": "Get Dubai weather and explain autonomous routing.",
        "selected_model": "test-model",
        "worker_results": [
            {
                "agent": "delegate_to_web_search",
                "query": "Dubai weather",
                "result": "Dubai is clear and 25 C.",
            },
            {
                "agent": "delegate_to_rag",
                "query": "Autonomous routing",
                "result": "The supervisor delegates and critic reviews evidence.",
            },
        ],
    }

    result = autonomous.critic_node(state)

    assert result["is_complete"] is True
    assert "Dubai is clear" in prompts[0]
    assert "critic reviews evidence" in prompts[0]
    assert "findings collectively" in prompts[0]


def test_supervisor_forces_pending_site_selection_then_project_docs(monkeypatch):
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
    docs_task = autonomous.supervisor_node(after_site_selection)
    assert docs_task["next_agent"] == "delegate_to_project_docs"
    assert "local project architecture documentation" in docs_task["worker_query"]


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


def test_project_docs_worker_returns_relevant_checked_in_documents():
    """Provide implemented features and workflow details as direct local evidence."""
    result = autonomous._run_project_docs_worker(
        "Explain about chat assistant features and autonomous workflow."
    )

    assert "Streamlit chat interface" in result
    assert "voice interaction or image generation" in result
    assert "four specialist executions" in result
    assert "critic" in result
    assert "supervisor" in result
    assert "finalizer" in result


def test_site_selection_searches_and_evaluates_water_supply(monkeypatch):
    """Carry water evidence through site research and into evaluation prompts."""
    searches = []
    evaluation_prompts = []

    class FakeSerper:
        def run(self, query):
            searches.append(query)
            return f"Search evidence for {query}"

    class FakeModel:
        def invoke(self, prompt):
            evaluation_prompts.append(str(prompt))
            return AIMessage(content="Water availability needs verification.")

    monkeypatch.setattr(site_selector, "GoogleSerperAPIWrapper", FakeSerper)
    monkeypatch.setattr(site_selector, "get_chat_model", lambda _: FakeModel())

    raw_data = site_selector.supervisor_agent_node(
        {
            "country": "United Arab Emirates",
            "locations": ["Dubai UAE"],
            "raw_data": {},
            "evaluations": {},
            "comparison_report": "",
            "report_filepath": "",
        }
    )["raw_data"]
    evaluations = site_selector.evaluation_subgraph_node(
        {
            "country": "United Arab Emirates",
            "locations": ["Dubai UAE"],
            "raw_data": raw_data,
            "evaluations": {},
            "comparison_report": "",
            "report_filepath": "",
        }
    )["evaluations"]

    assert len(searches) == 4
    assert any("water supply" in query.lower() for query in searches)
    assert "water_supply" in raw_data["Dubai UAE"]
    assert "WATER SUPPLY & COOLING" in evaluation_prompts[0]
    assert "capacity, reliability, quality, and permits" in evaluation_prompts[0]
    assert "Dubai UAE" in evaluations


def test_worker_executes_selected_project_docs_specialist():
    """Ensure supervisor delegation returns the checked-in project documentation."""
    result = autonomous.worker_node(
        {
            "next_agent": "delegate_to_project_docs",
            "worker_query": "Explain the autonomous workflow.",
            "worker_results": [],
            "rounds": 0,
        }
    )

    assert result["worker_results"][0]["agent"] == "delegate_to_project_docs"
    assert "Streamlit chat interface" in result["worker_results"][0]["result"]
    assert "four specialist executions" in result["worker_results"][0]["result"]


def test_weather_location_extraction_and_report(monkeypatch):
    """Build a local-time weather report from geocoding and forecast API responses."""
    requested_urls = []
    responses = iter(
        [
            {
                "results": [
                    {
                        "name": "Dubai",
                        "admin1": "Dubai",
                        "country": "United Arab Emirates",
                        "latitude": 25.2,
                        "longitude": 55.27,
                    }
                ]
            },
            {
                "timezone": "Asia/Dubai",
                "current": {
                    "time": "2026-11-18T20:00",
                    "temperature_2m": 25,
                    "apparent_temperature": 26,
                    "relative_humidity_2m": 60,
                    "precipitation": 0,
                    "weather_code": 0,
                    "wind_speed_10m": 5,
                },
                "daily": {
                    "time": ["2026-11-18"],
                    "temperature_2m_min": [22],
                    "temperature_2m_max": [29],
                    "precipitation_probability_max": [10],
                    "sunrise": ["2026-11-18T06:40"],
                    "sunset": ["2026-11-18T17:30"],
                },
            },
        ]
    )

    def fake_get_json(url):
        requested_urls.append(url)
        return next(responses)

    monkeypatch.setattr(weather, "_get_json", fake_get_json)

    query = (
        "What is today's weather report in Dubai, How does autonomous mode work?"
    )
    location = weather.extract_weather_location(query)
    report = weather.get_weather_report(location)

    assert location == "Dubai"
    assert weather.extract_weather_location("Weather in Dubai today") == "Dubai"
    assert weather.extract_weather_location(
        "Today's weather in Ireland\n\nFetch current conditions and today's forecast."
    ) == "Ireland"
    assert "2026-11-18T20:00 (Asia/Dubai)" in report
    assert "Today (2026-11-18)" in report
    assert "25 °C" in report
    assert len(requested_urls) == 2
    assert "timezone=auto" in requested_urls[1]


def test_web_weather_route_uses_live_weather_client(monkeypatch):
    """Weather intent uses the direct weather endpoint rather than generic search."""
    monkeypatch.setattr(
        graph_nodes,
        "get_weather_report",
        lambda location: f"Weather evidence for {location}",
    )
    class FakeSerperSearch:
        def run(self, _):
            pytest.fail("Weather must use the dedicated forecast client.")

    monkeypatch.setattr(graph_nodes, "serper_search", FakeSerperSearch())

    result = graph_nodes.domain_executor_node(
        {
            "messages": [
                HumanMessage(content="What is today's weather report in Dubai?")
            ],
            "intent": "WEB",
            "selected_model": "unused-model",
        }
    )

    assert result["messages"][-1].content == "[WEB] Weather evidence for Dubai"


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


def test_dubai_site_selection_uses_requested_uae_city(monkeypatch):
    """Keep Dubai site-selection requests local and independent of model extraction."""
    monkeypatch.setattr(
        intent_router,
        "get_chat_model",
        lambda _: pytest.fail("Dubai defaults should not require model extraction"),
    )

    country, locations = intent_router.extract_site_parameters(
        "Data center setup in Dubai, with water supply analysis"
    )

    assert country == "United Arab Emirates"
    assert locations == ["Dubai UAE"]


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


def test_embedding_cache_reports_progress_outside_cached_loader(
    monkeypatch, tmp_path
):
    """Cached initialization emits no UI callback; the uncached wrapper owns status."""
    constructed_stores = []
    status_messages = []

    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        def __init__(self, **kwargs):
            constructed_stores.append(kwargs)

    monkeypatch.setattr(
        "src.local_embeddings.LocalEmbeddingGemmaEmbeddings", FakeEmbeddings
    )
    monkeypatch.setattr(graph_nodes, "Chroma", FakeVectorStore)
    monkeypatch.setattr(graph_nodes, "DOCS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(graph_nodes, "PERSIST_DIRECTORY", str(tmp_path))
    graph_nodes._cached_load_embeddings_and_vector_store.clear()
    logger.set_status_callback(status_messages.append)
    try:
        first_store = graph_nodes.load_embeddings_and_vector_store(True)
        second_store = graph_nodes.load_embeddings_and_vector_store(True)
    finally:
        logger.set_status_callback(None)
        graph_nodes._cached_load_embeddings_and_vector_store.clear()

    assert first_store is second_store
    assert len(constructed_stores) == 1
    assert status_messages == [
        "Preparing local Hugging Face EmbeddingGemma embeddings and vector store...",
        "Vector store is ready.",
        "Preparing local Hugging Face EmbeddingGemma embeddings and vector store...",
        "Vector store is ready.",
    ]


def test_reranker_cache_reports_progress_outside_cached_loader(monkeypatch):
    """The cached CrossEncoder constructor does not replay Streamlit status UI."""
    constructed_models = []
    status_messages = []

    class FakeCrossEncoder:
        def __init__(self, model_name, device):
            constructed_models.append((model_name, device))

    monkeypatch.setattr(reranker, "CrossEncoder", FakeCrossEncoder)
    reranker._cached_load_reranker_model.clear()
    logger.set_status_callback(status_messages.append)
    try:
        first_model = reranker.load_reranker_model()
        second_model = reranker.load_reranker_model()
    finally:
        logger.set_status_callback(None)
        reranker._cached_load_reranker_model.clear()

    assert first_model is second_model
    assert constructed_models == [("BAAI/bge-reranker-large", "cpu")]
    assert status_messages == [
        "Preparing open-source BAAI bge-reranker-large model...",
        "Preparing open-source BAAI bge-reranker-large model...",
    ]


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
