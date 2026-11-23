"""End-to-end tests of RAGSystem.query for content questions (Claude API mocked)."""
from unittest.mock import MagicMock

import pytest

from config import config
from conftest import COURSE_TITLE, make_response, text_block, tool_use_block
from rag_system import RAGSystem


def build_rag(cfg, sample_course, sample_chunks) -> RAGSystem:
    rag = RAGSystem(cfg)
    rag.vector_store.add_course_metadata(sample_course)
    rag.vector_store.add_course_content(sample_chunks)
    rag.ai_generator.client = MagicMock()
    return rag


def script_search(rag, tool_input, final_text="Tool use lets Claude call your functions."):
    rag.ai_generator.client.messages.create.side_effect = [
        make_response([tool_use_block("search_course_content", tool_input)], stop_reason="tool_use"),
        make_response([text_block(final_text)]),
    ]


def tool_result_sent_to_claude(rag) -> str:
    second_call = rag.ai_generator.client.messages.create.call_args_list[1].kwargs
    return second_call["messages"][-1]["content"][0]["content"]


@pytest.fixture
def rag(test_config, sample_course, sample_chunks):
    return build_rag(test_config, sample_course, sample_chunks)


def test_content_query_end_to_end(rag):
    script_search(rag, {"query": "how does tool use work", "course_name": "Computer Use"})

    answer, sources = rag.query("How does tool use work?")

    assert answer == "Tool use lets Claude call your functions."
    result = tool_result_sent_to_claude(rag)
    assert "Search error" not in result
    assert "tool_result block" in result or "JSON schema" in result

    assert sources, "UI sources should be populated after a search"
    assert all(s["course_title"] == COURSE_TITLE for s in sources)
    assert any(s["link"] for s in sources)

    # Sources are reset after being returned
    assert rag.tool_manager.get_last_sources() == []


def test_content_query_with_production_max_results(test_config, sample_course, sample_chunks):
    """Same flow, but VectorStore uses the app's real config.MAX_RESULTS."""
    test_config.MAX_RESULTS = config.MAX_RESULTS
    rag = build_rag(test_config, sample_course, sample_chunks)
    script_search(rag, {"query": "how does tool use work"})

    _, sources = rag.query("How does tool use work?")

    result = tool_result_sent_to_claude(rag)
    assert "Search error" not in result, result
    assert sources


def test_query_prompt_and_tools_passed(rag):
    rag.ai_generator.client.messages.create.return_value = make_response([text_block("hi")])
    rag.query("What is RAG?")
    kwargs = rag.ai_generator.client.messages.create.call_args.kwargs
    assert "What is RAG?" in kwargs["messages"][0]["content"]
    assert {t["name"] for t in kwargs["tools"]} == {"search_course_content", "get_course_outline"}


def test_session_history_recorded_and_reused(rag):
    sid = rag.session_manager.create_session()
    rag.ai_generator.client.messages.create.return_value = make_response([text_block("first answer")])
    rag.query("first question", session_id=sid)
    rag.query("second question", session_id=sid)

    system = rag.ai_generator.client.messages.create.call_args.kwargs["system"]
    assert "User: first question" in system
    assert "Assistant: first answer" in system


def test_outline_query_returns_course_source(rag):
    rag.ai_generator.client.messages.create.side_effect = [
        make_response([tool_use_block("get_course_outline", {"course_title": "Computer Use"})], stop_reason="tool_use"),
        make_response([text_block("outline")]),
    ]
    _, sources = rag.query("Outline of the computer use course?")
    result = tool_result_sent_to_claude(rag)
    assert "Lesson 1: Tool Use Basics" in result
    assert sources == [{"text": COURSE_TITLE, "link": "https://example.com/course",
                        "course_title": COURSE_TITLE, "lesson_number": None}]
