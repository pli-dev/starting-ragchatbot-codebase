"""Tests that AIGenerator calls tools (CourseSearchTool etc.) correctly."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator
from conftest import make_response, text_block, tool_use_block
from search_tools import CourseSearchTool, ToolManager

TOOLS = [{"name": "search_course_content", "input_schema": {"type": "object"}}]


@pytest.fixture
def generator():
    with patch("ai_generator.anthropic.Anthropic") as MockClient:
        gen = AIGenerator(api_key="test-key", model="claude-test")
        gen.client = MockClient.return_value
        yield gen


def test_direct_answer_no_tool(generator):
    generator.client.messages.create.return_value = make_response([text_block("Paris")])
    tm = MagicMock()

    out = generator.generate_response("capital of France?", tools=TOOLS, tool_manager=tm)

    assert out == "Paris"
    assert generator.client.messages.create.call_count == 1
    kwargs = generator.client.messages.create.call_args.kwargs
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == {"type": "auto"}
    tm.execute_tool.assert_not_called()


def test_search_tool_called_with_model_inputs(generator):
    tool_input = {"query": "tool use", "course_name": "Computer Use"}
    generator.client.messages.create.side_effect = [
        make_response([text_block("Let me search."), tool_use_block("search_course_content", tool_input, "toolu_42")],
                      stop_reason="tool_use"),
        make_response([text_block("Tool use lets Claude call functions.")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "[Course - Lesson 1]\nchunk text"

    out = generator.generate_response("What is tool use?", tools=TOOLS, tool_manager=tm)

    assert out == "Tool use lets Claude call functions."
    tm.execute_tool.assert_called_once_with("search_course_content", **tool_input)

    second = generator.client.messages.create.call_args_list[1].kwargs
    msgs = second["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_42", "content": "[Course - Lesson 1]\nchunk text"}],
    }
    assert "system" in second


def test_real_tool_manager_routes_to_course_search_tool(generator, mock_vector_store):
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_vector_store))
    generator.client.messages.create.side_effect = [
        make_response([tool_use_block("search_course_content", {"query": "q", "lesson_number": 1})],
                      stop_reason="tool_use"),
        make_response([text_block("answer")]),
    ]

    generator.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)

    mock_vector_store.search.assert_called_once_with(query="q", course_name=None, lesson_number=1)


def test_outline_tool_routed(generator):
    generator.client.messages.create.side_effect = [
        make_response([tool_use_block("get_course_outline", {"course_title": "MCP"})], stop_reason="tool_use"),
        make_response([text_block("outline")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "Course: MCP"

    assert generator.generate_response("outline of MCP", tools=TOOLS, tool_manager=tm) == "outline"
    tm.execute_tool.assert_called_once_with("get_course_outline", course_title="MCP")


def test_extract_text_skips_non_text_blocks():
    resp = make_response([SimpleNamespace(type="thinking", thinking="hmm"), text_block("A"), text_block("B")])
    assert AIGenerator._extract_text(resp) == "AB"


def test_history_injected_into_system_prompt(generator):
    generator.client.messages.create.return_value = make_response([text_block("ok")])
    generator.generate_response("q", conversation_history="User: hi\nAssistant: hello")
    system = generator.client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:\nUser: hi\nAssistant: hello" in system


def test_tool_exception_does_not_crash_request(generator, mock_vector_store):
    """A failing tool should be reported to Claude as a tool_result, not raise out of the request."""
    mock_vector_store.search.side_effect = RuntimeError("db down")
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_vector_store))
    generator.client.messages.create.side_effect = [
        make_response([tool_use_block("search_course_content", {"query": "q"})], stop_reason="tool_use"),
        make_response([text_block("Sorry, search failed.")]),
    ]

    out = generator.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)
    assert out == "Sorry, search failed."
