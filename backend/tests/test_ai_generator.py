"""Tests that AIGenerator calls tools (CourseSearchTool etc.) correctly."""
import copy
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


# --------------------------------------------------------------------------- #
# Sequential tool calling (up to 2 rounds)
# --------------------------------------------------------------------------- #
def script(generator, responses):
    """Feed scripted responses; return a list of deep-copied kwargs, one per API call.

    Snapshots are needed because the messages list is mutated between rounds.
    """
    calls = []
    queue = list(responses)

    def fake_create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return queue.pop(0)

    generator.client.messages.create.side_effect = fake_create
    return calls


def tool_round(name, input, id):
    return make_response([tool_use_block(name, input, id)], stop_reason="tool_use")


def test_two_round_chain_makes_three_calls(generator):
    calls = script(generator, [
        tool_round("get_course_outline", {"course_title": "X"}, "t1"),
        tool_round("search_course_content", {"query": "Lesson four topic"}, "t2"),
        make_response([text_block("Course Y covers it.")]),
    ])
    tm = MagicMock()
    tm.execute_tool.side_effect = ["outline: lesson 4 = Lesson four topic", "Course Y chunk"]

    out = generator.generate_response("same topic as lesson 4 of X?", tools=TOOLS, tool_manager=tm)

    assert out == "Course Y covers it."
    assert [c.args for c in tm.execute_tool.call_args_list] == [("get_course_outline",), ("search_course_content",)]
    assert tm.execute_tool.call_args_list[1].kwargs == {"query": "Lesson four topic"}
    assert len(calls) == 3
    assert "tools" in calls[0] and "tools" in calls[1]
    assert "tools" not in calls[2]


def test_context_preserved_between_rounds(generator):
    calls = script(generator, [
        tool_round("get_course_outline", {"course_title": "X"}, "t1"),
        tool_round("search_course_content", {"query": "q"}, "t2"),
        make_response([text_block("done")]),
    ])
    tm = MagicMock()
    tm.execute_tool.side_effect = ["r1", "r2"]

    generator.generate_response("question", tools=TOOLS, tool_manager=tm)

    assert len(calls[1]["messages"]) == 3
    final = calls[2]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user", "assistant", "user"]
    assert final[0]["content"] == "question"
    assert final[2]["content"][0]["tool_use_id"] == "t1"
    assert final[2]["content"][0]["content"] == "r1"
    assert final[4]["content"][0]["tool_use_id"] == "t2"
    assert final[4]["content"][0]["content"] == "r2"


def test_second_call_still_has_tools_after_single_round(generator):
    calls = script(generator, [
        tool_round("search_course_content", {"query": "q"}, "t1"),
        make_response([text_block("answer")]),
    ])
    tm = MagicMock()
    tm.execute_tool.return_value = "chunk"

    assert generator.generate_response("q", tools=TOOLS, tool_manager=tm) == "answer"
    assert len(calls) == 2
    assert calls[1]["tools"] == TOOLS


def test_cap_at_two_rounds(generator):
    calls = script(generator, [
        tool_round("search_course_content", {"query": "a"}, "t1"),
        tool_round("search_course_content", {"query": "b"}, "t2"),
        make_response([text_block("final answer")]),
    ])
    tm = MagicMock()
    tm.execute_tool.return_value = "chunk"

    out = generator.generate_response("q", tools=TOOLS, tool_manager=tm)

    assert out == "final answer"
    assert tm.execute_tool.call_count == 2
    assert len(calls) == 3
    assert "tools" not in calls[2]


def test_tool_failure_stops_loop_and_flags_error(generator, mock_vector_store):
    mock_vector_store.search.side_effect = RuntimeError("db down")
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_vector_store))
    calls = script(generator, [
        tool_round("search_course_content", {"query": "q"}, "t1"),
        make_response([text_block("Search failed, sorry.")]),
    ])

    out = generator.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)

    assert out == "Search failed, sorry."
    assert len(calls) == 2
    assert "tools" not in calls[1]
    result = calls[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert "db down" in result["content"]


def test_raised_exception_from_tool_manager_is_reported(generator):
    calls = script(generator, [
        tool_round("search_course_content", {"query": "q"}, "t1"),
        make_response([text_block("sorry")]),
    ])
    tm = MagicMock()
    tm.execute_tool.side_effect = RuntimeError("boom")

    assert generator.generate_response("q", tools=TOOLS, tool_manager=tm) == "sorry"
    assert len(calls) == 2 and "tools" not in calls[1]
    assert calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_failure_in_second_round_stops_cleanly(generator):
    calls = script(generator, [
        tool_round("get_course_outline", {"course_title": "X"}, "t1"),
        tool_round("search_course_content", {"query": "q"}, "t2"),
        make_response([text_block("partial answer")]),
    ])
    tm = MagicMock()
    tm.execute_tool.side_effect = ["outline", "Tool 'search_course_content' failed: db down"]

    out = generator.generate_response("q", tools=TOOLS, tool_manager=tm)

    assert out == "partial answer"
    assert len(calls) == 3 and "tools" not in calls[2]
    first, second = calls[2]["messages"][2]["content"][0], calls[2]["messages"][4]["content"][0]
    assert "is_error" not in first
    assert second["is_error"] is True


def test_parallel_tool_use_blocks_return_in_one_message(generator):
    calls = script(generator, [
        make_response([
            tool_use_block("get_course_outline", {"course_title": "A"}, "t1"),
            tool_use_block("get_course_outline", {"course_title": "B"}, "t2"),
        ], stop_reason="tool_use"),
        make_response([text_block("compared")]),
    ])
    tm = MagicMock()
    tm.execute_tool.side_effect = ["A outline", "B outline"]

    assert generator.generate_response("compare A and B", tools=TOOLS, tool_manager=tm) == "compared"
    results = calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert [r["content"] for r in results] == ["A outline", "B outline"]


def test_empty_final_text_returns_fallback(generator):
    script(generator, [
        tool_round("search_course_content", {"query": "a"}, "t1"),
        tool_round("search_course_content", {"query": "b"}, "t2"),
        make_response([]),
    ])
    tm = MagicMock()
    tm.execute_tool.return_value = "chunk"

    out = generator.generate_response("q", tools=TOOLS, tool_manager=tm)
    assert out == AIGenerator.EMPTY_FALLBACK


def test_history_in_system_prompt_on_every_round(generator):
    calls = script(generator, [
        tool_round("search_course_content", {"query": "q"}, "t1"),
        tool_round("search_course_content", {"query": "q2"}, "t2"),
        make_response([text_block("ok")]),
    ])
    tm = MagicMock()
    tm.execute_tool.return_value = "chunk"

    generator.generate_response("q", conversation_history="User: hi", tools=TOOLS, tool_manager=tm)

    assert all("Previous conversation:\nUser: hi" in c["system"] for c in calls)


def test_system_prompt_allows_two_sequential_calls():
    assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
    assert "2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT
