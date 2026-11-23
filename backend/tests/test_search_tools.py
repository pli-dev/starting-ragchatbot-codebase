"""Tests for CourseSearchTool.execute (and the ToolManager plumbing around it)."""
from config import config
from conftest import COURSE_LINK, COURSE_TITLE
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# --------------------------------------------------------------------------- #
# Unit tests (mocked VectorStore)
# --------------------------------------------------------------------------- #
class TestExecuteWithMockStore:
    def test_formats_results_with_headers(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["chunk A", "chunk B"],
            metadata=[
                {"course_title": COURSE_TITLE, "lesson_number": 1},
                {"course_title": COURSE_TITLE, "lesson_number": None},
            ],
            distances=[0.1, 0.2],
        )
        out = CourseSearchTool(mock_vector_store).execute("tool use")

        assert f"[{COURSE_TITLE} - Lesson 1]\nchunk A" in out
        assert f"[{COURSE_TITLE}]\nchunk B" in out

    def test_sources_deduped_with_lesson_link_and_course_fallback(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["a", "b", "c"],
            metadata=[
                {"course_title": COURSE_TITLE, "lesson_number": 1},
                {"course_title": COURSE_TITLE, "lesson_number": 1},  # duplicate
                {"course_title": COURSE_TITLE, "lesson_number": 2},  # no lesson link
            ],
            distances=[0.1, 0.2, 0.3],
        )
        mock_vector_store.get_lesson_link.side_effect = (
            lambda title, n: "https://example.com/l1" if n == 1 else None
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("q")

        assert [s["text"] for s in tool.last_sources] == [
            f"{COURSE_TITLE} - Lesson 1",
            f"{COURSE_TITLE} - Lesson 2",
        ]
        assert tool.last_sources[0]["link"] == "https://example.com/l1"
        assert tool.last_sources[1]["link"] == COURSE_LINK  # fell back to course link

    def test_passes_filters_to_store(self, mock_vector_store):
        CourseSearchTool(mock_vector_store).execute("q", course_name="MCP", lesson_number=3)
        mock_vector_store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=3)

    def test_returns_store_error_verbatim(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults.empty("Search error: boom")
        assert CourseSearchTool(mock_vector_store).execute("q") == "Search error: boom"

    def test_empty_results_message_includes_filters(self, mock_vector_store):
        out = CourseSearchTool(mock_vector_store).execute("q", course_name="MCP", lesson_number=2)
        assert out == "No relevant content found in course 'MCP' in lesson 2."

    def test_empty_results_message_includes_lesson_zero(self, mock_vector_store):
        out = CourseSearchTool(mock_vector_store).execute("q", lesson_number=0)
        assert out == "No relevant content found in lesson 0."


class TestToolManager:
    def test_unknown_tool(self):
        assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"

    def test_dispatches_to_search_tool(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.execute_tool("search_course_content", query="q", course_name="MCP")
        mock_vector_store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=None)

    def test_tool_exception_returned_as_error_string(self, mock_vector_store):
        mock_vector_store.search.side_effect = RuntimeError("db down")
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        out = tm.execute_tool("search_course_content", query="q")
        assert "db down" in out


# --------------------------------------------------------------------------- #
# Integration tests (real ChromaDB in a temp dir)
# --------------------------------------------------------------------------- #
class TestExecuteWithRealStore:
    def test_returns_content_with_sane_limit(self, make_vector_store):
        tool = CourseSearchTool(make_vector_store(max_results=5))
        out = tool.execute("How does tool use work?")

        assert not out.startswith("Search error")
        assert "tool_use" in out or "Tool use" in out
        assert tool.last_sources

    def test_lesson_filter(self, make_vector_store):
        tool = CourseSearchTool(make_vector_store(max_results=5))
        out = tool.execute("welcome", course_name="Computer Use", lesson_number=0)
        assert "Lesson 0" in out
        assert "Lesson 1]" not in out

    def test_returns_content_with_configured_max_results(self, make_vector_store):
        """Uses the app's real config.MAX_RESULTS -- this is what production does."""
        tool = CourseSearchTool(make_vector_store(max_results=config.MAX_RESULTS))
        out = tool.execute("How does tool use work?")
        assert not out.startswith("Search error"), out


def test_config_max_results_positive():
    # Copy to a local first so pytest's assertion rewrite never prints the Config repr (contains the API key)
    max_results = config.MAX_RESULTS
    assert max_results > 0, f"MAX_RESULTS={max_results}; Chroma needs n_results >= 1"


def test_vector_store_rejects_non_positive_max_results(tmp_path):
    import pytest
    from vector_store import VectorStore
    with pytest.raises(ValueError, match="max_results must be >= 1"):
        VectorStore(str(tmp_path / "c"), config.EMBEDDING_MODEL, max_results=0)
