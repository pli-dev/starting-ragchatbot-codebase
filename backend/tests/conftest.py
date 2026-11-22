"""Shared fixtures for the backend test suite."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore

COURSE_TITLE = "Building Towards Computer Use with Anthropic"
COURSE_LINK = "https://example.com/course"


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_course() -> Course:
    return Course(
        title=COURSE_TITLE,
        course_link=COURSE_LINK,
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/l0"),
            Lesson(lesson_number=1, title="Tool Use Basics", lesson_link="https://example.com/l1"),
        ],
    )


@pytest.fixture
def sample_chunks() -> list[CourseChunk]:
    return [
        CourseChunk(
            content="Lesson 0 content: Welcome to the course. We introduce computer use and agents.",
            course_title=COURSE_TITLE, lesson_number=0, chunk_index=0,
        ),
        CourseChunk(
            content="Lesson 1 content: Tool use lets Claude call functions you define with a JSON schema.",
            course_title=COURSE_TITLE, lesson_number=1, chunk_index=1,
        ),
        CourseChunk(
            content="The model returns a tool_use block and your code returns a tool_result block.",
            course_title=COURSE_TITLE, lesson_number=1, chunk_index=2,
        ),
    ]


# --------------------------------------------------------------------------- #
# Vector stores
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_vector_store(tmp_path, sample_course, sample_chunks):
    """Factory: real VectorStore on a temp Chroma dir, loaded with the sample course."""

    def _make(max_results: int) -> VectorStore:
        store = VectorStore(str(tmp_path / f"chroma_{max_results}"), Config.EMBEDDING_MODEL, max_results)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_chunks)
        return store

    return _make


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = COURSE_LINK
    return store


@pytest.fixture
def test_config(tmp_path):
    """A Config copy pointing at a temp Chroma dir, MAX_RESULTS defaulting to a sane 5."""
    return SimpleNamespace(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL=Config.ANTHROPIC_MODEL,
        EMBEDDING_MODEL=Config.EMBEDDING_MODEL,
        CHUNK_SIZE=Config.CHUNK_SIZE,
        CHUNK_OVERLAP=Config.CHUNK_OVERLAP,
        MAX_RESULTS=5,
        MAX_HISTORY=Config.MAX_HISTORY,
        CHROMA_PATH=str(tmp_path / "rag_chroma"),
    )


# --------------------------------------------------------------------------- #
# Fake Anthropic responses
# --------------------------------------------------------------------------- #
def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, input: dict, id: str = "toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def make_response(content: list, stop_reason: str = "end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)
