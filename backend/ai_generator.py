import anthropic
from typing import List, Optional

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use **only** for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about a course's outline, structure, syllabus, or which lessons it contains
  - When answering, always include the course title, the course link, and the number and title of **every** lesson (complete list, no truncation)
- **Up to 2 sequential tool calls per query**, each in its own step so you can reason about earlier results
  - Make a second call only when it depends on the first result (e.g. get a course outline, then search using a lesson title)
  - Never repeat a call that already failed or returned nothing; after your tool use, answer directly
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    MAX_TOOL_ROUNDS = 2
    EMPTY_FALLBACK = "I wasn't able to complete that request."

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        messages = [{"role": "user", "content": query}]
        base = {**self.base_params, "system": system_content}

        # Up to MAX_TOOL_ROUNDS rounds, each a separate API request with tools attached
        for _ in range(self.MAX_TOOL_ROUNDS):
            request = {**base, "messages": messages}
            if tools:
                request["tools"] = tools
                request["tool_choice"] = {"type": "auto"}
            response = self.client.messages.create(**request)

            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            messages.append({"role": "assistant", "content": response.content})
            results, failed = self._run_tools(response, tool_manager)
            messages.append({"role": "user", "content": results})
            if failed:
                break

        # Rounds exhausted or a tool failed: force a text answer by omitting tools
        final = self.client.messages.create(**base, messages=messages)
        return self._extract_text(final) or self.EMPTY_FALLBACK

    @staticmethod
    def _extract_text(response) -> str:
        """Join all text blocks, skipping thinking and other non-text blocks."""
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    @staticmethod
    def _is_error_result(result) -> bool:
        """ToolManager reports failures as strings rather than raising."""
        return isinstance(result, str) and result.startswith("Tool '") and (
            "' failed:" in result or result.endswith("' not found")
        )

    def _run_tools(self, response, tool_manager):
        """Execute every tool_use block; return (tool_result blocks, any_failed)."""
        results, failed = [], False
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = tool_manager.execute_tool(block.name, **block.input)
                is_error = self._is_error_result(output)
            except Exception as e:
                output, is_error = f"Tool '{block.name}' failed: {e}", True
            result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                result["is_error"] = True
                failed = True
            results.append(result)
        return results, failed
