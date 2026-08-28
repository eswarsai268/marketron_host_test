import asyncio
from typing import Any, Awaitable, Callable

from src.firestore_db import get_messages
from src import llm_agent

AIExecutor = Callable[[str], Awaitable[Any]]


def build_session_ai_executor(session_id: str) -> AIExecutor:
    """
    Build an async executor for one collaboration session.
    Routes live conversation context directly to the Gemini/Groq LLM router.
    """

    async def execute(prompt: str) -> str:
        messages = get_messages(session_id=session_id)
        
        # 1. Reconstruct real conversation history
        ai_history = [
            message
            for message in messages
            if message.get("message_type") == "ai_message"
        ]

        ai_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # 2. Enforce token/message pruning
        prune = getattr(llm_agent, "prune_chat_history", None)
        if callable(prune):
            ai_history = prune(ai_history)

        fallback_router = getattr(llm_agent, "_call_llm_with_fallback", None)
        if not callable(fallback_router):
            raise RuntimeError(
                "Existing llm_agent.py does not expose the expected LLM router."
            )

        # 3. Offload blocking network calls to a thread pool
        return await asyncio.to_thread(
            fallback_router,
            ai_history,
        )

    return execute