import asyncio
from typing import Any, Awaitable, Callable, Optional

from backend.collaboration_state import collaboration_state


AIExecutor = Callable[[str], Awaitable[Any]]
QueueResultHandler = Callable[[dict[str, Any]], Awaitable[None]]

_worker_tasks: dict[str, asyncio.Task] = {}
_worker_lock = asyncio.Lock()


async def process_ai_prompt(
    queue_item: dict[str, Any],
    ai_executor: AIExecutor,
) -> Any:
    """Process one queued AI prompt using the injected AI executor."""

    required_fields = {"queue_id", "session_id", "user_id", "prompt"}
    missing_fields = required_fields - queue_item.keys()

    if missing_fields:
        raise ValueError(
            f"Missing queue fields: {', '.join(sorted(missing_fields))}"
        )

    session_id = str(queue_item["session_id"])
    user_id = str(queue_item["user_id"])
    prompt = str(queue_item["prompt"]).strip()

    if not prompt:
        raise ValueError("AI prompt cannot be empty.")

    await collaboration_state.set_processing(session_id, True)

    try:
        return await ai_executor(prompt)
    finally:
        await collaboration_state.remove_pending_prompt(
            session_id=session_id,
            user_id=user_id,
        )
        await collaboration_state.set_processing(session_id, False)


async def run_next_prompt(
    session_id: str,
    ai_executor: AIExecutor,
) -> Optional[dict[str, Any]]:
    """Process the next queued prompt for a session."""

    session = await collaboration_state.get_or_create_session(session_id)

    async with session.processing_lock:
        if session.queue.empty():
            return None

        _priority, queue_item = await session.queue.get()

        try:
            result = await process_ai_prompt(
                queue_item=queue_item,
                ai_executor=ai_executor,
            )

            return {
                "queue_item": queue_item,
                "result": result,
            }

        except Exception as exc:
            return {
                "queue_item": queue_item,
                "result": None,
                "error": str(exc),
            }

        finally:
            session.queue.task_done()


async def _worker_loop(
    session_id: str,
    ai_executor: AIExecutor,
    on_result: QueueResultHandler,
) -> None:
    """Continuously drain one session's queue with one AI execution at a time."""

    try:
        while True:
            result = await run_next_prompt(
                session_id=session_id,
                ai_executor=ai_executor,
            )

            if result is None:
                break

            try:
                await on_result(result)
            except Exception:
                # The queue must keep moving even if result delivery fails.
                # The integration layer is responsible for logging/broadcasting.
                continue
    finally:
        async with _worker_lock:
            current = _worker_tasks.get(session_id)
            if current is asyncio.current_task():
                _worker_tasks.pop(session_id, None)


async def ensure_session_worker(
    session_id: str,
    ai_executor: AIExecutor,
    on_result: QueueResultHandler,
) -> None:
    """Start exactly one queue worker for a session when needed."""

    async with _worker_lock:
        task = _worker_tasks.get(session_id)

        if task is not None and not task.done():
            return

        _worker_tasks[session_id] = asyncio.create_task(
            _worker_loop(
                session_id=session_id,
                ai_executor=ai_executor,
                on_result=on_result,
            )
        )


async def stop_session_worker(session_id: str) -> None:
    """Cancel the session worker if it exists."""

    async with _worker_lock:
        task = _worker_tasks.pop(session_id, None)

    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
