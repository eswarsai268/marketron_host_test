import asyncio

import pytest

import backend.collaboration_state as state_module
from backend.ai_executor import run_next_prompt, ensure_session_worker
from backend.collaboration_state import collaboration_state


@pytest.mark.asyncio
async def test_create_session():
    session_id = "test-session-create"
    session = await collaboration_state.get_or_create_session(session_id)
    assert session is not None
    assert session.session_id == session_id


@pytest.mark.asyncio
async def test_connect_and_disconnect_user():
    session_id = "test-session-users"
    user_id = "user-1"

    await collaboration_state.get_or_create_session(session_id)

    connection = await collaboration_state.connect_user(
        session_id=session_id,
        user_id=user_id,
        websocket=None,
    )

    assert connection.user_id == user_id
    users = await collaboration_state.get_connected_users(session_id)
    assert any(user.user_id == user_id for user in users)

    await collaboration_state.disconnect_user(
        session_id=session_id,
        user_id=user_id,
        websocket=None,
    )

    users = await collaboration_state.get_connected_users(session_id)
    assert not any(user.user_id == user_id for user in users)


@pytest.mark.asyncio
async def test_typing_state():
    session_id = "test-session-typing"
    user_id = "user-typing"

    await collaboration_state.set_typing(session_id, user_id)
    assert user_id in await collaboration_state.get_typing_users(session_id)

    await collaboration_state.stop_typing(session_id, user_id)
    assert user_id not in await collaboration_state.get_typing_users(session_id)


@pytest.mark.asyncio
async def test_one_pending_prompt_per_user():
    session_id = "test-session-pending"
    user_id = "user-prompt"

    assert await collaboration_state.add_pending_prompt(
        session_id, user_id, "Prompt 1"
    ) is True

    assert await collaboration_state.add_pending_prompt(
        session_id, user_id, "Prompt 2"
    ) is False

    assert await collaboration_state.remove_pending_prompt(
        session_id, user_id
    ) == "Prompt 1"


@pytest.mark.asyncio
async def test_fifo_order_for_distinct_submission_times(monkeypatch):
    session_id = "test-session-queue"
    times = iter([1_000_000_000, 1_010_000_000])
    monkeypatch.setattr(
        state_module.time,
        "monotonic_ns",
        lambda: next(times),
    )

    await collaboration_state.connect_user(session_id, "user-a", None)
    await collaboration_state.connect_user(session_id, "user-b", None)

    prompt_a = {"queue_id": "queue-a", "user_id": "user-a", "prompt": "A"}
    prompt_b = {"queue_id": "queue-b", "user_id": "user-b", "prompt": "B"}

    await collaboration_state.enqueue_prompt(session_id, prompt_a)
    await collaboration_state.enqueue_prompt(session_id, prompt_b)

    session = await collaboration_state.get_or_create_session(session_id)
    _priority_a, first = await session.queue.get()
    _priority_b, second = await session.queue.get()

    assert first == prompt_a
    assert second == prompt_b


@pytest.mark.asyncio
async def test_join_order_breaks_same_millisecond_tie(monkeypatch):
    session_id = "test-session-tie"
    monkeypatch.setattr(state_module.time, "monotonic_ns", lambda: 1_000_000_000)

    await collaboration_state.connect_user(session_id, "user-a", None)
    await collaboration_state.connect_user(session_id, "user-b", None)

    prompt_b = {"queue_id": "queue-b", "user_id": "user-b", "prompt": "B"}
    prompt_a = {"queue_id": "queue-a", "user_id": "user-a", "prompt": "A"}

    await collaboration_state.enqueue_prompt(session_id, prompt_b)
    await collaboration_state.enqueue_prompt(session_id, prompt_a)

    session = await collaboration_state.get_or_create_session(session_id)
    _priority_first, first = await session.queue.get()
    _priority_second, second = await session.queue.get()

    assert first == prompt_a
    assert second == prompt_b


@pytest.mark.asyncio
async def test_only_one_ai_execution_at_a_time():
    session_id = "test-session-concurrency"

    await collaboration_state.add_pending_prompt(session_id, "user-a", "A")
    await collaboration_state.add_pending_prompt(session_id, "user-b", "B")

    await collaboration_state.enqueue_prompt(
        session_id,
        {"queue_id": "a", "session_id": session_id, "user_id": "user-a", "prompt": "A"},
    )
    await collaboration_state.enqueue_prompt(
        session_id,
        {"queue_id": "b", "session_id": session_id, "user_id": "user-b", "prompt": "B"},
    )

    active = 0
    max_active = 0
    execution_order = []

    async def fake_ai(prompt):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        execution_order.append(prompt)
        await asyncio.sleep(0.03)
        active -= 1
        return f"Result {prompt}"

    first_task = asyncio.create_task(run_next_prompt(session_id, fake_ai))
    second_task = asyncio.create_task(run_next_prompt(session_id, fake_ai))

    results = await asyncio.gather(first_task, second_task)

    assert max_active == 1
    assert execution_order == ["A", "B"]
    assert [result["result"] for result in results] == ["Result A", "Result B"]


@pytest.mark.asyncio
async def test_session_worker_drains_queue_in_order():
    session_id = "test-session-worker"

    await collaboration_state.add_pending_prompt(session_id, "user-a", "A")
    await collaboration_state.add_pending_prompt(session_id, "user-b", "B")

    await collaboration_state.enqueue_prompt(
        session_id,
        {"queue_id": "a", "session_id": session_id, "user_id": "user-a", "prompt": "A"},
    )
    await collaboration_state.enqueue_prompt(
        session_id,
        {"queue_id": "b", "session_id": session_id, "user_id": "user-b", "prompt": "B"},
    )

    completed = []

    async def fake_ai(prompt):
        await asyncio.sleep(0.01)
        return f"Result {prompt}"

    async def on_result(result):
        completed.append(result["result"])

    await ensure_session_worker(session_id, fake_ai, on_result)

    for _ in range(100):
        if not await collaboration_state.is_processing(session_id):
            session = await collaboration_state.get_or_create_session(session_id)
            if session.queue.empty() and not session.pending_prompts:
                break
        await asyncio.sleep(0.01)

    assert completed == ["Result A", "Result B"]
