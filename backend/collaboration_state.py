import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectedUser:
    user_id: str
    websocket: Any
    joined_order: int


@dataclass
class CollaborationSessionState:
    session_id: str
    connected_users: dict[str, list[ConnectedUser]] = field(default_factory=dict)
    typing_users: set[str] = field(default_factory=set)
    queue: asyncio.PriorityQueue = field(default_factory=asyncio.PriorityQueue)
    next_queue_sequence: int = 0
    pending_prompts: dict[str, Any] = field(default_factory=dict)
    processing: bool = False
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    membership_order: dict[str, int] = field(default_factory=dict)
    next_join_order: int = 0


class CollaborationStateManager:
    """
    In-memory live collaboration state.

    This state is intentionally non-persistent.
    Persistent data belongs in Firestore.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, CollaborationSessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(
        self,
        session_id: str,
    ) -> CollaborationSessionState:
        async with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = CollaborationSessionState(
                    session_id=session_id
                )

            return self.sessions[session_id]

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def connect_user(
        self,
        session_id: str,
        user_id: str,
        websocket: Any,
    ) -> ConnectedUser:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            if user_id not in session.membership_order:
                session.membership_order[user_id] = session.next_join_order
                session.next_join_order += 1

            connection = ConnectedUser(
                user_id=user_id,
                websocket=websocket,
                joined_order=session.membership_order[user_id],
            )

            session.connected_users.setdefault(user_id, []).append(connection)

            return connection

    async def disconnect_user(
        self,
        session_id: str,
        user_id: str,
        websocket: Any,
    ) -> None:
        session = self.sessions.get(session_id)

        if session is None:
            return

        async with self._lock:
            connections = session.connected_users.get(user_id, [])

            session.connected_users[user_id] = [
                connection
                for connection in connections
                if connection.websocket is not websocket
            ]

            if not session.connected_users[user_id]:
                session.connected_users.pop(user_id, None)
                session.typing_users.discard(user_id)

    async def set_typing(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            session.typing_users.add(user_id)

    async def stop_typing(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        session = self.sessions.get(session_id)

        if session is None:
            return

        async with self._lock:
            session.typing_users.discard(user_id)

    async def is_prompt_pending(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            return user_id in session.pending_prompts

    async def add_pending_prompt(
        self,
        session_id: str,
        user_id: str,
        prompt: Any,
    ) -> bool:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            if user_id in session.pending_prompts:
                return False

            session.pending_prompts[user_id] = prompt
            return True

    async def remove_pending_prompt(
        self,
        session_id: str,
        user_id: str,
    ) -> Any:
        session = self.sessions.get(session_id)

        if session is None:
            return None

        async with self._lock:
            return session.pending_prompts.pop(user_id, None)

    async def enqueue_prompt(
        self,
        session_id: str,
        prompt: Any,
    ) -> None:
        session = await self.get_or_create_session(session_id)

        user_id = str(prompt.get("user_id", "")) if isinstance(prompt, dict) else ""
        submitted_ms = int(time.monotonic_ns() / 1_000_000)

        async with self._lock:
            join_order = session.membership_order.get(user_id, session.next_join_order)
            if user_id not in session.membership_order:
                session.membership_order[user_id] = join_order
                session.next_join_order += 1

            sequence = session.next_queue_sequence
            session.next_queue_sequence += 1

        priority = (submitted_ms, join_order, sequence)
        await session.queue.put((priority, prompt))

    async def get_queue_size(
        self,
        session_id: str,
    ) -> int:
        session = await self.get_or_create_session(session_id)
        return session.queue.qsize()

    async def set_processing(
        self,
        session_id: str,
        processing: bool,
    ) -> None:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            session.processing = processing

    async def is_processing(
        self,
        session_id: str,
    ) -> bool:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            return session.processing

    async def get_connected_users(
        self,
        session_id: str,
    ) -> list[ConnectedUser]:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            users: list[ConnectedUser] = []

            for connections in session.connected_users.values():
                users.extend(connections)

            return sorted(
                users,
                key=lambda connection: connection.joined_order,
            )

    async def get_typing_users(
        self,
        session_id: str,
    ) -> list[str]:
        session = await self.get_or_create_session(session_id)

        async with self._lock:
            return sorted(
                session.typing_users,
                key=lambda user_id: session.membership_order.get(
                    user_id,
                    float("inf"),
                ),
            )


collaboration_state = CollaborationStateManager()