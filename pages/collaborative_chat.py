import os
import queue
import threading
import time
from typing import Any
from urllib.parse import urlparse
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from websockets.sync.client import connect

load_dotenv()

COLLAB_SERVER_URL = os.getenv(
    "COLLAB_SERVER_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

def _ws_url(session_id: str) -> str:
    parsed = urlparse(COLLAB_SERVER_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws/collab/{session_id}"

def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

class CollabWebSocketClient:
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self._token = ""
        self._token_lock = threading.Lock()
        self._send_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None

    def update_token(self, token: str) -> None:
        with self._token_lock:
            self._token = token

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"collab-ws-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def send(self, event: dict[str, Any]) -> None:
        self._send_queue.put(event)

    def drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._inbox.get_nowait())
            except queue.Empty:
                return events

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def close(self) -> None:
        self._stop.set()
        self._connected.clear()

    def _get_token(self) -> str:
        with self._token_lock:
            return self._token

    def _run(self) -> None:
        while not self._stop.is_set():
            token = self._get_token()
            if not token:
                time.sleep(0.25)
                continue
            try:
                with connect(
                    _ws_url(self.session_id),
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    self._connected.set()
                    websocket.send(json.dumps({"event": "authenticate", "token": token}))
                    websocket.send(json.dumps({"event": "join"}))

                    while not self._stop.is_set():
                        while True:
                            try:
                                outgoing = self._send_queue.get_nowait()
                            except queue.Empty:
                                break
                            websocket.send(json.dumps(outgoing))

                        try:
                            incoming = websocket.recv(timeout=0.2)
                            if isinstance(incoming, str):
                                try:
                                    self._inbox.put(json.loads(incoming))
                                except json.JSONDecodeError:
                                    continue
                        except TimeoutError:
                            continue
            except Exception as exc:
                self._connected.clear()
                if not self._stop.is_set():
                    self._inbox.put({"event": "connection_error", "message": str(exc)})
                    time.sleep(1.0)
            finally:
                self._connected.clear()

def get_ws_client(session_id: str, user_id: str) -> CollabWebSocketClient:
    key = f"collab_ws_client_{session_id}_{user_id}"
    if key not in st.session_state:
        st.session_state[key] = CollabWebSocketClient(session_id, user_id)
    return st.session_state[key]

def get_session_messages(session_id: str, token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{COLLAB_SERVER_URL}/sessions/{session_id}/messages",
        headers=_auth_headers(token),
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("messages", [])

def get_session_members(session_id: str, token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{COLLAB_SERVER_URL}/sessions/{session_id}/members",
        headers=_auth_headers(token),
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("members", [])

def leave_session(session_id: str, token: str) -> None:
    response = requests.post(
        f"{COLLAB_SERVER_URL}/sessions/{session_id}/leave",
        headers=_auth_headers(token),
        timeout=10,
    )
    response.raise_for_status()

def end_session(session_id: str, token: str) -> None:
    response = requests.post(
        f"{COLLAB_SERVER_URL}/sessions/{session_id}/end",
        headers=_auth_headers(token),
        timeout=15,
    )
    response.raise_for_status()

# ====================================================================
# THE FIX: Protect the state with an immunized dictionary
# ====================================================================
if "collab_status" not in st.session_state:
    st.session_state.collab_status = {"locked": False}

def append_message(message: dict[str, Any]) -> None:
    message_id = message.get("message_id")
    
    if "collab_messages" not in st.session_state:
        st.session_state.collab_messages = []
        
    messages = st.session_state.collab_messages
    
    if message_id and any(existing.get("message_id") == message_id for existing in messages):
        return
        
    # IN-PLACE MUTATION: Streamlit cannot roll this back!
    messages.append(message)

def handle_ws_events(client: CollabWebSocketClient) -> bool:
    needs_rerun = False
    for event in client.drain_events():
        event_type = event.get("event")

        if event_type == "ai_message" or event_type == "team_message":
            message = event.get("message")
            if isinstance(message, dict):
                append_message(message)
                needs_rerun = True

        elif event_type == "ai_completed":
            message = event.get("message")
            if isinstance(message, dict):
                append_message(message)
            st.session_state.collab_queue_size = 0
            st.session_state.collab_processing = False
            st.session_state.collab_status["locked"] = False
            needs_rerun = True

        elif event_type == "ai_failed":
            st.session_state.collab_processing = False
            st.session_state.collab_status["locked"] = False
            needs_rerun = True

        elif event_type == "prompt_queued":
            st.session_state.collab_queue_size = event.get("position", st.session_state.get("collab_queue_size", 0))

        elif event_type == "queue_updated":
            st.session_state.collab_queue_size = event.get("queue_size", 0)
            st.session_state.collab_processing = bool(event.get("processing", False))

        elif event_type == "input_locked":
            if event.get("user_id") == st.session_state.get("collab_user_id"):
                st.session_state.collab_status["locked"] = True
                needs_rerun = True

        elif event_type == "unlock_input":
            if event.get("user_id") == st.session_state.get("collab_user_id"):
                st.session_state.collab_status["locked"] = False
                needs_rerun = True

        elif event_type == "user_joined" or event_type == "user_left":
            st.session_state.collab_members_refresh = True
            needs_rerun = True

        elif event_type == "error":
            st.session_state.collab_last_error = event.get("message", "Collaboration error.")

        elif event_type == "connection_error":
            st.session_state.collab_last_error = "Collaboration connection lost. Reconnecting..."

    return needs_rerun

def initialize_page(session_id: str, token: str, user_id: str) -> CollabWebSocketClient:
    if st.session_state.get("collab_loaded_session_id") != session_id:
        st.session_state.collab_loaded_session_id = session_id
        st.session_state.collab_messages = get_session_messages(session_id, token)
        st.session_state.collab_members = get_session_members(session_id, token)
        st.session_state.collab_queue_size = 0
        st.session_state.collab_processing = False
        st.session_state.collab_status["locked"] = False
        st.session_state.collab_last_error = None
        st.session_state.collab_members_refresh = False

    client = get_ws_client(session_id, user_id)
    client.update_token(token)
    client.start()
    client._connected.wait(timeout=3)

    if client.is_connected():
        st.session_state.collab_last_error = None

    handle_ws_events(client)

    if st.session_state.get("collab_members_refresh"):
        try:
            st.session_state.collab_members = get_session_members(session_id, token)
            st.session_state.collab_members_refresh = False
        except requests.RequestException:
            pass

    return client

session = st.session_state.get("collaboration_session")
user_id = st.session_state.get("collab_user_id")
token = st.session_state.get("collab_token")

if not session:
    collab_session_id = st.query_params.get("collab_session")
    if collab_session_id and token:
        try:
            response = requests.get(
                f"{COLLAB_SERVER_URL}/sessions/{collab_session_id}",
                headers=_auth_headers(token),
                timeout=10,
            )
            response.raise_for_status()
            session = response.json()
            st.session_state.collaboration_session = session
            st.session_state.collaboration_available = True
        except requests.RequestException:
            session = None

if not session or not user_id or not token:
    st.title("Collaborative Chat")
    st.info("Start or join a collaboration session from the Host page first.")
    st.stop()

session_id = session.get("session_id")
if not session_id:
    st.error("The collaboration session ID is unavailable.")
    st.stop()

# Reconnected exactly where it belongs
client = initialize_page(session_id, token, user_id)

st.markdown(
    """
    <div class="host-section-title">Collaborative Chat</div>
    <p class="host-section-description">
        Work together in the same Marketron AI conversation.
    </p>
    """,
    unsafe_allow_html=True,
)

header_left, header_mid, header_right = st.columns([2.8, 1.2, 1.2])

with header_left:
    st.markdown(f"### {session.get('title') or 'Marketron Collaboration'}")

with header_mid:
    connected_label = "Connected" if client.is_connected() else "Reconnecting"
    st.caption(connected_label)

with header_right:
    member_record = next(
        (member for member in st.session_state.get("collab_members", []) if member.get("user_id") == user_id),
        None,
    )
    is_host = bool(member_record and member_record.get("role") == "host")

    if is_host:
        if st.button("End Collaboration", key="end-collaboration-btn", width="stretch"):
            try:
                end_session(session_id, token)
                client.close()
                st.session_state.pop("collaboration_session", None)
                st.session_state.pop("collab_loaded_session_id", None)
                st.session_state.pop("collab_messages", None)
                st.success("Collaboration ended.")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Unable to end collaboration: {exc}")
    else:
        if st.button("Leave Collaboration", key="leave-collaboration-btn", width="stretch"):
            try:
                leave_session(session_id, token)
                client.close()
                st.session_state.pop("collaboration_session", None)
                st.session_state.pop("collab_loaded_session_id", None)
                st.session_state.pop("collab_messages", None)
                st.success("You left the collaboration.")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Unable to leave collaboration: {exc}")

@st.fragment(run_every="1s")
def live_collaboration():
    needs_rerun = handle_ws_events(client)

    if st.session_state.get("collab_last_error"):
        st.warning(st.session_state.collab_last_error)

    members_col, queue_col = st.columns([2.5, 1])
    member_names = {}
    
    with members_col:
        st.markdown("#### Members")
        active_members = [
            member for member in st.session_state.get("collab_members", [])
            if member.get("status") == "active"
        ]

        if active_members:
            member_labels = []
            for member in active_members:
                user_ident = member.get("user_id", "User")
                label = user_ident if user_ident != user_id else "You"
                
                if len(label) > 18 and label != "You":
                    label = label[:8] + "…"
                    
                if member.get("role") == "host":
                    label += " (Host)"
                    
                member_labels.append(label)
                member_names[user_ident] = label 
                
            st.write(" · ".join(member_labels))
        else:
            st.caption("No active members found.")

    with queue_col:
        if st.session_state.get("collab_processing"):
            st.info("Marketron AI is processing…")
        elif st.session_state.get("collab_queue_size"):
            st.info(f"Queue position: {st.session_state.collab_queue_size}")
        else:
            st.caption("AI queue is clear.")

    st.markdown("#### Conversation")

    messages = st.session_state.get("collab_messages", [])

    if not messages:
        st.caption("No messages yet.")
    else:
        for message in messages:
            if message.get("hidden", False):
                continue
                
            role = message.get("role")
            content = message.get("content", "")
            sender = message.get("user_id", "")

            if role == "assistant":
                label = "Marketron AI"
            elif sender == user_id:
                label = "You"
            else:
                label = member_names.get(sender, sender[:12] + "…")

            with st.chat_message("assistant" if role == "assistant" else "user"):
                st.markdown(f"**{label}**")
                st.markdown(content)

    if needs_rerun:
        st.rerun()

live_collaboration()

# Check the protected dictionary for the lock state
is_locked = st.session_state.collab_status.get("locked", False)

prompt = st.chat_input(
    placeholder="Ask Marketron AI..." if not is_locked else "Your prompt is waiting in the queue...",
    disabled=is_locked
)

if prompt:
    prompt_text = prompt.strip()
    
    if prompt_text:
        client.send({
            "event": "ai_prompt",
            "prompt": prompt_text
        })
        # Set the protected dictionary state directly
        st.session_state.collab_status["locked"] = True
        
    st.rerun()