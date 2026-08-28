import os
import requests
import streamlit as st
from src.auth import make_circular_profile_image
from dotenv import load_dotenv
from backend.collab_auth import create_collaboration_token

load_dotenv()

collab_user_id = st.session_state.get(
    "collab_user_id"
)

if not collab_user_id:
    st.error(
        "Authenticated collaboration identity is unavailable."
    )
    st.stop()

collab_token = create_collaboration_token(
    collab_user_id
)

st.session_state.collab_token = collab_token

COLLAB_SERVER_URL = os.getenv(
    "COLLAB_SERVER_URL",
    "http://127.0.0.1:8001",
)


def create_collaboration_session():
    headers = collab_headers()

    if segment_ready and batch_ready:
        initial_history = segment_history
        source = "Segment Analyzer"
    elif segment_ready:
        initial_history = segment_history
        source = "Segment Analyzer"
    elif batch_ready:
        initial_history = batch_history
        source = "Batch Segmentation"
    else:
        return None

    initial_messages = []

    for message in initial_history:
        role = message.get("role")
        content = message.get("content", "")
        hidden = message.get("hidden", False)

        # THE FIX: If the message is a hidden background prompt, skip it entirely!
        if hidden:
            continue

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        initial_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    response = requests.post(
        f"{COLLAB_SERVER_URL}/sessions",
        headers=headers,
        json={
            "title": (
                f"Marketron Collaboration — {source}"
            ),
            "initial_messages": initial_messages,
        },
        timeout=15,
    )

    response.raise_for_status()

    return response.json()
# ==========================================
# HELPERS
# ==========================================

def has_ai_conversation(messages):
    return any(
        message.get("role") == "assistant"
        and not message.get("hidden", False)
        and str(message.get("content", "")).strip()
        for message in messages
    )


# ==========================================
# CONVERSATION STATE
# ==========================================

segment_history = st.session_state.get(
    "chat_history",
    []
)

batch_history = st.session_state.get(
    "batch_chat_history",
    []
)

segment_ready = has_ai_conversation(
    segment_history
)

batch_ready = has_ai_conversation(
    batch_history
)

# STRICT PRODUCTION RULE: Must have an active conversation from a generator
has_conversation = segment_ready or batch_ready

collaboration_session = st.session_state.get(
    "collaboration_session"
)

session_created = collaboration_session is not None

if segment_ready and batch_ready:
    active_source = "Segment Analyzer / Batch Segmentation"
elif segment_ready:
    active_source = "Segment Analyzer"
elif batch_ready:
    active_source = "Batch Segmentation"
else:
    active_source = None

def collab_headers():
    token = st.session_state.get(
        "collab_token"
    )

    if not token:
        raise RuntimeError(
            "Collaboration authentication token is unavailable."
        )

    return {
        "Authorization": f"Bearer {token}",
    }


def get_invitations():
    headers = collab_headers()

    received_response = requests.get(
        f"{COLLAB_SERVER_URL}/invitations/received",
        headers=headers,
        timeout=10,
    )

    received_response.raise_for_status()

    sent_response = requests.get(
        f"{COLLAB_SERVER_URL}/invitations/sent",
        headers=headers,
        timeout=10,
    )

    sent_response.raise_for_status()

    return (
        received_response.json().get(
            "invitations",
            []
        ),
        sent_response.json().get(
            "invitations",
            []
        ),
    )


def send_invitation(
    session_id,
    recipient_user_id,
):
    response = requests.post(
        f"{COLLAB_SERVER_URL}/sessions/"
        f"{session_id}/invitations",
        headers=collab_headers(),
        json={
            "recipient_user_id": recipient_user_id,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def respond_to_invitation(
    invitation_id,
    status,
):
    response = requests.post(
        f"{COLLAB_SERVER_URL}/invitations/"
        f"{invitation_id}/respond",
        headers=collab_headers(),
        json={
            "status": status,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()

@st.dialog("Invite Teammates")
def invite_teammates_dialog():

    st.caption(
        "Search people who have already signed in to Marketron."
    )

    search_text = st.text_input(
        "Search by name or email",
        placeholder="e.g. ananya or ananya@gmail.com",
        key="invite_search_text",
    )

    if st.button(
        "Search",
        key="search_users_btn",
        width="stretch",
    ):

        if not search_text.strip():
            st.warning(
                "Enter a name or email to search."
            )
            return

        try:
            response = requests.get(
                f"{COLLAB_SERVER_URL}/users/search",
                headers=collab_headers(),
                params={
                    "q": search_text.strip(),
                    "limit": 10,
                },
                timeout=10,
            )

            response.raise_for_status()

            st.session_state.invite_search_results = (
                response.json().get("users", [])
            )

        except requests.RequestException as exc:
            st.error(
                f"Unable to search users: {exc}"
            )
            return

    results = st.session_state.get(
        "invite_search_results",
        [],
    )

    if not results:
        return

    st.markdown("### People")

    session = st.session_state.get(
        "collaboration_session"
    )

    if not session:
        st.warning(
            "Create a collaboration session first."
        )
        return
    invite_success_message = None

    for user in results:

        result_col1, result_col2 = st.columns(
            [3, 1]
        )

        with result_col1:

            display_name = (
                user.get("display_name")
                or "Marketron User"
            )

            email = user.get(
                "email",
                "",
            )

            st.markdown(
                f"""
<div class="invite-search-result">
    <div class="invite-search-name">
        {display_name}
    </div>
    <div class="invite-search-email">
        {email}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with result_col2:

            if st.button(
                "Invite",
                key=f"invite_user_{user['user_id']}",
                width="stretch",
            ):

                try:

                    send_invitation(
                        session_id=session[
                            "session_id"
                        ],
                        recipient_user_id=user[
                            "user_id"
                        ],
                    )

                    invite_success_message = (
                        f"Invitation sent to {display_name}."
                    )

                    st.session_state.pop(
                        "invite_search_results",
                        None,
                    )

                except requests.HTTPError as exc:

                    try:
                        detail = exc.response.json().get(
                            "detail",
                            "Unable to send invitation.",
                        )
                    except Exception:
                        detail = (
                            "Unable to send invitation."
                        )

                    st.error(detail)

                except requests.RequestException as exc:
                    st.error(
                        f"Unable to send invitation: {exc}"
                    )

    if invite_success_message:
        st.success(invite_success_message)

# ==========================================
# HOST HERO
# ==========================================

with st.container(key="host_hero"):

    st.markdown(
        """
<div class="host-eyebrow">
    MARKETRON COLLABORATION
</div>

<h1 class="host-title">
    Turn Your AI Strategy<br>
    Into a Team Workspace
</h1>

<p class="host-description">
    Take an existing Marketron AI conversation, invite your
    teammates, and continue refining the strategy together
    in a shared collaborative workspace.
</p>
""",
        unsafe_allow_html=True
    )

    if has_conversation and not session_created:

        host_left, host_center, host_right = st.columns([1.5, 1, 1.5])

        with host_center:
            if st.button(
                "Host Conversation",
                key="host_conversation_btn",
                width="stretch"
            ):
                try:
                    session = create_collaboration_session()

                    if session:
                        st.session_state.collaboration_session = session
                        st.session_state.collaboration_available = True

                        st.query_params["collab_session"] = session["session_id"]

                        st.success(
                            "Collaboration session created."
                        )

                        st.rerun()

                except requests.RequestException as exc:
                    st.error(
                        f"Unable to connect to the collaboration server: {exc}"
                    )

                except Exception as exc:
                    st.error(
                        f"Unable to start collaboration: {exc}"
                    )
    elif session_created:

        host_left, host_center, host_right = st.columns(
            [1.5, 1, 1.5]
        )

        with host_center:

            st.html(
                """
                <div class="host-ready-message">
                    Collaboration session is ready.
                </div>
                """
            )
    else:

        host_left, host_center, host_right = st.columns([1.5, 1, 1.5])

        with host_center:
            st.button(
                "Host Conversation",
                key="host_conversation_btn",
                width="stretch",
                disabled=True
            )


# ==========================================
# INVITATIONS
# ==========================================

collaboration_session = st.session_state.get(
    "collaboration_session"
)

st.markdown("<br>", unsafe_allow_html=True)

invite_left, invite_center, invite_right = st.columns(
    [1.6, 1, 1.6]
)

with invite_center:

    if collaboration_session:

        if st.button(
            "Invite People",
            key="invite_people_btn",
            width="stretch",
        ):
            invite_teammates_dialog()

    else:

        st.button(
            "Invite People",
            key="invite_people_btn",
            width="stretch",
            disabled=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

invite_header_col, invite_refresh_col = st.columns(
    [8, 1]
)

with invite_header_col:
    st.markdown(
        """
<div class="host-section-title">
    Invitations
</div>
<p class="host-section-description">
    Manage collaboration invitations you have sent or received.
</p>
""",
        unsafe_allow_html=True,
    )

with invite_refresh_col:
    if st.button(
        "↻",
        key="refresh-invitations-btn",
        help="Refresh invitations",
    ):
        st.rerun()

try:

    real_received_invitations, real_sent_invitations = (
        get_invitations()
    )

except requests.RequestException as exc:

    real_received_invitations = []
    real_sent_invitations = []

    st.error(
        f"Unable to load invitations: {exc}"
    )

sent_col, received_col = st.columns(2)


with sent_col:
    with st.container(border=True):

        st.markdown("### Sent")

        st.caption(
            "People you invited to collaborate."
        )

        for invitation in real_sent_invitations:
            recipient = invitation.get(
                "recipient"
            ) or {}

            name = (
                recipient.get("display_name")
                or "Marketron User"
            )

            email = recipient.get(
                "email",
                "",
            )

            profile_picture = recipient.get(
                "profile_picture",
                "",
            )

            initial = (
                name.strip()[:1].upper()
                if name.strip()
                else "?"
            )

            profile_image = ""

            if profile_picture:
                try:
                    profile_image = make_circular_profile_image(
                        profile_picture
                    )
                except Exception:
                    profile_image = ""

            status = invitation.get(
                "status",
                "pending",
            ).capitalize()
            st.html(
                f"""
            <div class="invitation-card">
                <div class="invitation-avatar">
                    {
                        f'<img src="{profile_image}" alt="Profile">'
                        if profile_image
                        else initial
                    }
                </div>

                <div class="invitation-info">
                    <div class="invitation-name">
                        {name}
                    </div>

                    <div class="invitation-email">
                        {email}
                    </div>

                    <div class="invitation-status">
                        {status}
                    </div>
                </div>
            </div>
            """
            )


with received_col:
    with st.container(border=True):

        st.markdown("### Received")

        st.caption(
            "Collaboration requests sent to you."
        )

        for index, invitation in enumerate(real_received_invitations):
            sender = invitation.get(
                "sender"
            ) or {}

            name = (
                sender.get("display_name")
                or "Marketron User"
            )

            email = sender.get(
                "email",
                "",
            )

            profile_picture = sender.get(
                "profile_picture",
                "",
            )

            initial = (
                name.strip()[:1].upper()
                if name.strip()
                else "?"
            )

            profile_image = ""

            if profile_picture:
                try:
                    profile_image = make_circular_profile_image(
                        profile_picture
                    )
                except Exception:
                    profile_image = ""

            status = invitation.get(
                "status",
                "pending",
            ).capitalize()

            st.html(
                f"""
            <div class="invitation-card">
                <div class="invitation-avatar">
                    {
                        f'<img src="{profile_image}" alt="Profile">'
                        if profile_image
                        else initial
                    }
                </div>

                <div class="invitation-info">
                    <div class="invitation-name">
                        {name}
                    </div>

                    <div class="invitation-email">
                        {email}
                    </div>

                    <div class="invitation-status">
                        {status}
                    </div>
                </div>
            </div>
            """
            )
            if invitation.get("status") == "pending":
                accept_col, decline_col = st.columns(2)

                with accept_col:
                    with st.container(key=f"accept-btn-{index}"):
                        if st.button(
                            "Accept",
                            key=f"accept_{invitation['invitation_id']}",
                            width="stretch",
                        ):

                            try:

                                respond_to_invitation(
                                    invitation_id=invitation[
                                        "invitation_id"
                                    ],
                                    status="accepted",
                                )

                                session_id = invitation.get(
                                    "session_id"
                                )

                                if session_id:

                                    session_response = requests.get(
                                        f"{COLLAB_SERVER_URL}/sessions/"
                                        f"{session_id}",
                                        headers=collab_headers(),
                                        timeout=10,
                                    )

                                    session_response.raise_for_status()

                                    st.session_state.collaboration_session = (
                                        session_response.json()
                                    )
                                    st.session_state.collaboration_available = True

                                    st.query_params["collab_session"] = session_response.json()["session_id"]

                                st.rerun()

                            except requests.RequestException as exc:
                                st.error(
                                    f"Unable to accept invitation: {exc}"
                                )

                with decline_col:
                    with st.container(key=f"decline-btn-{index}"):
                        if st.button(
                            "Decline",
                            key=f"decline_{invitation['invitation_id']}",
                            width="stretch",
                        ):

                            try:

                                respond_to_invitation(
                                    invitation_id=invitation[
                                        "invitation_id"
                                    ],
                                    status="declined",
                                )

                                st.rerun()

                            except requests.RequestException as exc:
                                st.error(
                                    f"Unable to decline invitation: {exc}"
                                )