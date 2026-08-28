import os
import secrets
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore
load_dotenv()

_firestore_client = None


def _get_firestore():
    """
    Lazily initialize and return the Firestore client.

    Firebase must not be initialized when this module is imported.
    """
    global _firestore_client

    if _firestore_client is not None:
        return _firestore_client

    if not firebase_admin._apps:
        credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

        if not credentials_path:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS_PATH environment variable is not set."
            )

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Firebase credentials file not found: {credentials_path}"
            )

        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)

    _firestore_client = firestore.client()
    return _firestore_client


def _utc_now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# USER DIRECTORY
# ---------------------------------------------------------------------------

def upsert_user(
    user_id: str,
    email: str,
    display_name: Optional[str] = None,
    profile_picture: Optional[str] = None,
):
    db = _get_firestore()

    user_ref = db.collection("users").document(user_id)
    existing = user_ref.get()

    data = {
        "user_id": user_id,
        "email": email,
        "display_name": display_name,
        "profile_picture": profile_picture,
        "last_seen": _utc_now(),
    }

    if not existing.exists:
        data["created_at"] = _utc_now()

    user_ref.set(data, merge=True)

    result = user_ref.get()
    return result.to_dict()


def get_user(user_id: str):
    db = _get_firestore()

    snapshot = db.collection("users").document(user_id).get()

    if not snapshot.exists:
        return None

    return snapshot.to_dict()


def get_user_by_email(email: str):
    db = _get_firestore()

    query = (
        db.collection("users")
        .where("email", "==", email)
        .limit(1)
        .stream()
    )

    for document in query:
        return document.to_dict()

    return None


def search_users(search_text: str, limit: int = 10):
    """
    Search only users already stored in the Marketron user directory.

    Firestore does not provide a general case-insensitive substring search,
    so this performs a bounded read and filters locally.
    """
    db = _get_firestore()

    search_text = search_text.strip().lower()

    if not search_text:
        return []

    documents = (
        db.collection("users")
        .order_by("email")
        .limit(100)
        .stream()
    )

    results = []

    for document in documents:
        user = document.to_dict()

        email = str(user.get("email", "")).lower()
        display_name = str(user.get("display_name", "")).lower()

        if search_text in email or search_text in display_name:
            results.append(user)

            if len(results) >= limit:
                break

    return results


def update_last_seen(user_id: str):
    db = _get_firestore()

    user_ref = db.collection("users").document(user_id)

    user_ref.set(
        {
            "last_seen": _utc_now(),
        },
        merge=True,
    )


# ---------------------------------------------------------------------------
# COLLABORATION SESSIONS
# ---------------------------------------------------------------------------

def _generate_session_id() -> str:
    return secrets.token_urlsafe(24)


def create_session(host_user_id: str, title: Optional[str] = None):
    db = _get_firestore()

    session_id = _generate_session_id()
    now = _utc_now()

    session_data = {
        "session_id": session_id,
        "host_user_id": host_user_id,
        "title": title,
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    session_ref = db.collection("sessions").document(session_id)
    session_ref.set(session_data)

    add_member(
        session_id=session_id,
        user_id=host_user_id,
        role="host",
    )

    return session_data


def get_session(session_id: str):
    db = _get_firestore()

    snapshot = db.collection("sessions").document(session_id).get()

    if not snapshot.exists:
        return None

    return snapshot.to_dict()


def close_session(session_id: str):
    update_session(
        session_id,
        status="closed",
    )


def _delete_collection_documents(collection_ref, batch_size: int = 400):
    """Delete documents from a collection in bounded Firestore batches without crashing memory."""
    db = _get_firestore()
    
    while True:
        # THE FIX: Use .limit() to only pull safe chunks into memory
        documents = list(collection_ref.limit(batch_size).stream())
        
        if not documents:
            break

        batch = db.batch()
        for document in documents:
            batch.delete(document.reference)
        batch.commit()


def delete_session_data(session_id: str):
    """Permanently delete one collaboration session and all temporary data."""
    db = _get_firestore()
    session_ref = db.collection("sessions").document(session_id)

    if not session_ref.get().exists:
        return False

    _delete_collection_documents(session_ref.collection("messages"))
    _delete_collection_documents(session_ref.collection("members"))

    # THE FIX: Apply the same memory-safe chunking to invitations
    while True:
        invitations = list(
            db.collection("invitations")
            .where("session_id", "==", session_id)
            .limit(400)
            .stream()
        )
        
        if not invitations:
            break
            
        batch = db.batch()
        for document in invitations:
            batch.delete(document.reference)
        batch.commit()

    session_ref.delete()
    return True


def update_session(
    session_id: str,
    title: Optional[str] = None,
    status: Optional[str] = None,
):
    db = _get_firestore()

    updates = {
        "updated_at": _utc_now(),
    }

    if title is not None:
        updates["title"] = title

    if status is not None:
        if status not in {"active", "closed"}:
            raise ValueError("status must be 'active' or 'closed'")

        updates["status"] = status

    session_ref = db.collection("sessions").document(session_id)
    session_ref.update(updates)

    return get_session(session_id)


# ---------------------------------------------------------------------------
# SESSION MEMBERSHIP
# ---------------------------------------------------------------------------

def add_member(
    session_id: str,
    user_id: str,
    role: str = "member",
):
    if role not in {"host", "member"}:
        raise ValueError("role must be 'host' or 'member'")

    db = _get_firestore()

    member_ref = (
        db.collection("sessions")
        .document(session_id)
        .collection("members")
        .document(user_id)
    )

    existing = member_ref.get()
    now = _utc_now()

    if existing.exists:
        existing_data = existing.to_dict()

        member_ref.set(
            {
                "user_id": user_id,
                "role": existing_data.get("role", role),
                "status": "active",
                "last_seen": now,
            },
            merge=True,
        )

        return member_ref.get().to_dict()

    member_data = {
        "user_id": user_id,
        "role": role,
        "joined_at": now,
        "last_seen": now,
        "status": "active",
    }

    member_ref.set(member_data)

    return member_data


def remove_member(session_id: str, user_id: str):
    return set_member_status(
        session_id=session_id,
        user_id=user_id,
        status="left",
    )


def get_member(session_id: str, user_id: str):
    db = _get_firestore()

    member_ref = (
        db.collection("sessions")
        .document(session_id)
        .collection("members")
        .document(user_id)
    )

    snapshot = member_ref.get()

    if not snapshot.exists:
        return None

    return snapshot.to_dict()


def list_members(session_id: str):
    db = _get_firestore()

    documents = (
        db.collection("sessions")
        .document(session_id)
        .collection("members")
        .stream()
    )

    members = []

    for document in documents:
        member = document.to_dict()

        user_id = member.get("user_id")

        if user_id:
            user_doc = (
                db.collection("users")
                .document(user_id)
                .get()
            )

            if user_doc.exists:
                user_data = user_doc.to_dict()

                member["display_name"] = user_data.get(
                    "display_name",
                    "Unknown User"
                )

        members.append(member)

    return sorted(
        members,
        key=lambda member: member.get("joined_at") or datetime.min.replace(
            tzinfo=timezone.utc
        ),
    )


def set_member_status(
    session_id: str,
    user_id: str,
    status: str,
):
    if status not in {"active", "left"}:
        raise ValueError("status must be 'active' or 'left'")

    db = _get_firestore()

    member_ref = (
        db.collection("sessions")
        .document(session_id)
        .collection("members")
        .document(user_id)
    )

    if not member_ref.get().exists:
        return None

    member_ref.update(
        {
            "status": status,
            "last_seen": _utc_now(),
        }
    )

    return member_ref.get().to_dict()


# ---------------------------------------------------------------------------
# INVITATIONS
# ---------------------------------------------------------------------------

def _generate_invitation_id() -> str:
    return secrets.token_urlsafe(24)


def create_invitation(
    session_id: str,
    sender_user_id: str,
    recipient_user_id: str,
):
    db = _get_firestore()

    session = get_session(session_id)

    if session is None:
        raise ValueError("Session not found.")

    if session.get("host_user_id") != sender_user_id:
        raise PermissionError(
            "Only the session host can create invitations."
        )

    recipient_member = get_member(
        session_id,
        recipient_user_id,
    )

    if recipient_member and recipient_member.get("status") == "active":
        raise ValueError(
            "User is already an active member of this session."
        )

    existing_invitations = (
        db.collection("invitations")
        .where("session_id", "==", session_id)
        .where("recipient_user_id", "==", recipient_user_id)
        .where("status", "==", "pending")
        .limit(1)
        .stream()
    )

    for _ in existing_invitations:
        raise ValueError(
            "A pending invitation already exists for this user."
        )

    invitation_id = _generate_invitation_id()
    now = _utc_now()

    invitation_data = {
        "invitation_id": invitation_id,
        "session_id": session_id,
        "sender_user_id": sender_user_id,
        "recipient_user_id": recipient_user_id,
        "status": "pending",
        "created_at": now,
        "responded_at": None,
    }

    db.collection("invitations").document(invitation_id).set(
        invitation_data
    )

    return invitation_data


def get_invitation(invitation_id: str):
    db = _get_firestore()

    snapshot = (
        db.collection("invitations")
        .document(invitation_id)
        .get()
    )

    if not snapshot.exists:
        return None

    return snapshot.to_dict()


def list_received_invitations(user_id: str):
    db = _get_firestore()

    documents = (
        db.collection("invitations")
        .where("recipient_user_id", "==", user_id)
        .stream()
    )

    invitations = [
        document.to_dict()
        for document in documents
    ]

    return sorted(
        invitations,
        key=lambda invitation: invitation.get("created_at")
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def list_sent_invitations(user_id: str):
    db = _get_firestore()

    documents = (
        db.collection("invitations")
        .where("sender_user_id", "==", user_id)
        .stream()
    )

    invitations = [
        document.to_dict()
        for document in documents
    ]

    return sorted(
        invitations,
        key=lambda invitation: invitation.get("created_at")
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def respond_to_invitation(
    invitation_id: str,
    user_id: str,
    status: str,
):
    if status not in {"accepted", "declined"}:
        raise ValueError(
            "status must be 'accepted' or 'declined'"
        )

    db = _get_firestore()
    transaction = db.transaction()

    @firestore.transactional
    def _apply_response(transaction):
        invitation_ref = (
            db.collection("invitations")
            .document(invitation_id)
        )

        invitation_snapshot = invitation_ref.get(
            transaction=transaction
        )

        if not invitation_snapshot.exists:
            raise ValueError("Invitation not found.")

        invitation = invitation_snapshot.to_dict()

        if invitation.get("recipient_user_id") != user_id:
            raise PermissionError(
                "Only the intended recipient can respond to this invitation."
            )

        if invitation.get("status") != "pending":
            raise ValueError("Invitation is no longer pending.")

        now = _utc_now()

        member_ref = None
        existing_member = None

        if status == "accepted":
            session_id = invitation["session_id"]

            member_ref = (
                db.collection("sessions")
                .document(session_id)
                .collection("members")
                .document(user_id)
            )

            existing_member = member_ref.get(
                transaction=transaction
            )

        transaction.update(
            invitation_ref,
            {
                "status": status,
                "responded_at": now,
            },
        )

        if status == "accepted":
            if existing_member.exists:
                existing_data = existing_member.to_dict()
                role = existing_data.get("role", "member")
                joined_at = existing_data.get(
                    "joined_at",
                    now,
                )
            else:
                role = "member"
                joined_at = now

            transaction.set(
                member_ref,
                {
                    "user_id": user_id,
                    "role": role,
                    "joined_at": joined_at,
                    "last_seen": now,
                    "status": "active",
                },
                merge=True,
            )

        return invitation

    _apply_response(transaction)

    updated_invitation = get_invitation(
        invitation_id
    )

    if updated_invitation is None:
        raise ValueError(
            "Invitation disappeared after response."
        )

    return updated_invitation


# ---------------------------------------------------------------------------
# PERSISTENT MESSAGES
# ---------------------------------------------------------------------------

def _generate_message_id() -> str:
    return secrets.token_urlsafe(24)


def add_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    message_type: str = "ai_message",
):
    if role not in {"user", "assistant", "system"}:
        raise ValueError(
            "role must be 'user', 'assistant', or 'system'"
        )

    if message_type not in {"ai_message", "team_message"}:
        raise ValueError(
            "message_type must be 'ai_message' or 'team_message'"
        )

    db = _get_firestore()

    session = get_session(session_id)

    if session is None:
        raise ValueError("Session not found.")

    message_id = _generate_message_id()
    now = _utc_now()

    message_data = {
        "message_id": message_id,
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": now,
        "message_type": message_type,
    }

    message_ref = (
        db.collection("sessions")
        .document(session_id)
        .collection("messages")
        .document(message_id)
    )

    message_ref.set(message_data)

    return message_data


def get_messages(
    session_id: str,
    limit: Optional[int] = None,
):
    db = _get_firestore()

    query = (
        db.collection("sessions")
        .document(session_id)
        .collection("messages")
        .order_by("created_at")
    )

    if limit is not None:
        query = query.limit(limit)

    documents = query.stream()

    return [document.to_dict() for document in documents]


def get_messages_after(
    session_id: str,
    cursor_or_timestamp,
):
    db = _get_firestore()

    if isinstance(cursor_or_timestamp, str):
        cursor_message = (
            db.collection("sessions")
            .document(session_id)
            .collection("messages")
            .document(cursor_or_timestamp)
            .get()
        )

        if not cursor_message.exists:
            raise ValueError("Message cursor not found.")

        cursor_data = cursor_message.to_dict()
        cursor_timestamp = cursor_data.get("created_at")

    else:
        cursor_timestamp = cursor_or_timestamp

    query = (
        db.collection("sessions")
        .document(session_id)
        .collection("messages")
        .where("created_at", ">", cursor_timestamp)
        .order_by("created_at")
        .stream()
    )

    return [document.to_dict() for document in query]