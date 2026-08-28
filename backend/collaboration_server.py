import asyncio
from typing import Any, Optional
from fastapi import Header
import json
from datetime import datetime, date

from backend.collab_auth import (
    CollaborationAuthError,
    verify_collaboration_token,
)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.ai_executor import ensure_session_worker, stop_session_worker
from backend.collaboration_state import collaboration_state
from backend.marketron_ai_adapter import build_session_ai_executor
from src.firestore_db import (
    add_message,
    create_invitation,
    create_session,
    delete_session_data,
    get_invitation,
    get_member,
    get_messages,
    get_session,
    get_user,
    list_members,
    list_received_invitations,
    list_sent_invitations,
    respond_to_invitation,
    search_users,
    update_last_seen,
)


app = FastAPI(
    title="Marketron Collaboration Service",
    version="1.0.0",
)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    initial_messages: list[dict[str, Any]] = Field(default_factory=list)


class CreateInvitationRequest(BaseModel):
    recipient_user_id: str


class InvitationResponseRequest(BaseModel):
    status: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_error(message: str) -> dict[str, str]:
    return {
        "event": "error",
        "message": message,
    }


async def _broadcast(
    session_id: str,
    event: dict[str, Any],
    exclude_user_id: Optional[str] = None,
) -> None:
    connections = await collaboration_state.get_connected_users(
        session_id
    )

    dead_connections = []

    for connection in connections:
        if (
            exclude_user_id is not None
            and connection.user_id == exclude_user_id
        ):
            continue

        try:
            # THE FIX: Safe transmission prevents the silent disconnect!
            await _send_json_safe(connection.websocket, event)
        except Exception as exc:
            print(f"⚠️ [DEBUG] BROADCAST FAILED TO {connection.user_id}: {exc}", flush=True)
            dead_connections.append(connection)

    for connection in dead_connections:
        await collaboration_state.disconnect_user(
            session_id=session_id,
            user_id=connection.user_id,
            websocket=connection.websocket,
        )


async def _send_to_user(
    session_id: str,
    user_id: str,
    event: dict[str, Any],
) -> None:
    connections = await collaboration_state.get_connected_users(
        session_id
    )

    for connection in connections:
        if connection.user_id != user_id:
            continue

        try:
            # THE FIX: Safe transmission prevents the silent disconnect!
            await _send_json_safe(connection.websocket, event)
        except Exception as exc:
            print(f"⚠️ [DEBUG] SEND_TO_USER FAILED TO {user_id}: {exc}", flush=True)
            await collaboration_state.disconnect_user(
                session_id=session_id,
                user_id=user_id,
                websocket=connection.websocket,
            )

def _require_http_user(
    authorization: Optional[str],
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization required.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format.",
        )

    token = authorization[7:].strip()

    try:
        payload = verify_collaboration_token(token)
    except CollaborationAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    return payload["sub"]

def _validate_session(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return session


async def _handle_ai_result(result: dict[str, Any]) -> None:
    """Persist and broadcast one completed/failed AI queue item."""

    queue_item = result["queue_item"]
    session_id = queue_item["session_id"]
    user_id = queue_item["user_id"]

    error = result.get("error")

    if error:
        await _broadcast(
            session_id=session_id,
            event={
                "event": "ai_failed",
                "queue_id": queue_item["queue_id"],
                "user_id": user_id,
                "message": "The AI could not process this prompt.",
            },
        )
    else:
        ai_result = result.get("result")
        assistant_content = (
            ai_result
            if isinstance(ai_result, str)
            else str(ai_result)
        ).strip()

        if not assistant_content:
            assistant_content = "The AI returned an empty response."

        assistant_message = add_message(
            session_id=session_id,
            user_id="marketron-ai",
            role="assistant",
            content=assistant_content,
            message_type="ai_message",
        )

        await _broadcast(
            session_id=session_id,
            event={
                "event": "ai_completed",
                "queue_id": queue_item["queue_id"],
                "user_id": user_id,
                "message": assistant_message,
            },
        )

    await _send_to_user(
        session_id=session_id,
        user_id=user_id,
        event={
            "event": "unlock_input",
            "user_id": user_id,
        },
    )

    await _broadcast(
        session_id=session_id,
        event={
            "event": "queue_updated",
            "queue_size": await collaboration_state.get_queue_size(session_id),
            "processing": await collaboration_state.is_processing(session_id),
        },
    )


async def _queue_ai_prompt(
    session_id: str,
    user_id: str,
    prompt: str,
) -> dict[str, Any]:
    """Validate, persist, broadcast, and queue one collaboration AI prompt."""
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("AI prompt cannot be empty.")

    member = get_member(session_id, user_id)
    if member is None or member.get("status") != "active":
        raise PermissionError(
            "You are not an active member of this session."
        )

    accepted = await collaboration_state.add_pending_prompt(
        session_id=session_id,
        user_id=user_id,
        prompt=prompt,
    )

    if not accepted:
        raise ValueError("You already have a prompt waiting.")

    user_message = add_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=prompt,
        message_type="ai_message",
    )

    await _broadcast(
        session_id=session_id,
        event={
            "event": "ai_message",
            "message": user_message,
        },
    )

    queue_id = f"{session_id}:{user_id}:{user_message['message_id']}"
    queue_item = {
        "queue_id": queue_id,
        "session_id": session_id,
        "user_id": user_id,
        "prompt": prompt,
        "message_id": user_message["message_id"],
    }

    await collaboration_state.enqueue_prompt(
        session_id=session_id,
        prompt=queue_item,
    )

    position = await collaboration_state.get_queue_size(session_id)

    await _broadcast(
        session_id=session_id,
        event={
            "event": "prompt_queued",
            "queue_id": queue_id,
            "user_id": user_id,
            "position": position,
        },
    )

    await _send_to_user(
        session_id=session_id,
        user_id=user_id,
        event={
            "event": "input_locked",
            "user_id": user_id,
            "queue_id": queue_id,
        },
    )

    await ensure_session_worker(
        session_id=session_id,
        ai_executor=build_session_ai_executor(session_id),
        on_result=_handle_ai_result,
    )

    return {
        "queue_id": queue_id,
        "position": position,
        "message": user_message,
    }

def _json_default(obj: Any) -> Any:
    """Safely convert datetime objects to ISO format for WebSocket transmission."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def _send_json_safe(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """Safely transmit JSON over WebSocket without crashing on datetime fields."""
    text = json.dumps(payload, default=_json_default)
    await websocket.send_text(text)

# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "marketron-collaboration",
    }


# ============================================================================
# SESSION ROUTES
# ============================================================================

@app.post("/sessions")
async def create_session_route(
    request: CreateSessionRequest,
    authorization: Optional[str] = Header(default=None),
):
    host_user_id = _require_http_user(authorization)

    try:
        user = get_user(host_user_id)

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="Authenticated user not found.",
            )

        validated_messages = []

        for message in request.initial_messages:
            role = message.get("role")
            content = message.get("content")

            if role not in {"user", "assistant", "system"}:
                raise ValueError(
                    "Initial conversation contains an invalid role."
                )

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Initial conversation contains empty content."
                )

            if role == "assistant":
                source_user_id = "marketron-ai"
            elif role == "system":
                source_user_id = "system"
            else:
                source_user_id = host_user_id

            validated_messages.append(
                {
                    "role": role,
                    "content": content,
                    "user_id": str(source_user_id),
                }
            )

        session = create_session(
            host_user_id=host_user_id,
            title=request.title,
        )

        for message in validated_messages:
            add_message(
                session_id=session["session_id"],
                user_id=message["user_id"],
                role=message["role"],
                content=message["content"],
                message_type="ai_message",
            )

        return session

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to create session.",
        )


@app.get("/sessions/{session_id}")
async def get_session_route(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(
        authorization
    )

    _validate_session(session_id)

    member = get_member(
        session_id,
        user_id,
    )

    if member is None or member.get("status") != "active":
        raise HTTPException(
            status_code=403,
            detail="You are not an active member of this session.",
        )

    return get_session(session_id)


@app.get("/sessions/{session_id}/members")
async def list_members_route(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(
        authorization
    )

    _validate_session(session_id)

    member = get_member(
        session_id,
        user_id,
    )

    if member is None or member.get("status") != "active":
        raise HTTPException(
            status_code=403,
            detail="You are not an active member of this session.",
        )
    return {
        "members": list_members(session_id),
    }


@app.post("/sessions/{session_id}/leave")
async def leave_session_route(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(authorization)

    session = _validate_session(session_id)

    member = get_member(
        session_id,
        user_id,
    )

    if member is None or member.get("status") != "active":
        raise HTTPException(
            status_code=403,
            detail="You are not an active member of this session.",
        )

    if session.get("host_user_id") == user_id:
        raise HTTPException(
            status_code=400,
            detail="The host must end the collaboration instead of leaving it.",
        )

    try:
        from src.firestore_db import remove_member

        updated_member = remove_member(
            session_id=session_id,
            user_id=user_id,
        )

        await _broadcast(
            session_id=session_id,
            event={
                "event": "user_left",
                "user_id": user_id,
            },
            exclude_user_id=user_id,
        )

        return {
            "status": "left",
            "member": updated_member,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to leave collaboration.",
        )


@app.post("/sessions/{session_id}/end")
async def end_session_route(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(authorization)

    session = _validate_session(session_id)

    if session.get("host_user_id") != user_id:
        raise HTTPException(
            status_code=403,
            detail="Only the session host can end the collaboration.",
        )

    if session.get("status") != "active":
        raise HTTPException(
            status_code=409,
            detail="Collaboration session is already closed.",
        )

    try:
        # Close the persistent session first so new operations can no longer
        # treat it as an active collaboration while cleanup is running.
        from src.firestore_db import close_session

        close_session(session_id)

        # Stop the in-memory AI worker before deleting its persisted data.
        await stop_session_worker(session_id)

        # Inform every currently connected browser and close its socket.
        connections = await collaboration_state.get_connected_users(
            session_id
        )

        for connection in connections:
            try:
                await connection.websocket.send_json(
                    {
                        "event": "session_ended",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass

        for connection in connections:
            try:
                await connection.websocket.close(code=1000)
            except Exception:
                pass

        await collaboration_state.remove_session(session_id)

        # Collaboration data is temporary. Delete messages, members,
        # invitations, and finally the session document itself.
        deleted = delete_session_data(session_id)

        return {
            "status": "ended",
            "session_id": session_id,
            "deleted": deleted,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to end collaboration.",
        )


# ============================================================================
# USER SEARCH
# ============================================================================

@app.get("/users/search")
async def search_users_route(
    q: str,
    limit: int = 10,
    authorization: Optional[str] = Header(default=None),
):
    _require_http_user(
        authorization
    )

    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50.",
        )

    return {
        "users": search_users(
            search_text=q,
            limit=limit,
        ),
    }

# ============================================================================
# INVITATIONS
# ============================================================================

@app.post("/sessions/{session_id}/invitations")
async def create_invitation_route(
    session_id: str,
    request: CreateInvitationRequest,
    authorization: Optional[str] = Header(default=None),
):
    sender_user_id = _require_http_user(
        authorization
    )

    session = _validate_session(session_id)

    if session["host_user_id"] != sender_user_id:
        raise HTTPException(
            status_code=403,
            detail="Only the session host can create invitations.",
        )

    try:
        invitation = create_invitation(
            session_id=session_id,
            sender_user_id=sender_user_id,
            recipient_user_id=request.recipient_user_id,
        )

        await _send_to_user(
            session_id=session_id,
            user_id=request.recipient_user_id,
            event={
                "event": "invitation_received",
                "invitation_id": invitation["invitation_id"],
                "session_id": session_id,
                "sender_user_id": sender_user_id,
            },
        )

        return invitation

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        message = str(exc)

        if (
            "already exists" in message.lower()
            or "already an active member" in message.lower()
        ):
            raise HTTPException(
                status_code=409,
                detail=message,
            )

        raise HTTPException(
            status_code=400,
            detail=message,
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to create invitation.",
        )

@app.post("/invitations/{invitation_id}/respond")
async def respond_invitation_route(
    invitation_id: str,
    request: InvitationResponseRequest,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(authorization)

    invitation = get_invitation(invitation_id)

    if invitation is None:
        raise HTTPException(
            status_code=404,
            detail="Invitation not found.",
        )

    try:
        result = respond_to_invitation(
            invitation_id=invitation_id,
            user_id=user_id,
            status=request.status,
        )

        return result

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to respond to invitation.",
        )

@app.get("/invitations/received")
async def received_invitations_route(
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(
        authorization
    )

    invitations = list_received_invitations(
        user_id
    )

    enriched = []

    for invitation in invitations:
        sender = get_user(
            invitation.get("sender_user_id", "")
        )

        enriched.append(
            {
                **invitation,
                "sender": {
                    "user_id": sender.get("user_id"),
                    "display_name": sender.get("display_name"),
                    "email": sender.get("email"),
                    "profile_picture": sender.get("profile_picture"),
                }
                if sender
                else None,
            }
        )

    return {
        "invitations": enriched,
    }


@app.get("/invitations/sent")
async def sent_invitations_route(
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(
        authorization
    )

    invitations = list_sent_invitations(
        user_id
    )

    enriched = []

    for invitation in invitations:
        recipient = get_user(
            invitation.get("recipient_user_id", "")
        )

        enriched.append(
            {
                **invitation,
                "recipient": {
                    "user_id": recipient.get("user_id"),
                    "display_name": recipient.get("display_name"),
                    "email": recipient.get("email"),
                    "profile_picture": recipient.get("profile_picture"),
                }
                if recipient
                else None,
            }
        )

    return {
        "invitations": enriched,
    }


# ============================================================================
# MESSAGE ROUTES
# ============================================================================

@app.get("/sessions/{session_id}/messages")
async def get_messages_route(
    session_id: str,
    limit: Optional[int] = None,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _require_http_user(
        authorization
    )

    _validate_session(session_id)

    member = get_member(
        session_id,
        user_id,
    )

    if member is None or member.get("status") != "active":
        raise HTTPException(
            status_code=403,
            detail="You are not an active member of this session.",
        )

    if limit is not None and (limit < 1 or limit > 500):
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 500.",
        )

    return {
        "messages": get_messages(
            session_id=session_id,
            limit=limit,
        ),
    }


# ============================================================================
# WEBSOCKET COLLABORATION
# ============================================================================

@app.websocket("/ws/collab/{session_id}")
async def collaboration_websocket(
    websocket: WebSocket,
    session_id: str,
):
    await websocket.accept()

    user_id: Optional[str] = None
    authenticated = False

    try:
        session = get_session(session_id)

        if session is None:
            await websocket.send_json(
                _safe_error("Session not found.")
            )
            await websocket.close(code=1008)
            return

        while True:
            payload = await websocket.receive_json()

            if not isinstance(payload, dict):
                await websocket.send_json(
                    _safe_error(
                        "Event payload must be a JSON object."
                    )
                )
                continue

            event_type = payload.get("event")

            if not isinstance(event_type, str):
                await websocket.send_json(
                    _safe_error("Missing event type.")
                )
                continue

            # ----------------------------------------------------------------
            # AUTHENTICATE
            # ----------------------------------------------------------------

            if event_type == "authenticate":
                token = payload.get("token")

                try:
                    token_payload = verify_collaboration_token(
                        token
                    )

                except CollaborationAuthError as exc:
                    await websocket.send_json(
                        _safe_error(str(exc))
                    )
                    await websocket.close(code=1008)
                    return

                user_id = token_payload["sub"]
                authenticated = True

                await websocket.send_json(
                    {
                        "event": "authenticated",
                        "user_id": user_id,
                    }
                )

                continue

            if not authenticated:
                await websocket.send_json(
                    _safe_error(
                        "Authenticate before using the collaboration session."
                    )
                )
                continue

            # ----------------------------------------------------------------
            # JOIN
            # ----------------------------------------------------------------

            if event_type == "join":

                if user_id is None:
                    await websocket.send_json(
                        _safe_error(
                            "Authenticated user identity is unavailable."
                        )
                    )
                    await websocket.close(code=1008)
                    return

                member = get_member(
                    session_id,
                    user_id,
                )

                if member is None:
                    await websocket.send_json(
                        _safe_error(
                            "User is not a member of this session."
                        )
                    )
                    await websocket.close(code=1008)
                    return

                if member.get("status") != "active":
                    await websocket.send_json(
                        _safe_error(
                            "User is not an active member of this session."
                        )
                    )
                    await websocket.close(code=1008)
                    return

                await collaboration_state.connect_user(
                    session_id=session_id,
                    user_id=user_id,
                    websocket=websocket,
                )

                update_last_seen(user_id)

                await websocket.send_json(
                    {
                        "event": "joined",
                        "session_id": session_id,
                        "user_id": user_id,
                    }
                )

                await _broadcast(
                    session_id=session_id,
                    event={
                        "event": "user_joined",
                        "user_id": user_id,
                    },
                    exclude_user_id=user_id,
                )

                continue

            # ----------------------------------------------------------------
            # REQUIRE JOIN FOR ALL OTHER EVENTS
            # ----------------------------------------------------------------

            if user_id is None:
                await websocket.send_json(
                    _safe_error(
                        "Join the collaboration session first."
                    )
                )
                continue

            # ----------------------------------------------------------------
            # LEAVE
            # ----------------------------------------------------------------

            if event_type == "leave":
                await collaboration_state.disconnect_user(
                    session_id=session_id,
                    user_id=user_id,
                    websocket=websocket,
                )

                await _broadcast(
                    session_id=session_id,
                    event={
                        "event": "user_left",
                        "user_id": user_id,
                    },
                    exclude_user_id=user_id,
                )

                break

            # ----------------------------------------------------------------
            # TYPING
            # ----------------------------------------------------------------

            if event_type == "typing":
                await collaboration_state.set_typing(
                    session_id=session_id,
                    user_id=user_id,
                )

                await _broadcast(
                    session_id=session_id,
                    event={
                        "event": "typing",
                        "user_id": user_id,
                    },
                    exclude_user_id=user_id,
                )

                continue

            # ----------------------------------------------------------------
            # TYPING STOPPED
            # ----------------------------------------------------------------

            if event_type == "typing_stopped":
                await collaboration_state.stop_typing(
                    session_id=session_id,
                    user_id=user_id,
                )

                await _broadcast(
                    session_id=session_id,
                    event={
                        "event": "typing_stopped",
                        "user_id": user_id,
                    },
                    exclude_user_id=user_id,
                )

                continue

        

            # ----------------------------------------------------------------
            # AI PROMPT
            # ----------------------------------------------------------------

            if event_type == "ai_prompt":
                prompt = payload.get("prompt")
                print(f"🔥 [DEBUG] WEBSOCKET RECEIVED PROMPT: '{prompt}' from {user_id}", flush=True)

                try:
                    result = await _queue_ai_prompt(
                        session_id=session_id,
                        user_id=user_id,
                        prompt=prompt,
                    )
                    print(f"✅ [DEBUG] PROMPT QUEUED SUCCESSFULLY!", flush=True)
                except PermissionError as exc:
                    print(f"❌ [DEBUG] PERMISSION ERROR: {exc}", flush=True)
                    await websocket.send_json(_safe_error(str(exc)))
                    continue
                except ValueError as exc:
                    print(f"❌ [DEBUG] VALUE ERROR: {exc}", flush=True)
                    await websocket.send_json({"event": "queue_rejected", "reason": str(exc)})
                    continue
                except Exception as exc:
                    print(f"❌ [DEBUG] CRITICAL ERROR: {exc}", flush=True)
                    await websocket.send_json(_safe_error("Unable to submit AI prompt."))
                    continue

                continue

            # ----------------------------------------------------------------
            # UNKNOWN EVENT
            # ----------------------------------------------------------------

            await websocket.send_json(
                {
                    "event": "error",
                    "message": f"Unknown event: {event_type}",
                }
            )

    except WebSocketDisconnect:
        if user_id is not None:
            await collaboration_state.disconnect_user(
                session_id=session_id,
                user_id=user_id,
                websocket=websocket,
            )

            await _broadcast(
                session_id=session_id,
                event={
                    "event": "user_left",
                    "user_id": user_id,
                },
                exclude_user_id=user_id,
            )

    except Exception:
        if user_id is not None:
            await collaboration_state.disconnect_user(
                session_id=session_id,
                user_id=user_id,
                websocket=websocket,
            )

        try:
            await websocket.send_json(
                _safe_error(
                    "An unexpected collaboration error occurred."
                )
            )
        except Exception:
            pass