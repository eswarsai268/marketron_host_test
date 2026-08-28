import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any
from dotenv import load_dotenv

load_dotenv()

TOKEN_TTL_SECONDS = 86400


class CollaborationAuthError(ValueError):
    """Raised when a collaboration token is invalid."""


def _get_secret() -> bytes:
    secret = os.getenv("COLLAB_AUTH_SECRET")

    if not secret:
        raise CollaborationAuthError(
            "COLLAB_AUTH_SECRET is not configured."
        )

    return secret.encode("utf-8")


def _urlsafe_b64encode(data: bytes) -> str:
    return (
        base64.urlsafe_b64encode(data)
        .decode("utf-8")
        .rstrip("=")
    )


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)

    try:
        return base64.urlsafe_b64decode(
            value + padding
        )
    except Exception as exc:
        raise CollaborationAuthError(
            "Malformed collaboration token."
        ) from exc


def create_collaboration_token(
    user_id: str,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> str:
    """
    Create a short-lived HMAC-signed collaboration token.

    The token contains only:
        sub  -> authenticated user id
        iat  -> issued-at timestamp
        exp  -> expiration timestamp
    """

    if not user_id:
        raise CollaborationAuthError(
            "user_id is required."
        )

    if ttl_seconds < 30 or ttl_seconds > 86400:
        raise CollaborationAuthError(
            "Token TTL must be between 30 and 86400 seconds."
        )

    now = int(time.time())

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + ttl_seconds,
    }

    payload_bytes = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    payload_encoded = _urlsafe_b64encode(
        payload_bytes
    )

    signature = hmac.new(
        _get_secret(),
        payload_encoded.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    signature_encoded = _urlsafe_b64encode(
        signature
    )

    return (
        f"{payload_encoded}.{signature_encoded}"
    )


def verify_collaboration_token(
    token: str,
) -> dict[str, Any]:
    """
    Verify signature and expiry.

    Returns the decoded payload when valid.
    """

    if not token or not isinstance(token, str):
        raise CollaborationAuthError(
            "Collaboration token is required."
        )

    parts = token.split(".")

    if len(parts) != 2:
        raise CollaborationAuthError(
            "Malformed collaboration token."
        )

    payload_encoded, signature_encoded = parts

    expected_signature = hmac.new(
        _get_secret(),
        payload_encoded.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    supplied_signature = _urlsafe_b64decode(
        signature_encoded
    )

    if not hmac.compare_digest(
        supplied_signature,
        expected_signature,
    ):
        raise CollaborationAuthError(
            "Invalid collaboration token signature."
        )

    payload_bytes = _urlsafe_b64decode(
        payload_encoded
    )

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8")
        )
    except Exception as exc:
        raise CollaborationAuthError(
            "Invalid collaboration token payload."
        ) from exc

    user_id = payload.get("sub")
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")

    if not isinstance(user_id, str) or not user_id:
        raise CollaborationAuthError(
            "Token does not contain a valid user identity."
        )

    if not isinstance(issued_at, int):
        raise CollaborationAuthError(
            "Token issued-at timestamp is invalid."
        )

    if not isinstance(expires_at, int):
        raise CollaborationAuthError(
            "Token expiration timestamp is invalid."
        )

    now = int(time.time())

    if expires_at <= now:
        raise CollaborationAuthError(
            "Collaboration token has expired."
        )

    if issued_at > now + 30:
        raise CollaborationAuthError(
            "Collaboration token issued-at time is invalid."
        )

    return payload