import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.collab_auth as collab_auth
from backend.collab_auth import create_collaboration_token
from backend.collaboration_server import app


client = TestClient(app)


# ============================================================
# TEST DATA
# ============================================================

USER_A = "security-test-user-a"
USER_B = "security-test-user-b"
USER_C = "security-test-user-c"

FAKE_SESSION = "does-not-exist"


def bearer(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}"
    }


# ============================================================
# 1. TOKEN CREATION
# ============================================================

def test_valid_token_creation():
    token = create_collaboration_token(USER_A)

    assert isinstance(token, str)
    assert token.count(".") == 1

    print("✅ Valid token creation")


# ============================================================
# 2. HEALTH
# ============================================================

def test_health_is_public():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    print("✅ Health endpoint remains public")


# ============================================================
# 3. INVALID TOKEN
# ============================================================

def test_fake_token_is_rejected():
    response = client.get(
        "/users/search",
        params={"q": "test"},
        headers=bearer("fake.token.value"),
    )

    assert response.status_code == 401

    print("✅ Forged token rejected")


# ============================================================
# 4. MALFORMED AUTH HEADER
# ============================================================

def test_malformed_authorization_is_rejected():
    response = client.get(
        "/users/search",
        params={"q": "test"},
        headers={
            "Authorization": "NotBearer something"
        },
    )

    assert response.status_code == 401

    print("✅ Malformed authorization rejected")


# ============================================================
# 5. MISSING AUTH
# ============================================================

def test_missing_auth_is_rejected():
    response = client.get(
        "/users/search",
        params={"q": "test"},
    )

    assert response.status_code == 401

    print("✅ Missing authentication rejected")


# ============================================================
# 6. EXPIRED TOKEN
# ============================================================

def test_expired_token_is_rejected():
    now = int(collab_auth.time.time())

    token = create_collaboration_token(
        USER_A,
        ttl_seconds=30,
    )

    with patch.object(
        collab_auth.time,
        "time",
        return_value=now + 31,
    ):
        response = client.get(
            "/users/search",
            params={"q": "test"},
            headers=bearer(token),
        )

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()

    print("✅ Expired token rejected")


# ============================================================
# 7. UNAUTHENTICATED SESSION ACCESS
# ============================================================

def test_unauthenticated_session_access():
    response = client.get(
        f"/sessions/{FAKE_SESSION}"
    )

    assert response.status_code == 401

    print("✅ Unauthenticated session access rejected")


# ============================================================
# 8. UNAUTHENTICATED MESSAGE ACCESS
# ============================================================

def test_unauthenticated_message_access():
    response = client.get(
        f"/sessions/{FAKE_SESSION}/messages"
    )

    assert response.status_code == 401

    print("✅ Unauthenticated message access rejected")


# ============================================================
# 9. WEBSOCKET — FORGED TOKEN
# ============================================================

def test_websocket_forged_token():
    with client.websocket_connect(
        f"/ws/collab/{FAKE_SESSION}"
    ) as websocket:

        websocket.send_json(
            {
                "event": "authenticate",
                "token": "fake.token",
            }
        )

        message = websocket.receive_json()

        assert message["event"] == "error"

        print("✅ WebSocket rejects forged token")


# ============================================================
# 10. WEBSOCKET — NO AUTHENTICATION
# ============================================================

def test_websocket_requires_authentication():
    with client.websocket_connect(
        f"/ws/collab/{FAKE_SESSION}"
    ) as websocket:

        websocket.send_json(
            {
                "event": "join",
            }
        )

        message = websocket.receive_json()

        assert message["event"] == "error"

        print("✅ WebSocket rejects unauthenticated JOIN")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("MARKETRON COLLABORATION SECURITY TEST")
    print("=" * 60)
    print()

    tests = [
        test_valid_token_creation,
        test_health_is_public,
        test_fake_token_is_rejected,
        test_malformed_authorization_is_rejected,
        test_missing_auth_is_rejected,
        test_expired_token_is_rejected,
        test_unauthenticated_session_access,
        test_unauthenticated_message_access,
        test_websocket_forged_token,
        test_websocket_requires_authentication,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"❌ {test.__name__}: {exc}")

    print()
    print("=" * 60)
    print(f"RESULT: {passed}/{len(tests)} TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()