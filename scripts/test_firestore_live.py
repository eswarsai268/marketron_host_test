from src.firestore_db import upsert_user


result = upsert_user(
    user_id="test-user-001",
    email="test@example.com",
    display_name="Firestore Test User",
    profile_picture=None,
)

print("Firestore write successful:")
print(result)