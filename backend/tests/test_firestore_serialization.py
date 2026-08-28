from datetime import datetime, timezone

from src.firestore_db import _utc_now


def test_utc_now_returns_timezone_aware_datetime():
    value = _utc_now()

    assert isinstance(value, datetime)
    assert value.tzinfo is not None
    assert value.utcoffset() is not None


def test_utc_now_is_utc():
    value = _utc_now()

    assert value.tzinfo == timezone.utc