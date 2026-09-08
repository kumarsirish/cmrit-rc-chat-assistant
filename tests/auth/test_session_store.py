from datetime import datetime, timedelta

import mysql.connector
import pytest

from rag.common.config import DatabaseConfig
from rag.auth.session_store import SessionStore

TEST_DATABASE_CONFIG = DatabaseConfig(
    host="127.0.0.1",
    port=3306,
    user="sirkumar",
    password="Issa123#",
    database="rag_sessions_test",
)


def _database_is_reachable() -> bool:
    try:
        connection = mysql.connector.connect(
            host=TEST_DATABASE_CONFIG.host,
            port=TEST_DATABASE_CONFIG.port,
            user=TEST_DATABASE_CONFIG.user,
            password=TEST_DATABASE_CONFIG.password,
        )
        connection.close()
        return True
    except mysql.connector.Error:
        return False


requires_mysql = pytest.mark.skipif(
    not _database_is_reachable(), reason="MySQL server is not reachable"
)


@pytest.fixture
def session_store():
    store = SessionStore(TEST_DATABASE_CONFIG, timeout_minutes=5)
    yield store
    connection = mysql.connector.connect(
        host=TEST_DATABASE_CONFIG.host,
        port=TEST_DATABASE_CONFIG.port,
        user=TEST_DATABASE_CONFIG.user,
        password=TEST_DATABASE_CONFIG.password,
    )
    cursor = connection.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DATABASE_CONFIG.database}")
    connection.commit()
    cursor.close()
    connection.close()


def _backdate_last_active(session_id: str, minutes_ago: int) -> None:
    connection = mysql.connector.connect(
        host=TEST_DATABASE_CONFIG.host,
        port=TEST_DATABASE_CONFIG.port,
        user=TEST_DATABASE_CONFIG.user,
        password=TEST_DATABASE_CONFIG.password,
        database=TEST_DATABASE_CONFIG.database,
    )
    cursor = connection.cursor()
    backdated_time = datetime.now() - timedelta(minutes=minutes_ago)
    cursor.execute(
        "UPDATE sessions SET last_active_at = %s WHERE session_id = %s",
        (backdated_time, session_id),
    )
    connection.commit()
    cursor.close()
    connection.close()


@requires_mysql
def test_create_session_is_not_expired(session_store):
    session_id = session_store.create_session("student@cmrit.ac.in")
    assert session_store.is_expired(session_id) is False


@requires_mysql
def test_unknown_session_is_expired(session_store):
    assert session_store.is_expired("not-a-real-session-id") is True


@requires_mysql
def test_session_expires_after_timeout(session_store):
    session_id = session_store.create_session("student@cmrit.ac.in")
    _backdate_last_active(session_id, minutes_ago=10)
    assert session_store.is_expired(session_id) is True


@requires_mysql
def test_touch_session_resets_the_timeout(session_store):
    session_id = session_store.create_session("student@cmrit.ac.in")
    _backdate_last_active(session_id, minutes_ago=10)
    assert session_store.is_expired(session_id) is True

    session_store.touch_session(session_id)
    assert session_store.is_expired(session_id) is False


@requires_mysql
def test_messages_are_stored_in_order(session_store):
    session_id = session_store.create_session("student@cmrit.ac.in")
    session_store.add_message(session_id, "user", "What is the leave policy?")
    session_store.add_message(session_id, "assistant", "Staff get 12 days of leave.")

    messages = session_store.get_messages(session_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "What is the leave policy?"
    assert messages[1].content == "Staff get 12 days of leave."


@requires_mysql
def test_get_timing_summary_is_empty_before_any_timing_recorded(session_store):
    assert session_store.get_timing_summary() == (0, None)


@requires_mysql
def test_assistant_message_timing_is_recorded_and_averaged(session_store):
    session_id = session_store.create_session("student@cmrit.ac.in")
    session_store.add_message(session_id, "user", "What is the leave policy?")
    session_store.add_message(
        session_id, "assistant", "Staff get 12 days of leave.",
        retrieval_seconds=0.2, llm_seconds=1.0, total_seconds=1.2,
    )
    session_store.add_message(session_id, "user", "What about sick leave?")
    session_store.add_message(
        session_id, "assistant", "Sick leave is separate.",
        retrieval_seconds=0.4, llm_seconds=2.0, total_seconds=2.4,
    )

    count, average_seconds = session_store.get_timing_summary()
    assert count == 2
    assert average_seconds == pytest.approx(1.8)

    messages = session_store.get_messages(session_id)
    assert messages[0].total_seconds is None  # user message: no timing
    assert messages[1].total_seconds == pytest.approx(1.2)


@requires_mysql
def test_get_timing_summary_only_covers_the_most_recent_limit(session_store):
    from rag.auth.session_store import RECENT_TIMING_LIMIT

    session_id = session_store.create_session("student@cmrit.ac.in")
    for i in range(RECENT_TIMING_LIMIT + 5):
        session_store.add_message(
            session_id, "assistant", f"answer {i}", total_seconds=float(i)
        )

    count, _ = session_store.get_timing_summary()
    assert count == RECENT_TIMING_LIMIT


@requires_mysql
def test_legacy_assistant_messages_without_timing_are_backfilled_on_startup(session_store):
    from rag.auth.session_store import LEGACY_TIMING_SECONDS

    session_id = session_store.create_session("student@cmrit.ac.in")
    # Simulate a message stored before timing columns existed, by
    # inserting directly without going through add_message's defaults.
    session_store.add_message(session_id, "assistant", "An old answer with no recorded timing.")

    # Re-opening the store re-runs the startup migration, which should
    # backfill any assistant row still missing a total_seconds.
    reopened_store = SessionStore(TEST_DATABASE_CONFIG, timeout_minutes=5)
    messages = reopened_store.get_messages(session_id)
    assert messages[0].total_seconds == LEGACY_TIMING_SECONDS
