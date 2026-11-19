"""Step 5: User session and conversation storage.

Tracks each browser session in MySQL, identified by the user's email
address, so multiple concurrent users (the app expects fewer than 50
at a time) each get their own conversation history. A session that
has been inactive for longer than the configured timeout (default 5
minutes) is treated as expired, and the UI starts a fresh session for
that user.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import mysql.connector

from rag._09_common.config import DatabaseConfig


@dataclass
class Message:
    role: str
    content: str
    created_at: datetime
    # Populated for assistant messages only (see SessionStore.
    # add_message) — how long that answer took to produce.
    retrieval_seconds: float = None
    llm_seconds: float = None
    total_seconds: float = None
    # User-submitted 1-5 rating for an assistant message (see
    # SessionStore.rate_message) — None until rated, always None for a
    # user-role message.
    rating: int = None


@dataclass
class RecentQuery:
    """One answered question, for the "Recent Queries" page — shared
    across every user's session (see SessionStore.get_recent_queries),
    not scoped to the viewer's own session like get_messages is."""

    question: str
    answer: str
    answered_at: datetime
    rating: int = None


# Placeholder total_seconds backfilled onto assistant messages that
# were stored before timing was tracked, so old rows don't show up as
# NULL/zero and skew the average — see _backfill_legacy_timings.
LEGACY_TIMING_SECONDS = 120.0

# The "average response time" shown in the UI is computed over only
# the most recent this-many assistant answers, so it reflects current
# performance rather than being dragged down/up by the whole history.
RECENT_TIMING_LIMIT = 50

# Default cap for the "Recent Queries" page (see get_recent_queries) —
# newest first, across every user's session.
RECENT_QUERIES_LIMIT = 100


class SessionStore:
    """MySQL-backed store for session activity and chat history."""

    def __init__(self, database_config: DatabaseConfig, timeout_minutes: int):
        self._config = database_config
        self._timeout_minutes = timeout_minutes
        self._ensure_database_and_tables()

    def _connect(self, use_database: bool = True):
        connect_kwargs = {
            "host": self._config.host,
            "port": self._config.port,
            "user": self._config.user,
            "password": self._config.password,
        }
        if use_database:
            connect_kwargs["database"] = self._config.database
        return mysql.connector.connect(**connect_kwargs)

    def _ensure_database_and_tables(self) -> None:
        connection = self._connect(use_database=False)
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {self._config.database} "
            "CHARACTER SET utf8mb4"
        )
        connection.commit()
        cursor.close()
        connection.close()

        connection = self._connect(use_database=True)
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                created_at DATETIME NOT NULL,
                last_active_at DATETIME NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                role VARCHAR(16) NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                retrieval_seconds FLOAT NULL,
                llm_seconds FLOAT NULL,
                total_seconds FLOAT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.commit()
        self._add_timing_columns_if_missing(cursor)
        self._add_rating_column_if_missing(cursor)
        self._backfill_legacy_timings(cursor)
        # Superseded by the timing columns on `messages` above — a
        # query's timing now lives on the assistant message it belongs
        # to instead of a separate, unlinked table.
        cursor.execute("DROP TABLE IF EXISTS query_timings")
        connection.commit()
        cursor.close()
        connection.close()

    def _column_exists(self, cursor, table: str, column: str) -> bool:
        # SHOW COLUMNS stays scoped to the current database connection and
        # only requires privileges on `table` itself — unlike querying
        # information_schema.columns, which some restricted MySQL accounts
        # (e.g. managed hosting, locked-down deployment users) cannot read,
        # causing startup to fail on machines where the DB user only has
        # grants on this one application database.
        cursor.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (column,))
        return cursor.fetchone() is not None

    def _add_timing_columns_if_missing(self, cursor) -> None:
        """Adds the timing columns to a `messages` table created before
        timings were tracked in it (CREATE TABLE IF NOT EXISTS above is
        a no-op on an existing table, so this covers upgrading it)."""
        for column in ("retrieval_seconds", "llm_seconds", "total_seconds"):
            if not self._column_exists(cursor, "messages", column):
                cursor.execute(f"ALTER TABLE messages ADD COLUMN {column} FLOAT NULL")

    def _add_rating_column_if_missing(self, cursor) -> None:
        """Adds the rating column to a `messages` table created before
        answer rating existed (same upgrade pattern as the timing
        columns above)."""
        if not self._column_exists(cursor, "messages", "rating"):
            cursor.execute("ALTER TABLE messages ADD COLUMN rating TINYINT NULL")

    def _backfill_legacy_timings(self, cursor) -> None:
        """Assistant messages stored before timing was tracked have no
        total_seconds; give them a fixed placeholder rather than
        leaving them NULL (which would exclude them from the average)
        or 0 (which would understate it)."""
        cursor.execute(
            "UPDATE messages SET total_seconds = %s "
            "WHERE role = 'assistant' AND total_seconds IS NULL",
            (LEGACY_TIMING_SECONDS,),
        )

    def create_session(self, user_email: str) -> str:
        """Creates a new session row for user_email and returns its session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, user_email, created_at, last_active_at) "
            "VALUES (%s, %s, %s, %s)",
            (session_id, user_email, now, now),
        )
        connection.commit()
        cursor.close()
        connection.close()
        return session_id

    def get_last_active_at(self, session_id: str):
        """Returns the last-active timestamp for session_id, or None."""
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT last_active_at FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        return row[0] if row else None

    def is_expired(self, session_id: str) -> bool:
        """True if session_id is unknown or inactive past the timeout."""
        last_active_at = self.get_last_active_at(session_id)
        if last_active_at is None:
            return True
        timeout = timedelta(minutes=self._timeout_minutes)
        return datetime.now() - last_active_at > timeout

    def touch_session(self, session_id: str) -> None:
        """Updates last_active_at to now, keeping the session alive."""
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE sessions SET last_active_at = %s WHERE session_id = %s",
            (datetime.now(), session_id),
        )
        connection.commit()
        cursor.close()
        connection.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieval_seconds: float = None,
        llm_seconds: float = None,
        total_seconds: float = None,
    ) -> int:
        """Stores one chat message and returns its new row id —
        callers (see dashboard.py) keep this id alongside the message
        so a later rate_message() call knows which row to update.
        retrieval_seconds/llm_seconds/total_seconds are only meaningful
        for role="assistant" (how long that answer took to produce)
        and left as NULL otherwise.
        """
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO messages "
            "(session_id, role, content, created_at, retrieval_seconds, llm_seconds, total_seconds) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (session_id, role, content, datetime.now(), retrieval_seconds, llm_seconds, total_seconds),
        )
        connection.commit()
        message_id = cursor.lastrowid
        cursor.close()
        connection.close()
        return message_id

    def rate_message(self, message_id: int, rating: int) -> None:
        """Records a 1-5 rating for the assistant message with this id
        (see dashboard.py's star-rating widget under each answer).
        Overwrites any existing rating for that message — a user can
        re-rate, there's no "first rating wins" rule."""
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE messages SET rating = %s WHERE id = %s AND role = 'assistant'",
            (rating, message_id),
        )
        connection.commit()
        cursor.close()
        connection.close()

    def get_messages(self, session_id: str) -> list:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT role, content, created_at, retrieval_seconds, llm_seconds, total_seconds, rating "
            "FROM messages WHERE session_id = %s ORDER BY id ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return [
            Message(
                role=row[0],
                content=row[1],
                created_at=row[2],
                retrieval_seconds=row[3],
                llm_seconds=row[4],
                total_seconds=row[5],
                rating=row[6],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Removes a session and its messages, used when a session expires."""
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        connection.commit()
        cursor.close()
        connection.close()

    def clear_all_history(self) -> None:
        """Deletes EVERY session and message, for every user — used by
        the Admin tab's "Clear all history" control. messages has a
        FOREIGN KEY ... ON DELETE CASCADE on sessions, but that cascade
        only fires for rows actually deleted via DELETE (not a bulk
        TRUNCATE on sessions alone wouldn't need it either way) — this
        deletes messages first regardless, so the intent is explicit
        rather than relying on the cascade silently doing the work.
        """
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM sessions")
        connection.commit()
        cursor.close()
        connection.close()

    def get_timing_summary(self) -> tuple:
        """Returns (count, average_total_seconds) over the most recent
        RECENT_TIMING_LIMIT assistant answers (across all sessions).
        average_total_seconds is None if none have been recorded yet.
        """
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*), AVG(total_seconds) FROM ("
            "  SELECT total_seconds FROM messages "
            "  WHERE role = 'assistant' AND total_seconds IS NOT NULL "
            "  ORDER BY id DESC LIMIT %s"
            ") AS recent",
            (RECENT_TIMING_LIMIT,),
        )
        count, average = cursor.fetchone()
        cursor.close()
        connection.close()
        return count, (float(average) if average is not None else None)

    def get_recent_queries(self, limit: int = RECENT_QUERIES_LIMIT) -> list:
        """Returns the most recent `limit` answered questions, newest
        first, across every session/user — not scoped to the caller's
        own session like get_messages is. Each assistant message is
        paired with the user message immediately before it in the same
        session (the question it answered), found via a correlated
        subquery rather than a join, since a plain join on "same
        session" would match every prior user message in that session,
        not just the one directly preceding this particular answer.
        An assistant message with no preceding user message in its
        session (shouldn't normally happen) is skipped.
        """
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT "
            "  (SELECT u.content FROM messages u "
            "   WHERE u.session_id = a.session_id AND u.role = 'user' AND u.id < a.id "
            "   ORDER BY u.id DESC LIMIT 1) AS question, "
            "  a.content AS answer, "
            "  a.created_at AS answered_at, "
            "  a.rating AS rating "
            "FROM messages a "
            "WHERE a.role = 'assistant' "
            "ORDER BY a.id DESC "
            "LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return [
            RecentQuery(question=row[0], answer=row[1], answered_at=row[2], rating=row[3])
            for row in rows
            if row[0] is not None
        ]
