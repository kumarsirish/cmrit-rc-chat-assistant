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

from rag.config import DatabaseConfig


@dataclass
class Message:
    role: str
    content: str
    created_at: datetime


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
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.commit()
        cursor.close()
        connection.close()

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

    def add_message(self, session_id: str, role: str, content: str) -> None:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (session_id, role, content, datetime.now()),
        )
        connection.commit()
        cursor.close()
        connection.close()

    def get_messages(self, session_id: str) -> list:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = %s ORDER BY id ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return [Message(role=row[0], content=row[1], created_at=row[2]) for row in rows]

    def delete_session(self, session_id: str) -> None:
        """Removes a session and its messages, used when a session expires."""
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        connection.commit()
        cursor.close()
        connection.close()
