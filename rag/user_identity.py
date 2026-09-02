"""Identifies who is asking a question.

Each session is tied to the user's email address instead of being
fully anonymous, so conversation history in MySQL can be traced back
to a person.
"""

import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    """Basic sanity check: local-part@domain.tld, no whitespace."""
    return bool(EMAIL_PATTERN.match(email.strip()))


def normalize_email(email: str) -> str:
    """Trims whitespace and lowercases so the same person is not
    treated as two different users because of casing or stray spaces.
    """
    return email.strip().lower()
