from rag.user_identity import is_valid_email, normalize_email


def test_valid_email_is_accepted():
    assert is_valid_email("jane.doe@cmrit.ac.in") is True


def test_missing_at_sign_is_rejected():
    assert is_valid_email("jane.doe.cmrit.ac.in") is False


def test_missing_domain_dot_is_rejected():
    assert is_valid_email("jane.doe@cmrit") is False


def test_email_with_whitespace_is_rejected():
    assert is_valid_email("jane doe@cmrit.ac.in") is False


def test_email_with_surrounding_spaces_is_still_valid():
    assert is_valid_email("  jane.doe@cmrit.ac.in  ") is True


def test_normalize_email_trims_and_lowercases():
    assert normalize_email("  Jane.Doe@CMRIT.AC.IN  ") == "jane.doe@cmrit.ac.in"
