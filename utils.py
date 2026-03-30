"""Utility functions."""


def clean_surrogates(text: str) -> str:
    """Remove surrogate characters that cause UTF-8 encoding errors."""
    return text.encode("utf-8", errors="replace").decode("utf-8")
