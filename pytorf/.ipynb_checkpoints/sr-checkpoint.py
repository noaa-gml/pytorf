def sr(text: str, n: int) -> str:
    """Extracts the last n characters of a string."""
    if n <= 0:
        return ""
    return text[-n:]
