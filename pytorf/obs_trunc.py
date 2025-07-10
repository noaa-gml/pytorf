def obs_trunc(n: float, dec: int) -> float:
    """
    Truncates a number to a specified number of decimal places.

    Args:
        n: The number to truncate.
        dec: The number of decimal places to keep.

    Returns:
        The truncated number.

    Note: Uses a small epsilon to handle potential floating point
          representation issues near truncation boundaries. Be cautious
          with precision for very large/small numbers or many decimals.
          Source: https://stackoverflow.com/a/47015304/2418532 adapted.
    """
    if not isinstance(n, (int, float)):
        raise TypeError("Input 'n' must be numeric.")
    if not isinstance(dec, int) or dec < 0:
        raise ValueError("Input 'dec' must be a non-negative integer.")

    if math.isnan(n) or math.isinf(n):
        return n # Return NaN/Inf as is

    multiplier = 10 ** dec
    # Add small epsilon based on sign to push numbers slightly away
    # from the truncation point before integer truncation.
    epsilon = math.copysign(10**-(dec + 5), n)
    # Check for potential overflow before multiplying
    if abs(n * multiplier) > 1e300: # Heuristic check
        warnings.warn("Potential overflow during truncation, result might be inaccurate.")
    return math.trunc((n + epsilon) * multiplier) / multiplier

# --- Time/Frequency Helpers ---

