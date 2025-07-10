def obs_roundtime(x: Union[List[pydt.datetime], Any], n: int = 10) -> List[Optional[int]]:
    """
    Rounds the seconds part of datetime objects to the nearest multiple of n.

    Args:
        x: A list of datetime objects or something coercible (e.g., datatable Column).
        n: The factor to round seconds to (e.g., 10).

    Returns:
        A list of rounded second values (integers), or None for invalid inputs.
    """
    rounded_seconds = []
    if DT_AVAILABLE and isinstance(x, dt.Column):
        x = x.to_list()[0] # Extract list if it's a datatable Column

    if not isinstance(x, list):
        try:
            x = list(x) # Try converting iterables
        except TypeError:
            raise TypeError("Input `x` must be a list of datetime objects or coercible.")

    if n <= 0:
        raise ValueError("Rounding factor 'n' must be positive.")

    for dt_obj in x:
        if isinstance(dt_obj, pydt.datetime):
            # R formula: (( second(dt$timeUTC) + 5/2) %/% n)*n
            # Python equivalent: round half up for the division
            seconds = dt_obj.second + dt_obj.microsecond / 1e6
            rounded = int(math.floor(seconds / n + 0.5) * n) # Round half up
            # Handle rounding up to 60
            rounded_seconds.append(0 if rounded >= 60 else rounded)
        else:
            rounded_seconds.append(None) # Append None for non-datetime objects
    return rounded_seconds
