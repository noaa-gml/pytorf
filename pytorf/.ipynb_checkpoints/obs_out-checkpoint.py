def obs_out(x: list, y: list) -> list:
    """
    Returns elements not common in both lists (symmetric difference).
    Equivalent to R's setdiff(x,y) U setdiff(y,x).
    """
    set_x = set(x)
    set_y = set(y)
    return sorted(list(set_x.symmetric_difference(set_y)))
