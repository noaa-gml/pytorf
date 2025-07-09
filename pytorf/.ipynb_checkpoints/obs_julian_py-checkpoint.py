def obs_julian_py(m, d, y, origin=None):
    """
    Calculates days since origin date using the algorithm from R's 'chron' package,
    likely based on the "S book (p.269)".

    Args:
        m: Month (scalar or array-like).
        d: Day (scalar or array-like).
        y: Year (scalar or array-like).
        origin: A tuple or list representing the origin date (month, day, year).
                If None, defaults to (1, 1, 1960).

    Returns:
        The number of days elapsed between the origin date and the input date(s).
        Returns a float or a NumPy array of floats.
    """
    # Default origin if not provided
    if origin is None:
        origin = (1, 1, 1960) # month, day, year

    # Ensure inputs are numpy arrays for vectorized operations
    m = np.asarray(m)
    d = np.asarray(d)
    y = np.asarray(y)
    origin_m, origin_d, origin_y = origin

    # Internal function to calculate the raw 'Julian Day' number using the S book algorithm
    def _calculate_raw_jd(m_in, d_in, y_in):
        # Ensure inputs inside are arrays
        m_in = np.asarray(m_in)
        d_in = np.asarray(d_in)
        y_in = np.asarray(y_in)

        # Algorithm from R code (S book p.269)
        y_adj = y_in + np.where(m_in > 2, 0, -1)
        m_adj = m_in + np.where(m_in > 2, -3, 9)
        c = y_adj // 100
        ya = y_adj - 100 * c
        # Perform calculations using integer arithmetic where R uses %/%
        jd = (146097 * c) // 4 + (1461 * ya) // 4 + (153 * m_adj + 2) // 5 + d_in + 1721119
        return jd

    # Calculate the raw JD for the input date(s)
    jd_inputs = _calculate_raw_jd(m, d, y)

    # Calculate the raw JD for the origin date
    jd_origin = _calculate_raw_jd(origin_m, origin_d, origin_y)

    # The result is the difference
    days_since_origin = jd_inputs - jd_origin

    # R function returns a simple numeric vector/value
    # If the input was scalar, numpy might return a 0-dim array, convert back
    if days_since_origin.ndim == 0:
        return days_since_origin.item() # Return scalar float
    return days_since_origin # Return numpy array