def obs_freq(x: Union[List[float], np.ndarray], freq: Union[List[float], np.ndarray], rightmost_closed: bool = True, left_open: bool = False) -> np.ndarray:
    """
    Assigns elements of x to the interval defined by freq they fall into.
    Similar to findInterval in R.

    Args:
        x: Numeric vector or list of values.
        freq: Numeric vector or list defining interval boundaries (must be sorted).
        rightmost_closed: If True, interval is closed on the right.
        left_open: If True, interval is open on the left. (Note: findInterval in R has slightly different options)

    Returns:
        A numpy array where each element is the lower bound of the interval
        from 'freq' that the corresponding element of 'x' falls into.
    """
    if not isinstance(freq, np.ndarray):
        freq = np.array(freq)
    if not np.all(np.diff(freq) >= 0):
        raise ValueError("'freq' must be sorted non-decreasingly.")
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if len(freq) > 0 and freq[0] > np.nanmin(x): # Add len(freq) > 0 check
        warnings.warn("First element of 'freq' is greater than the minimum of 'x'. Values below freq[0] will be mapped to NaN or the first interval depending on flags.")
        # R findInterval maps these to 0, which corresponds to index before first element.
        # np.searchsorted gives index where element would be inserted.

    # numpy.searchsorted finds indices where elements should be inserted to maintain order.
    # side='right' means insertion point is to the right (like upper bound)
    # side='left' means insertion point is to the left (like lower bound)

    # R's findInterval(x, vec, rightmost.closed = T, left.open = T) -> (left, right]
    # R's findInterval(x, vec, rightmost.closed = F, left.open = F) -> [left, right) (default)

    # Translating R's logic is tricky. Let's mimic rightmost_closed=T behavior:
    # We want the index `i` such that vec[i-1] < x <= vec[i]
    # `np.searchsorted(freq, x, side='right')` gives insertion index `idx`
    # If x == freq[k], side='right' gives k+1.
    # If freq[k-1] < x < freq[k], side='right' gives k.
    indices = np.searchsorted(freq, x, side='right')

    # Values exactly equal to a boundary might need adjustment depending on flags
    # Values below the first boundary: searchsorted gives 0. findInterval gives 0.
    # Values above the last boundary: searchsorted gives len(freq). findInterval gives len(freq)-1.

    # Correct for values above the last boundary
    indices = np.clip(indices, 0, len(freq) - 1 if len(freq) > 0 else 0) # Adjust clip if freq is empty

    # Handle values exactly equal to left boundaries if left_open=True? (More complex)
    # For simplicity, this version returns the lower bound corresponding to the found interval index.
    # Note: Nan input results in Nan output if we don't handle it explicitly
    result_freq = np.full_like(x, np.nan, dtype=freq.dtype if len(freq) > 0 else float) # Handle empty freq dtype
    if len(freq) > 0:
        valid_indices_mask = (indices >= 0) & ~np.isnan(x) # FindInterval returns 0 for < min
        # Ensure indices are within bounds of freq before assignment
        valid_freq_indices = indices[valid_indices_mask]
        # Clip again to be absolutely sure, though previous clip should handle it
        valid_freq_indices = np.clip(valid_freq_indices, 0, len(freq) - 1)
        result_freq[valid_indices_mask] = freq[valid_freq_indices]


    # If rightmost_closed=False, adjust boundaries? Numpy doesn't directly support this combination like R.
    if not rightmost_closed:
        warnings.warn("rightmost_closed=False is not fully implemented like R's findInterval. Behavior might differ at boundaries.")
        # Simple adjustment: if x equals the upper bound, map to the *next* interval's lower bound?

    return result_freq


