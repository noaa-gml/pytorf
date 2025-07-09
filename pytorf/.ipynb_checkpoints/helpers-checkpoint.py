"""
Helper utility functions for the pytorf package.
"""

import os
import re
import math
import datetime as pydt
from pathlib import Path
import yaml
import warnings
import numpy as np
from typing import List, Union, Optional, Tuple, Any, Dict

# Import datatable conditionally or where needed
try:
    import datatable as dt
    from datatable import f, isna, Frame
    DT_AVAILABLE = True
except ImportError:
    DT_AVAILABLE = False
    # Define dummy types if datatable not installed, useful for type hinting
    class Frame: pass # Dummy class
    class f: pass      # Dummy class

# --- File/String Helpers ---

def fex(filepath: Union[str, Path]) -> str:
    """Extracts the file extension without the leading dot."""
    return os.path.splitext(filepath)[1][1:]

def sr(text: str, n: int) -> str:
    """Extracts the last n characters of a string."""
    if n <= 0:
        return ""
    return text[-n:]

def obs_out(x: list, y: list) -> list:
    """
    Returns elements not common in both lists (symmetric difference).
    Equivalent to R's setdiff(x,y) U setdiff(y,x).
    """
    set_x = set(x)
    set_y = set(y)
    return sorted(list(set_x.symmetric_difference(set_y)))

# --- Numeric Helpers ---

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

    if freq[0] > np.nanmin(x):
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
    indices = np.clip(indices, 0, len(freq) - 1)

    # Handle values exactly equal to left boundaries if left_open=True? (More complex)
    # For simplicity, this version returns the lower bound corresponding to the found interval index.
    # Note: Nan input results in Nan output if we don't handle it explicitly
    result_freq = np.full_like(x, np.nan, dtype=freq.dtype)
    valid_indices_mask = (indices >= 0) & ~np.isnan(x) # FindInterval returns 0 for < min
    result_freq[valid_indices_mask] = freq[indices[valid_indices_mask]]

    # If rightmost_closed=False, adjust boundaries? Numpy doesn't directly support this combination like R.
    if not rightmost_closed:
        warnings.warn("rightmost_closed=False is not fully implemented like R's findInterval. Behavior might differ at boundaries.")
        # Simple adjustment: if x equals the upper bound, map to the *next* interval's lower bound?

    return result_freq


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

# --- Datatable Helpers ---


