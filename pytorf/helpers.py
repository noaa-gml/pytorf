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

def obs_rbind(dt1: Frame, dt2: Frame, verbose: bool = True) -> Optional[Frame]:
    """
    Binds two datatable Frames by row, aligning columns.

    Args:
        dt1: First datatable Frame.
        dt2: Second datatable Frame.
        verbose: If True, print information about common names and type mismatches.

    Returns:
        A combined datatable Frame, or None if errors occur.
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_rbind.")
    if not isinstance(dt1, Frame) or not isinstance(dt2, Frame):
        raise TypeError("Inputs must be datatable Frames.")

    names1 = set(dt1.names)
    names2 = set(dt2.names)
    common_names = sorted(list(names1.intersection(names2)))
    all_names = sorted(list(names1.union(names2)))

    if verbose:
        print("Identifying common names...")
        print(f"Common: {common_names}")
        print(f"Only in dt1: {sorted(list(names1 - names2))}")
        print(f"Only in dt2: {sorted(list(names2 - names1))}")

    # Check for type mismatches in common columns (basic check)
    mismatched_types = []
    if verbose: print("\nComparing classes for common columns:")
    for name in common_names:
        type1 = dt1.stypes[name]
        type2 = dt2.stypes[name]
        if type1 != type2:
            mismatched_types.append((name, type1, type2))
            if verbose: print(f"- {name}: {type1} vs {type2} <-- MISMATCH")
        # else:
        #     if verbose: print(f"- {name}: {type1} (OK)")

    if mismatched_types:
        warnings.warn(f"Type mismatches found for columns: {[m[0] for m in mismatched_types]}. rbind might coerce types or fail.")

    try:
        # datatable's rbind handles column alignment and fills missing with NA
        combined_dt = dt.rbind(dt1, dt2, force=True) # force=True might help with type coercion
        return combined_dt
    except Exception as e:
        print(f"Error during rbind: {e}")
        # Optional: Implement manual alignment as a fallback if needed,
        # similar to the fallback logic in obs_read/obs_read_nc,
        # but rbind itself is generally robust.
        return None

# --- CSVY Helpers ---

def obs_write_csvy(
    dt_frame: Frame,
    notes: List[str],
    out: Union[str, Path],
    sep: str = ",",
    # nchar_max: int = 80, # Less relevant for Python structure printing
    **kwargs
):
    """
    Writes a datatable Frame to a CSVY file with YAML header.

    Args:
        dt_frame: The datatable Frame to write.
        notes: A list of strings to include as notes in the YAML header.
        out: The output file path.
        sep: The CSV separator.
        **kwargs: Additional arguments passed to datatable's `to_csv`.
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_write_csvy.")
    if not isinstance(dt_frame, Frame):
        raise TypeError("Input dt_frame must be a datatable Frame.")

    out_path = Path(out)
    metadata = {
        'name': 'Metadata',
        'notes': notes,
        'structure': {
            'rows': dt_frame.nrows,
            'columns': dt_frame.ncols,
            'names': dt_frame.names,
            'types': [str(t) for t in dt_frame.stypes] # Store datatable stypes
        },
        'generated_by': 'rtorf Python package',
        'timestamp': pydt.datetime.now(pydt.timezone.utc).isoformat()
    }

    try:
        with out_path.open('w', encoding='utf-8') as f:
            f.write("---\n")
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, indent=2)
            f.write("---\n")

        # Append the data using datatable's efficient writer
        dt_frame.to_csv(str(out_path), sep=sep, header=True, append=True, **kwargs)

    except Exception as e:
        print(f"Error writing CSVY file {out_path}: {e}")


def obs_read_csvy(f: Union[str, Path], n_header_lines: int = 100, **kwargs) -> Tuple[Optional[Dict], Optional[Frame]]:
    """
    Reads a CSVY file, prints YAML header, and returns header dict and data frame.

    Args:
        f: Path to the CSVY file.
        n_header_lines: Max lines to search for YAML delimiters.
        **kwargs: Additional arguments passed to datatable's `fread`.

    Returns:
        A tuple containing: (YAML header as dict or None, data as datatable Frame or None).
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_read_csvy.")

    f_path = Path(f)
    yaml_content = []
    in_yaml = False
    header_data = None
    dt_frame = None
    skip_rows = 0 # 0-based line number *after* the second '---'

    try:
        with f_path.open('r', encoding='utf-8') as file:
            found_delimiters = 0
            for i, line in enumerate(file):
                current_line_num = i + 1 # 1-based line number
                if line.strip() == '---':
                    found_delimiters += 1
                    if found_delimiters == 1:
                        in_yaml = True
                    elif found_delimiters == 2:
                        skip_rows = current_line_num # Next line is data header/start
                        break # Found end of YAML
                elif in_yaml:
                    yaml_content.append(line)

                if current_line_num >= n_header_lines * 2: # Safety break
                    warnings.warn(f"YAML delimiters '---' not found within expected lines in {f_path}. Reading might be incorrect.")
                    # Attempt to read from beginning if delimiters not found? Risky.
                    skip_rows = 0 # Assume no header if delimiters not found
                    yaml_content = [] # Discard potentially partial YAML
                    break
            else: # Loop finished without break (EOF before second '---')
                if found_delimiters == 1:
                     warnings.warn(f"YAML end delimiter '---' not found in {f_path}. Reading might be incorrect.")
                     skip_rows = current_line_num # Read after the last line scanned
                     yaml_content = [] # Discard partial YAML


        if yaml_content:
            try:
                # Use safe_load for security
                header_data = yaml.safe_load("".join(yaml_content))
                print("--- YAML Header ---")
                print("".join(yaml_content).strip())
                print("-------------------")
            except yaml.YAMLError as ye:
                 warnings.warn(f"Error parsing YAML header in {f_path}: {ye}")
                 header_data = {"error": "YAML parsing failed", "raw": "".join(yaml_content)}
                 print("--- Raw Header ---")
                 print("".join(yaml_content).strip())
                 print("------------------")

        # Read the data part using datatable, skipping the YAML header
        # skip_to_line expects 1-based line number
        dt_frame = dt.fread(str(f_path), skip_to_line=skip_rows + 1, **kwargs)

    except FileNotFoundError:
        print(f"Error: File not found {f_path}")
    except Exception as e:
        print(f"Error reading CSVY file {f_path}: {e}")

    return header_data, dt_frame

# --- Formatting Helper ---

def obs_format(
    dt_frame: Frame,
    spf: List[str] = ["month", "day", "hour", "minute", "second",
                     "month_end", "day_end", "hour_end", "minute_end", "second_end"],
    spffmt: str = "{:02d}", # Python format spec mini-language
    rnd: List[str] = ["latitude", "longitude"],
    rndn: int = 4,
    spfrnd: bool = True, # Apply sprintf-style formatting after rounding?
    spf_rnd_fmt: str = "{:.4f}", # Format for rounded columns if spfrnd=True
    out: Optional[Union[str, Path]] = None,
    **kwargs
) -> Frame:
    """
    Formats specific columns in a datatable Frame (rounding, padding).

    Args:
        dt_frame: The input datatable Frame.
        spf: List of columns to format using `spffmt` (typically integers needing padding).
        spffmt: Python format string specifier (e.g., '{:02d}' for 2-digit padding).
        rnd: List of columns to round using `rndn` decimals.
        rndn: Number of decimal places for rounding.
        spfrnd: If True, format the rounded columns using `spf_rnd_fmt`.
        spf_rnd_fmt: Python format string specifier for rounded columns (e.g., '{:.4f}').
        out: Optional output path to save the formatted Frame using `fwrite`.
        **kwargs: Additional arguments for `fwrite`.

    Returns:
        The modified datatable Frame (changes are done in-place).
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_format.")
    if not isinstance(dt_frame, Frame):
        raise TypeError("Input dt_frame must be a datatable Frame.")

    # Format columns in spf (padding integers)
    for col_name in spf:
        if col_name in dt_frame.names:
            try:
                # Ensure column is integer-like first, handle NAs
                temp_list = [spffmt.format(int(x)) if x is not None and not math.isnan(x) else None
                             for x in dt_frame[col_name].to_list()[0]]
                # Update column - will likely become string type
                dt_frame[:, update(**{col_name: dt.Frame(temp_list)[f[0]]})]
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not apply format '{spffmt}' to column '{col_name}': {e}. Skipping.")
        else:
            warnings.warn(f"Column '{col_name}' not found for formatting. Skipping.")

    # Round columns in rnd
    for col_name in rnd:
        if col_name in dt_frame.names:
            try:
                # datatable doesn't have a direct round function easily usable in update
                # Extract, round using numpy/math, then update
                col_data = dt_frame[col_name].to_numpy()
                # Use np.round for potentially better handling of different types/NAs
                rounded_data = np.round(col_data.astype(float), rndn) # Convert to float first

                if spfrnd:
                     # Format after rounding
                     formatted_list = [spf_rnd_fmt.format(x) if x is not None and not np.isnan(x) else None
                                       for x in rounded_data]
                     # Update column - will likely become string type
                     dt_frame[:, update(**{col_name: dt.Frame(formatted_list)[f[0]]})]
                else:
                     # Update with rounded numeric data
                     dt_frame[:, update(**{col_name: dt.Frame(rounded_data)[f[0]]})]

            except Exception as e: # Catch broader errors during numpy conversion/rounding
                 warnings.warn(f"Could not round or format column '{col_name}': {e}. Skipping.")
        else:
            warnings.warn(f"Column '{col_name}' not found for rounding. Skipping.")

    # Save if output path provided
    if out:
        try:
            out_path = Path(out)
            # Note: fwrite doesn't exist in datatable. Use to_csv.
            dt_frame.to_csv(str(out_path), **kwargs)
        except Exception as e:
            print(f"Error saving formatted data to {out}: {e}")

    return dt_frame # Return the modified frame