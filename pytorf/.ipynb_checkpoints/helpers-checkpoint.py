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
    from datatable import f, isna, Frame # f needs to be imported for obs_format
    DT_AVAILABLE = True
except ImportError:
    DT_AVAILABLE = False
    # Define dummy types if datatable not installed, useful for type hinting
    class Frame: pass # Dummy class
    class f: pass      # Dummy class

# Assuming pandas is used for obs_id2pos based on pd.DataFrame
try:
    import pandas as pd
except ImportError:
    class pd: # Dummy class if pandas is not available but code uses pd.DataFrame
        @staticmethod
        def DataFrame(*args, **kwargs):
            warnings.warn("pandas is not installed. DataFrame functionality in obs_id2pos will not work as expected.")
            return args[0] if args else {} # Return first arg (expected to be list of dicts) or empty dict

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
        with out_path.open('w', encoding='utf-8') as f_obj: # Renamed f to f_obj to avoid conflict
            f_obj.write("---\n")
            yaml.dump(metadata, f_obj, default_flow_style=False, sort_keys=False, indent=2)
            f_obj.write("---\n")

        # Append the data using datatable's efficient writer
        dt_frame.to_csv(str(out_path), sep=sep, header=True, append=True, **kwargs)

    except Exception as e:
        print(f"Error writing CSVY file {out_path}: {e}")


def obs_read_csvy(f_path_in: Union[str, Path], n_header_lines: int = 100, **kwargs) -> Tuple[Optional[Dict], Optional[Frame]]:
    """
    Reads a CSVY file, prints YAML header, and returns header dict and data frame.

    Args:
        f_path_in: Path to the CSVY file (renamed from f to avoid conflict).
        n_header_lines: Max lines to search for YAML delimiters.
        **kwargs: Additional arguments passed to datatable's `fread`.

    Returns:
        A tuple containing: (YAML header as dict or None, data as datatable Frame or None).
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_read_csvy.")

    f_path = Path(f_path_in)
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
        out: Optional output path to save the formatted Frame using `to_csv`.
        **kwargs: Additional arguments for `to_csv`.

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
                temp_list = [spffmt.format(int(x)) if x is not None and not math.isnan(x) and not (isinstance(x, float) and math.isinf(x)) else None
                             for x in dt_frame[col_name].to_list()[0]]
                # Update column - will likely become string type
                # Assuming `update` and `f` are from `datatable` context.
                dt_frame[:, dt.update(**{col_name: dt.Frame(temp_list)[dt.f[0]]})]
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
                    formatted_list = [spf_rnd_fmt.format(x) if x is not None and not np.isnan(x) and not np.isinf(x) else None
                                      for x in rounded_data]
                    # Update column - will likely become string type
                    dt_frame[:, dt.update(**{col_name: dt.Frame(formatted_list)[dt.f[0]]})]
                else:
                    # Update with rounded numeric data
                    dt_frame[:, dt.update(**{col_name: dt.Frame(rounded_data)[dt.f[0]]})]

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

def obs_footname(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    lat: float,
    lon: float,
    alt: float,
    fullpath: bool = False,
    time: Optional[pydt.datetime] = None # Allow passing datetime object
) -> str:
    """Generates the expected HYSPLIT footprint NetCDF filename."""

    if time:
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute

    # Use obs_trunc if precise R compatibility is needed for coords
    # lat_t = obs_trunc(abs(lat), 4)
    # lon_t = obs_trunc(abs(lon), 4)
    lat_abs = abs(lat) # Use absolute value for formatting
    lon_abs = abs(lon)

    lats = "N" if lat >= 0 else "S"
    lons = "E" if lon >= 0 else "W"

    # Format numbers with leading zeros and specific widths/precision
    s_yr = f"{year:04d}" # Use 4-digit year for clarity/consistency
    s_mo = f"{month:02d}"
    s_dy = f"{day:02d}"
    s_hr = f"{hour:02d}"
    s_mn = f"{minute:02d}"
    # Specific formatting for lat/lon based on R examples/HERA needs
    # Lat: 7 wide, 4 decimal (e.g., 012.3456) - achieved via width
    # Lon: 8 wide, 4 decimal (e.g., 0123.4567) - achieved via width
    s_lat = f"{lat_abs:07.4f}"
    s_lon = f"{lon_abs:08.4f}"
    s_alt = f"{round(alt):05d}" # Round alt to integer first

    basename = f"{s_yr}x{s_mo}x{s_dy}x{s_hr}x{s_mn}x{s_lat}{lats}x{s_lon}{lons}x{s_alt}"

    if fullpath:
        # Assume YYYY/MM structure relative to some base path (handled elsewhere)
        # This function generates the relative part for fullpath=True
        return str(Path(f"{s_yr}") / f"{s_mo}" / f"hysplit{basename}.nc")
    else:
        # Return just the filename base + extension if not full path
        return f"{basename}.nc"

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

def obs_id2pos(id_list: Union[str, List[str]],
               sep: str = "x", # sep is not actually used in current logic due to hardcoded 'x' split
               as_dataframe: bool = False) -> Union[pd.DataFrame, np.ndarray, Dict[str, Any], None]:
    """
    Replicates the R id2pos function to parse identifying labels for
    location & time.

    Returns time as fractional day since 1/1/1960 (using obs_julian_py logic) and
    alt as altitude above ground in meters.

    Args:
        id_list: A single ID string or a list of ID strings.
                 Format examples:
                 '2002x08x03x10x45.00Nx090.00Ex00030' (no minutes)
                 '2002x08x03x10x55x45.335Sx179.884Wx00030' (with minutes)
        sep: The separator character used in the ID string (currently unused
             as 'x' is hardcoded in the split logic based on R code).
        as_dataframe: If True, returns a pandas DataFrame. Otherwise, returns
                      a numpy array (if multiple valid IDs) or a 1D numpy array
                      (if single valid ID). Returns None if no valid IDs processed.

    Returns:
        A pandas DataFrame or numpy array containing the parsed data:
        'time', 'lat', 'lon', 'alt', 'yr', 'mon', 'day', 'hr', 'min'.
        Returns None if input is empty or no IDs could be parsed.

    Raises:
        ValueError: If an ID string has an unexpected format.
        TypeError: If input 'id_list' is not a string or list.
    """
    if isinstance(id_list, str):
        id_list = [id_list] # Process single string as a list of one
        single_input = True
    elif not isinstance(id_list, list):
        raise TypeError("Input 'id_list' must be a string or a list of strings.")
    else:
        single_input = False

    if not id_list:
        return None # Handle empty input list

    results = []

    for id_str in id_list:
        try:
            parts = id_str.split('x')
            num_parts = len(parts)

            encode_minutes = num_parts == 8

            if not (encode_minutes or num_parts == 7):
                raise ValueError(f"ID string '{id_str}' has unexpected number of parts ({num_parts}). Expected 7 or 8.")

            # --- Parse Date and Time ---
            yr4 = int(parts[0])
            mon = int(parts[1])
            day = int(parts[2])
            hr = int(parts[3])

            ipos = 4

            if encode_minutes:
                min_val = int(parts[4])
                frac_hr = hr + min_val / 60.0
                ipos += 1
            else:
                min_val = 0
                frac_hr = float(hr)

            # --- Parse Latitude ---
            lat_str = parts[ipos]
            if not lat_str: raise ValueError(f"Missing latitude part in '{id_str}'")
            lat = float(lat_str[:-1])
            lat_sign = lat_str[-1].upper()
            if lat_sign == "S": lat = -lat
            elif lat_sign != "N": raise ValueError(f"Invalid latitude direction '{lat_sign}' in '{id_str}'")
            ipos += 1

            # --- Parse Longitude ---
            lon_str = parts[ipos]
            if not lon_str: raise ValueError(f"Missing longitude part in '{id_str}'")
            lon = float(lon_str[:-1])
            lon_sign = lon_str[-1].upper()
            if lon_sign == "W": lon = -lon
            elif lon_sign != "E": raise ValueError(f"Invalid longitude direction '{lon_sign}' in '{id_str}'")
            ipos += 1

            # --- Parse Altitude ---
            alt_str = parts[ipos]
            if not alt_str: raise ValueError(f"Missing altitude part in '{id_str}'")
            alt_numeric_str = re.sub(r"[^0-9.-]+$", "", alt_str) # Remove trailing non-numeric chars (like units)
            alt = float(alt_numeric_str)

            # --- Calculate Time (days since epoch using obs_julian_py) ---
            # Call the translated obs_julian function
            days_since = obs_julian_py(mon, day, yr4) # Origin defaults to (1, 1, 1960)

            # Check for NaN or potential errors from date calculation if needed
            # (obs_julian_py currently doesn't explicitly return NaN for invalid dates like the datetime version did)
            # The algorithm might produce unexpected results for invalid dates (e.g., Feb 30)
            # Add a basic date validity check before calling obs_julian_py? Or trust the algorithm?
            # Let's trust the algorithm mimics R for now.

            time_val = days_since + frac_hr / 24.0

            results.append({
                "time": time_val, "lat": lat, "lon": lon, "alt": alt,
                "yr": yr4, "mon": mon, "day": day, "hr": hr, "min": min_val
            })

        except (ValueError, IndexError, TypeError) as e:
            warnings.warn(f"Warning: Skipping ID '{id_str}' due to processing error: {e}")
            continue

    if not results:
        return None

    # --- Format Output ---
    cols_order = ["time", "lat", "lon", "alt", "yr", "mon", "day", "hr", "min"]
    if as_dataframe:
        # This part requires pandas to be imported as pd
        df = pd.DataFrame(results)
        return df[cols_order]
    else:
        arr = np.array([[res[col] for col in cols_order] for res in results], dtype=np.float64) # Ensure float array
        if single_input and arr.shape[0] == 1:
            return arr.flatten() # Return 1D array for single input
        else:
            return arr # Return 2D array