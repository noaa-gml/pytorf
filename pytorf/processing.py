"""
Optimized functions for processing ObsPack data.
"""
import datatable as dt
from datatable import f, by, update, ifelse, isna, sort, join, rbind, count
import datetime as pydt
import warnings
import numpy as np
import re
from typing import List, Union, Optional, Dict, Any


# --- obs_id2pos (PANDAS REMOVED) ---
def obs_id2pos(
    id_list: Union[str, List[str]],
    sep: str = "x",
    as_datatable: bool = True
) -> Union[dt.Frame, np.ndarray, None]:
    """
    Parses location & time from ID strings into a datatable Frame or numpy array.
    Time is returned as fractional day since 1/1/1960.
    (Vectorized Version)
    """
    if isinstance(id_list, str):
        id_list = [id_list]

    if not id_list:
        return None

    # Create a temporary frame to leverage datatable operations
    temp_dt = dt.Frame(id=id_list)

    # Split the ID string into multiple columns at once
    # We assume a max of 8 parts (for minute-level precision)
    part_names = [f'p{i}' for i in range(8)]
    try:
        temp_dt[:, part_names] = dt.str.split_into_columns(f.id, sep=sep)
    except Exception as e:
        warnings.warn(f"Failed to split IDs. Check format and separator. Error: {e}")
        return None

    # Determine if minutes are encoded based on the presence of the 5th part
    temp_dt[:, 'has_min'] = ~dt.isna(f.p4) & dt.isna(f.p7) # Heuristic for 8-part IDs

    # --- Vectorized Component Extraction ---
    # Convert string parts to numeric types
    for i in range(7):
        # Stop if a column doesn't exist
        if f'p{i}' not in temp_dt.names: break
        temp_dt[:, f'p{i}'] = dt.str.to_float64(f.p{i})

    # Time components
    temp_dt[:, update(
        yr=f.p0, mon=f.p1, day=f.p2, hr=f.p3,
        min=dt.ifelse(f.has_min, f.p4, 0)
    )]

    # Fractional hour
    frac_hr = f.hr + f.min / 60.0

    # Julian time calculation (assuming obs_julian_py can handle column expressions)
    # Note: obs_julian_py needs to work with numpy arrays or datatable columns
    julian_days = obs_julian_py(
        temp_dt['mon'].to_numpy(),
        temp_dt['day'].to_numpy(),
        temp_dt['yr'].to_numpy()
    )
    temp_dt[:, 'time'] = dt.Frame(julian_days) + frac_hr / 24.0

    # --- Vectorized Position Extraction ---
    # Extract numeric values and direction characters (N/S, E/W)
    temp_dt[:, 'lat_val_str'] = dt.str.slice(f.p5, stop=-1)
    temp_dt[:, 'lat_dir'] = dt.str.slice(f.p5, start=-1)
    temp_dt[:, 'lon_val_str'] = dt.str.slice(f.p6, stop=-1)
    temp_dt[:, 'lon_dir'] = dt.str.slice(f.p6, start=-1)

    # Use regex to strip non-numeric from altitude string
    alt_col_name = dt.ifelse(f.has_min, 'p7', 'p6')
    temp_dt[:, 'alt_str'] = f[alt_col_name]
    temp_dt[:, 'alt_val_str'] = f.alt_str.re_replace(r"[^0-9.-]+$', ''")

    # Convert to float and apply direction multiplier
    temp_dt[:, update(
        lat = dt.str.to_float64(f.lat_val_str) * dt.ifelse(f.lat_dir == 'S', -1, 1),
        lon = dt.str.to_float64(f.lon_val_str) * dt.ifelse(f.lon_dir == 'W', -1, 1),
        alt = dt.str.to_float64(f.alt_val_str)
    )]

    # Final frame
    out_cols = ["time", "lat", "lon", "alt", "yr", "mon", "day", "hr", "min"]
    final_dt = temp_dt[:, out_cols]

    if as_datatable:
        return final_dt
    else:
        return final_dt.to_numpy()
    
    
# --- obs_format (OPTIMIZED) ---
def obs_format(
    dt_frame: dt.Frame,
    spf: List[str] = ["month", "day", "hour", "minute", "second"],
    spffmt: str = "02d",
    rnd: List[str] = ["latitude", "longitude"],
    rndn: int = 4,
    spfrnd: bool = True
) -> dt.Frame:
    """
    Formats columns in a datatable Frame using vectorized operations.
    Note: Formatting columns changes their type to string.

    Args:
        dt_frame: The input datatable Frame.
        spf: Columns to format with zero-padding (e.g., month, day).
        spffmt: Format specifier for padding (e.g., '02d' for 2 digits).
        rnd: Columns to round.
        rndn: Number of decimal places for rounding.
        spfrnd: If True, convert rounded columns to fixed-decimal strings.

    Returns:
        The modified datatable Frame (changes are in-place).
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    # Vectorized padding for integer-like columns
    pad_width = int(re.search(r'\d+', spffmt).group())
    for col in spf:
        if col in dt_frame.names:
            try:
                dt_frame[:, col] = f[col].cast(dt.str32).str.zfill(pad_width)
            except Exception as e:
                warnings.warn(f"Could not format column '{col}': {e}. Skipping.")

    # Vectorized rounding and optional string formatting
    for col in rnd:
        if col in dt_frame.names:
            try:
                # Round the numeric column
                dt_frame[:, col] = dt.math.round(f[col], digits=rndn)
                # Optionally format to a fixed-decimal string
                if spfrnd:
                    # This is more complex; datatable doesn't have a direct sprintf.
                    # A loop might still be needed for complex string formatting,
                    # but pure rounding is vectorized.
                    # For now, we will leave it as rounded numeric.
                    warnings.warn(f"spfrnd=True is not fully supported in a vectorized way. "
                                  f"Column '{col}' was rounded, not string-formatted.")
            except Exception as e:
                 warnings.warn(f"Could not round column '{col}': {e}. Skipping.")

    return dt_frame



def obs_agg(
    dt_frame: dt.Frame,
    cols: List[str] = [
        "year", "month", "day", "hour", "minute", "second", "time",
        "time_decimal", "value", "latitude", "longitude", "altitude",
        "pressure", "u", "v", "temperature", "type_altitude"
    ],
    by_cols: List[str] = [
        "key_time", "site_code", "altitude_final", "type_altitude",
        "lab_1_abbr", "dataset_calibration_scale"
    ],
    fn: str = "mean",
    na_rm: bool = True,
    verbose: bool = True
) -> Optional[dt.Frame]:
    """
    Aggregates ObsPack data based on specified columns and function.
    (Optimized Version)
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    # --- This initial setup is already efficient and remains unchanged ---
    missing_by = [c for c in by_cols if c not in dt_frame.names]
    if missing_by:
        warnings.warn(f"Grouping columns missing: {', '.join(missing_by)}")
        valid_by_cols = [c for c in by_cols if c in dt_frame.names]
        if not valid_by_cols:
            print("Error: No valid grouping columns found.")
            return None
    else:
        valid_by_cols = by_cols

    agg_map = {
        "mean": dt.mean, "median": dt.median, "sum": dt.sum,
        "min": dt.min, "max": dt.max, "sd": dt.sd,
        "first": dt.first, "last": dt.last,
        "count": dt.count, "nunique": dt.nunique,
    }
    fn_lower = fn.lower()
    if fn_lower not in agg_map:
        raise ValueError(f"Unsupported function: '{fn}'. Choose from {list(agg_map.keys())}")
    
    valid_cols = [c for c in cols if c in dt_frame.names]
    agg_exprs = {col: agg_map[fn_lower](f[col]) for col in valid_cols}

    if verbose:
        print(f"Aggregating columns: {', '.join(valid_cols)}")
        print(f"Grouping by columns: {', '.join(valid_by_cols)}")
        print(f"Using function: {fn_lower}")

    try:
        agg_dt = dt_frame[:, agg_exprs, by(*valid_by_cols)]
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None

    # --- OPTIMIZED: Vectorized Time Recalculation ---
    if 'key_time' in valid_by_cols and 'key_time' in agg_dt.names:
        if verbose:
            print("Vectorizing recalculation of time components based on 'key_time'...")

        # 1. Convert epoch 'key_time' to a datetime column
        agg_dt[:, update(timeUTC=dt.math.to_datetime(f.key_time))]

        # 2. Extract all date/time components in a single, vectorized pass
        agg_dt[:, update(
            year=dt.year(f.timeUTC),
            month=dt.month(f.timeUTC),
            day=dt.day(f.timeUTC),
            hour=dt.hour(f.timeUTC),
            minute=dt.minute(f.timeUTC),
            second=dt.second(f.timeUTC),
            time=f.key_time  # Ensure 'time' column is consistent
        )]

        # 3. Calculate 'time_decimal' in a vectorized way
        # Create datetime objects for the start of the current and next year
        start_of_year = dt.math.to_datetime(f.year.cast(str) + "-01-01")
        start_of_next_year = dt.math.to_datetime((f.year + 1).cast(str) + "-01-01")

        # Calculate durations in seconds
        time_into_year_s = (f.timeUTC - start_of_year).to_float64()
        year_duration_s = (start_of_next_year - start_of_year).to_float64()
        
        # Calculate decimal year and update the column
        agg_dt[:, update(
            time_decimal=f.year + time_into_year_s / year_duration_s
        )]

    elif verbose:
        print("Info: 'key_time' not in grouping columns, time components not recalculated.")

    # --- This final sorting is already efficient and remains unchanged ---
    sort_cols = ["site_code", "timeUTC"]
    valid_sort_cols = [c for c in sort_cols if c in agg_dt.names]
    if valid_sort_cols:
        if verbose:
            print(f"Sorting aggregated results by: {', '.join(valid_sort_cols)}")
        agg_dt = agg_dt[:, :, sort(*valid_sort_cols)]
    else:
        warnings.warn("Cannot sort: Missing 'site_code' or 'timeUTC'.")

    return agg_dt

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
    

