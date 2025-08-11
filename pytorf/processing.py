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

# --- obs_julian_py (Helper) ---
# This helper is needed by obs_id2pos
def obs_julian_py(m, d, y, origin=(1, 1, 1960)):
    """
    Calculates days since an origin date using a vectorized algorithm.
    """
    m, d, y = np.asarray(m), np.asarray(d), np.asarray(y)
    origin_m, origin_d, origin_y = origin

    def _calculate_raw_jd(m_in, d_in, y_in):
        y_adj = y_in + np.where(m_in > 2, 0, -1)
        m_adj = m_in + np.where(m_in > 2, -3, 9)
        c = y_adj // 100
        ya = y_adj - 100 * c
        jd = (146097 * c) // 4 + (1461 * ya) // 4 + (153 * m_adj + 2) // 5 + d_in + 1721119
        return jd

    days_since_origin = _calculate_raw_jd(m, d, y) - _calculate_raw_jd(origin_m, origin_d, origin_y)
    return days_since_origin.item() if days_since_origin.ndim == 0 else days_since_origin


# --- obs_id2pos (CORRECTED) ---
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

    temp_dt = dt.Frame(id=id_list)

    part_names = [f'p{i}' for i in range(8)]
    try:
        temp_dt[:, part_names] = dt.str.split_into_columns(f.id, sep=sep)
    except Exception as e:
        warnings.warn(f"Failed to split IDs. Check format and separator. Error: {e}")
        return None

    temp_dt[:, 'has_min'] = ~dt.isna(f.p4) & ~dt.isna(f.p7)

    # --- Vectorized Component Extraction (CORRECTED) ---
    for i in range(8): # Iterate through all possible parts
        col_name = f'p{i}'
        if col_name in temp_dt.names:
            # Use dictionary-style access f['col_name'] for dynamic columns
            temp_dt[:, col_name] = dt.str.to_float64(f[col_name])

    temp_dt[:, update(
        yr=f.p0, mon=f.p1, day=f.p2, hr=f.p3,
        min=dt.ifelse(f.has_min, f.p4, 0)
    )]

    frac_hr = f.hr + f.min / 60.0

    julian_days = obs_julian_py(
        temp_dt['mon'].to_numpy(),
        temp_dt['day'].to_numpy(),
        temp_dt['yr'].to_numpy()
    )
    temp_dt[:, 'time'] = dt.Frame(julian_days) + frac_hr / 24.0

    # --- Vectorized Position Extraction (CORRECTED) ---
    temp_dt[:, 'lat_val_str'] = dt.str.slice(f.p5, stop=-1)
    temp_dt[:, 'lat_dir'] = dt.str.slice(f.p5, start=-1)
    temp_dt[:, 'lon_val_str'] = dt.str.slice(f.p6, stop=-1)
    temp_dt[:, 'lon_dir'] = dt.str.slice(f.p6, start=-1)

    # Correctly select altitude string based on has_min
    temp_dt[:, 'alt_str'] = dt.ifelse(f.has_min, f.p7, f.p6).cast(dt.str32)
    # Corrected regex replacement
    temp_dt[:, 'alt_val_str'] = f.alt_str.re_replace(r"[^0-9.-]+$", "")


    temp_dt[:, update(
        lat = dt.str.to_float64(f.lat_val_str) * dt.ifelse(f.lat_dir == 'S', -1, 1),
        lon = dt.str.to_float64(f.lon_val_str) * dt.ifelse(f.lon_dir == 'W', -1, 1),
        alt = dt.str.to_float64(f.alt_val_str)
    )]

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
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    pad_width = int(re.search(r'\d+', spffmt).group())
    for col in spf:
        if col in dt_frame.names:
            try:
                dt_frame[:, col] = f[col].cast(dt.str32).str.zfill(pad_width)
            except Exception as e:
                warnings.warn(f"Could not format column '{col}': {e}. Skipping.")

    for col in rnd:
        if col in dt_frame.names:
            try:
                dt_frame[:, col] = dt.math.round(f[col], digits=rndn)
                if spfrnd:
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

    if 'key_time' in valid_by_cols and 'key_time' in agg_dt.names:
        if verbose:
            print("Vectorizing recalculation of time components based on 'key_time'...")

        agg_dt[:, update(timeUTC=dt.math.to_datetime(f.key_time))]
        agg_dt[:, update(
            year=dt.year(f.timeUTC),
            month=dt.month(f.timeUTC),
            day=dt.day(f.timeUTC),
            hour=dt.hour(f.timeUTC),
            minute=dt.minute(f.timeUTC),
            second=dt.second(f.timeUTC),
            time=f.key_time
        )]

        start_of_year = dt.math.to_datetime(f.year.cast(str) + "-01-01")
        start_of_next_year = dt.math.to_datetime((f.year + 1).cast(str) + "-01-01")

        time_into_year_s = (f.timeUTC - start_of_year).to_float64()
        year_duration_s = (start_of_next_year - start_of_year).to_float64()

        agg_dt[:, update(
            time_decimal=f.year + time_into_year_s / year_duration_s
        )]

    elif verbose:
        print("Info: 'key_time' not in grouping columns, time components not recalculated.")

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
    """
    if not isinstance(freq, np.ndarray):
        freq = np.array(freq)
    if not np.all(np.diff(freq) >= 0):
        raise ValueError("'freq' must be sorted non-decreasingly.")
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if len(freq) > 0 and freq[0] > np.nanmin(x):
        warnings.warn("First element of 'freq' is greater than the minimum of 'x'. Values below freq[0] will be mapped to NaN or the first interval depending on flags.")

    indices = np.searchsorted(freq, x, side='right')
    indices = np.clip(indices, 0, len(freq) - 1 if len(freq) > 0 else 0)
    result_freq = np.full_like(x, np.nan, dtype=freq.dtype if len(freq) > 0 else float)
    if len(freq) > 0:
        valid_indices_mask = (indices >= 0) & ~np.isnan(x)
        valid_freq_indices = indices[valid_indices_mask]
        valid_freq_indices = np.clip(valid_freq_indices, 0, len(freq) - 1)
        result_freq[valid_indices_mask] = freq[valid_freq_indices]

    if not rightmost_closed:
        warnings.warn("rightmost_closed=False is not fully implemented like R's findInterval. Behavior might differ at boundaries.")

    return result_freq

def obs_rbind(dt1: dt.Frame, dt2: dt.Frame, verbose: bool = True) -> Optional[dt.Frame]:
    """
    Binds two datatable Frames by row, aligning columns.
    """
    if not isinstance(dt1, dt.Frame) or not isinstance(dt2, dt.Frame):
        raise TypeError("Inputs must be datatable Frames.")

    names1 = set(dt1.names)
    names2 = set(dt2.names)
    common_names = sorted(list(names1.intersection(names2)))

    if verbose:
        print("Identifying common names...")
        print(f"Common: {common_names}")
        print(f"Only in dt1: {sorted(list(names1 - names2))}")
        print(f"Only in dt2: {sorted(list(names2 - names1))}")

    mismatched_types = []
    if verbose: print("\nComparing classes for common columns:")
    for name in common_names:
        type1 = dt1.stypes[name]
        type2 = dt2.stypes[name]
        if type1 != type2:
            mismatched_types.append((name, type1, type2))
            if verbose: print(f"- {name}: {type1} vs {type2} <-- MISMATCH")

    if mismatched_types:
        warnings.warn(f"Type mismatches found for columns: {[m[0] for m in mismatched_types]}. rbind might coerce types or fail.")

    try:
        combined_dt = dt.rbind(dt1, dt2, force=True)
        return combined_dt
    except Exception as e:
        print(f"Error during rbind: {e}")
        return None