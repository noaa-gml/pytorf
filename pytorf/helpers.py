"""
Optimized helper utility functions for the pytorf package.
"""
import datatable as dt
from datatable import f, ifelse, isna, update, Frame
import os
import math
import re
import datetime as pydt
from pathlib import Path
import warnings
import numpy as np
from typing import List, Union, Optional, Any

# --- File/String Helpers ---
def obs_roundtime(dt_frame: Frame, time_col: str = 'timeUTC', n: int = 10) -> Frame:
    """
    Rounds the seconds of a datetime column in a datatable Frame to the
    nearest multiple of n, using vectorized operations.
    """
    if not isinstance(dt_frame, Frame):
        raise TypeError("Input must be a datatable Frame.")
    if time_col not in dt_frame.names:
        raise ValueError(f"Column '{time_col}' not found in Frame.")
    if n <= 0:
        raise ValueError("Rounding factor 'n' must be positive.")

    dt_frame[:, update(
        __secs_float=(
            dt.second(f[time_col]) + dt.microsecond(f[time_col]) / 1e6
        )
    )]
    dt_frame[:, update(
        __rounded_val=(dt.math.floor(f.__secs_float / n + 0.5) * n)
    )]

    new_col_name = f"{time_col}_rounded"
    dt_frame[:, new_col_name] = f[time_col] + dt.timedelta(seconds=f.__rounded_val - f.__secs_float)

    del dt_frame[:, ['__secs_float', '__rounded_val']]
    
    return dt_frame

def fex(filepath: Union[str, Path]) -> str:
    """Extracts the file extension without the leading dot."""
    return os.path.splitext(filepath)[1][1:]

def sr(text: str, n: int) -> str:
    """Extracts the last n characters of a string."""
    return text[-n:] if n > 0 else ""

def obs_out(x: list, y: list) -> list:
    """
    Returns elements not common in both lists (symmetric difference).
    """
    set_x = set(x)
    set_y = set(y)
    return sorted(list(set_x.symmetric_difference(set_y)))

def obs_trunc(n: float, dec: int) -> float:
    """
    Truncates a number to a specified number of decimal places.
    """
    if not isinstance(n, (int, float)):
        raise TypeError("Input 'n' must be numeric.")
    if not isinstance(dec, int) or dec < 0:
        raise ValueError("Input 'dec' must be a non-negative integer.")

    if math.isnan(n) or math.isinf(n):
        return n

    multiplier = 10 ** dec
    epsilon = math.copysign(10**-(dec + 5), n)
    
    if abs(n * multiplier) > 1e300:
        warnings.warn("Potential overflow during truncation, result might be inaccurate.")
    return math.trunc((n + epsilon) * multiplier) / multiplier

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


def obs_addtime(
    dt_frame: dt.Frame,
    verbose: bool = True,
    tz_str: str = "UTC"
) -> dt.Frame:
    """
    Adds datetime columns based on epoch time columns (time, start_time).
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")
    if tz_str != "UTC":
         warnings.warn("Timezone handling currently assumes UTC for epoch conversions.")

    if verbose: print("Adding timeUTC (from 'time') and timeUTC_start (from 'start_time')")
    if 'time' in dt_frame.names:
        dt_frame[:, update(timeUTC=dt.math.to_datetime(f.time))]
    if 'start_time' in dt_frame.names:
        dt_frame[:, update(timeUTC_start=dt.math.to_datetime(f.start_time))]

    if verbose: print("Calculating timeUTC_end")
    dt_frame[:, 'timeUTC_end'] = dt.Type.time64

    if 'time_interval' in dt_frame.names and 'timeUTC_start' in dt_frame.names:
        if verbose: print("  Using 'time_interval' where available.")
        dt_frame[:, 'time_interval_td'] = dt.math.to_timedelta(f.time_interval)
        dt_frame[:, 'timeUTC_end'] = f.timeUTC_start + f.time_interval_td
        del dt_frame[:, 'time_interval_td']

    if 'midpoint_time' in dt_frame.names and 'start_time' in dt_frame.names:
        if verbose: print("  Using midpoint calculation for remaining rows.")
        duration_s = 2.0 * (f.midpoint_time - f.start_time)
        duration_td = dt.math.to_timedelta(duration_s)
        dt_frame[:, update(
            timeUTC_end=ifelse(isna(f.timeUTC_end), f.timeUTC_start + duration_td, f.timeUTC_end)
        )]

    if verbose: print("Calculating time differences and end components.")
    if 'timeUTC_start' in dt_frame.names and 'timeUTC_end' in dt_frame.names:
        time_diff = f.timeUTC_end - f.timeUTC_start
        dt_frame[:, 'dif_time_seconds'] = time_diff.to_float64()

    if 'timeUTC_end' in dt_frame.names:
        dt_frame[:, update(
            year_end=dt.year(f.timeUTC_end),
            month_end=dt.month(f.timeUTC_end),
            day_end=dt.day(f.timeUTC_end),
            hour_end=dt.hour(f.timeUTC_end),
            minute_end=dt.minute(f.timeUTC_end),
            second_end=dt.second(f.timeUTC_end)
        )]

    if 'timeUTC' in dt_frame.names and 'timeUTC_end' in dt_frame.names:
        dt_frame[:, 'time_warning'] = ifelse(
            f.timeUTC == f.timeUTC_end, "warning, timeUTC == timeUTC_end", "all good"
        )

    return dt_frame


def obs_addltime(
    dt_frame: dt.Frame,
    time_utc_col: str = "timeUTC",
    utc2lt_col: str = "site_utc2lst",
    longitude_col: str = "longitude",
) -> dt.Frame:
    """
    Calculates approximate local time based on UTC time and longitude or a UTC offset.
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    if time_utc_col not in dt_frame.names:
        raise ValueError(f"Required column '{time_utc_col}' not found.")
    has_utc_offset = utc2lt_col in dt_frame.names
    has_longitude = longitude_col in dt_frame.names

    if not has_utc_offset and not has_longitude:
        warnings.warn(f"Neither '{utc2lt_col}' nor '{longitude_col}' found. Cannot calculate local time.")
        dt_frame[:, update(local_time=dt.obj64, lh=dt.int8)]
        return dt_frame

    dt_frame[:, 'offset_hours'] = dt.float64

    if has_utc_offset and has_longitude:
        dt_frame[:, 'offset_hours'] = ifelse(
            ~isna(f[utc2lt_col]), f[utc2lt_col],
            f[longitude_col] / 15.0
        )
    elif has_utc_offset:
        dt_frame[:, 'offset_hours'] = f[utc2lt_col]
    else:
        dt_frame[:, 'offset_hours'] = f[longitude_col] / 15.0

    offset_seconds = f.offset_hours * 3600.0
    dt_frame[:, 'offset_td'] = dt.math.to_timedelta(offset_seconds)
    dt_frame[:, 'local_time'] = f[time_utc_col] + f.offset_td
    dt_frame[:, 'lh'] = dt.hour(f.local_time)

    del dt_frame[:, ['offset_hours', 'offset_td']]

    return dt_frame


def obs_addstime(
    dt_frame: dt.Frame,
) -> dt.Frame:
    """
    Adds a 'timeUTC_st' column by combining solar time component columns.
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    st_cols = ['year_st', 'month_st', 'day_st', 'hour_st', 'minute_st', 'second_st']
    missing_cols = [c for c in st_cols if c not in dt_frame.names]
    if missing_cols:
        warnings.warn(f"Missing solar time component columns: {', '.join(missing_cols)}. Cannot create 'timeUTC_st'.")
        dt_frame[:, update(timeUTC_st=dt.obj64)]
        return dt_frame

    sec_int = f.second_st.cast(dt.int64)
    usec_val = ((f.second_st - sec_int) * 1_000_000).cast(dt.int64)

    dt_frame[:, update(
        year_str=f.year_st.cast(str),
        month_str=f.month_st.cast(str).str.zfill(2),
        day_str=f.day_st.cast(str).str.zfill(2),
        hour_str=f.hour_st.cast(str).str.zfill(2),
        min_str=f.minute_st.cast(str).str.zfill(2),
        sec_str=sec_int.cast(str).str.zfill(2)
    )]

    dt_frame[:, 'timeUTC_st_str'] = (
        f.year_str + '-' + f.month_str + '-' + f.day_str + ' ' +
        f.hour_str + ':' + f.min_str + ':' + f.sec_str
    )

    dt_frame[:, 'timeUTC_st'] = dt.math.to_datetime(f.timeUTC_st_str)

    dt_frame[:, 'usec_td'] = dt.math.to_timedelta(usec_val / 1_000_000.0)
    dt_frame[:, update(timeUTC_st = f.timeUTC_st + f.usec_td)]

    del dt_frame[:, [
        'year_str', 'month_str', 'day_str', 'hour_str', 'min_str', 'sec_str',
        'timeUTC_st_str', 'usec_td'
    ]]

    return dt_frame

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
    time: Optional[str] = None
) -> str:
    """
    Generates the expected HYSPLIT footprint NetCDF filename.
    """
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    lats = f"{lat_abs:.4f}{'N' if lat >= 0 else 'S'}"
    lons = f"{lon_abs:.4f}{'E' if lon >= 0 else 'W'}"

    s_yr = f"{year:04d}"
    s_mo = f"{month:02d}"
    s_dy = f"{day:02d}"
    s_hr = f"{hour:02d}"
    s_mn = f"{minute:02d}"

    s_alt = f"{alt:05.0f}"

    basename = f"hysplit-footprint_{s_yr}{s_mo}{s_dy}{s_hr}{s_mn}_{lats}{lons}_{s_alt}magl.nc"

    if fullpath:
        warnings.warn("fullpath=True is not fully implemented without a defined directory structure.")
        return basename
    else:
        return basename