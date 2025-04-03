
"""
Functions for processing ObsPack data (time additions, aggregation).
"""

import datatable as dt
from datatable import f, by, count, ifelse, isna, sort, update, join, rbind
import math
import datetime as pydt
import warnings
import numpy as np # Needed for decimal year calculation
from typing import List, Union, Optional, Dict, Any, Tuple

# Import helpers if needed
# from .helpers import ...

# --- obs_addtime ---
def obs_addtime(
    dt_frame: dt.Frame,
    verbose: bool = True,
    tz_str: str = "UTC" # Currently only supports UTC for epoch conversion
) -> dt.Frame:
    """
    Adds datetime columns based on epoch time columns (time, start_time).
    Calculates end times based on start/midpoint or interval.

    Assumes input columns 'time', 'start_time', 'midpoint_time' (and optionally
    'time_interval') contain epoch seconds (seconds since 1970-01-01 UTC).

    Args:
        dt_frame: Input datatable Frame.
        verbose: Print progress.
        tz_str: Timezone string (currently only 'UTC' supported for epoch).

    Returns:
        The input Frame with added/modified time columns (modified in-place).
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    if tz_str != "UTC":
         warnings.warn("Timezone handling currently assumes UTC for epoch conversions.")

    # Helper to convert epoch seconds (datatable column) to datetime objects (object column)
    def epoch_col_to_datetime_col(epoch_col: dt.Column):
        results = []
        # Check if column is numeric
        if not epoch_col.stype in (dt.stype.int32, dt.stype.int64, dt.stype.float32, dt.stype.float64):
             warnings.warn(f"Epoch column '{epoch_col.name}' is not numeric, cannot convert to datetime.")
             return dt.Frame([None] * epoch_col.nrows)[f[0]] # Return column of Nones

        epoch_list = epoch_col.to_list()[0]
        for epoch in epoch_list:
            # Check for None (NA) or NaN before conversion
            if epoch is None or (isinstance(epoch, float) and math.isnan(epoch)):
                results.append(None)
            else:
                try:
                   # Create timezone-aware UTC datetime objects
                   results.append(pydt.datetime.fromtimestamp(float(epoch), pydt.timezone.utc))
                except (OSError, ValueError): # Handle out-of-range timestamps
                    results.append(None)
                except TypeError: # Handle non-numeric values that slipped through
                     results.append(None)
        # Return as a new datatable column (object type for datetime)
        return dt.Frame(results)[f[0]] # Extract the column from the temporary frame


    if verbose: print("Adding timeUTC (from 'time')")
    if 'time' in dt_frame.names:
        dt_frame[:, update(timeUTC = epoch_col_to_datetime_col(f.time))]
    else:
        warnings.warn("'time' column not found for timeUTC calculation.")
        dt_frame[:, update(timeUTC = dt.obj64)] # Add empty object column

    if verbose: print("Adding timeUTC_start (from 'start_time')")
    if 'start_time' in dt_frame.names:
        dt_frame[:, update(timeUTC_start = epoch_col_to_datetime_col(f.start_time))]
    else:
        warnings.warn("'start_time' column not found for timeUTC_start calculation.")
        dt_frame[:, update(timeUTC_start = dt.obj64)]

    # --- Calculate timeUTC_end ---
    if verbose: print("Calculating timeUTC_end")
    dt_frame[:, update(timeUTC_end = dt.obj64)] # Initialize as object type

    # Check required columns exist before calculation
    has_interval = 'time_interval' in dt_frame.names
    has_midpoint = 'midpoint_time' in dt_frame.names
    has_start_epoch = 'start_time' in dt_frame.names
    has_start_dt = 'timeUTC_start' in dt_frame.names

    list_time_end = [None] * dt_frame.nrows # Python list to accumulate results

    # Extract necessary columns as Python lists for row-wise processing
    # This is often less efficient than pure datatable operations but easier for complex row logic
    time_intervals = dt_frame[:, f.time_interval].to_list()[0] if has_interval else [None] * dt_frame.nrows
    midpoint_times = dt_frame[:, f.midpoint_time].to_list()[0] if has_midpoint else [None] * dt_frame.nrows
    start_times = dt_frame[:, f.start_time].to_list()[0] if has_start_epoch else [None] * dt_frame.nrows
    timeUTC_starts = dt_frame[:, f.timeUTC_start].to_list()[0] if has_start_dt else [None] * dt_frame.nrows

    # Preferred method: Use time_interval if available
    if has_interval:
        if verbose: print("  Using 'time_interval' where available.")
        for i in range(dt_frame.nrows):
            start_dt = timeUTC_starts[i]
            interval_s = time_intervals[i]

            # Check if start_dt is valid datetime and interval is valid number
            if isinstance(start_dt, pydt.datetime) and isinstance(interval_s, (int, float)) and not math.isnan(interval_s):
                try:
                    list_time_end[i] = start_dt + pydt.timedelta(seconds=float(interval_s))
                except OverflowError: pass # Handle large timedeltas if necessary
                except TypeError: pass
            # Else: leave list_time_end[i] as None for this row if interval method fails

    # Fallback method: Use midpoint calculation where interval wasn't used/available
    can_use_midpoint = has_midpoint and has_start_epoch and has_start_dt
    if can_use_midpoint:
        if verbose: print("  Using midpoint calculation for rows without valid 'time_interval'.")
        for i in range(dt_frame.nrows):
            # Only calculate if end time hasn't been set by interval method
            if list_time_end[i] is None:
                start_dt = timeUTC_starts[i]
                mid_t = midpoint_times[i]
                start_t = start_times[i]

                # Check if all needed values are valid
                if isinstance(start_dt, pydt.datetime) and \
                   isinstance(mid_t, (int, float)) and not math.isnan(mid_t) and \
                   isinstance(start_t, (int, float)) and not math.isnan(start_t):
                    try:
                        # Duration = 2 * (midpoint_epoch - start_epoch)
                        duration_s = (float(mid_t) - float(start_t)) * 2.0
                        list_time_end[i] = start_dt + pydt.timedelta(seconds=duration_s)
                    except OverflowError: pass
                    except TypeError: pass
                # Else: leave list_time_end[i] as None if midpoint method fails

    # If neither method was possible
    if not has_interval and not can_use_midpoint:
        warnings.warn("Cannot calculate 'timeUTC_end'. Missing 'time_interval' or ('midpoint_time' and 'start_time').")

    # Update the datatable Frame with the calculated end times
    dt_frame[:, update(timeUTC_end = dt.Frame(list_time_end)[f[0]])]


    # --- Calculate difference and add end components ---
    if verbose: print("Calculating time differences and end components.")
    dt_frame[:, update(
        dif_time_seconds = dt.float64,
        time_warning = dt.str64,
        year_end = dt.int32, month_end = dt.int8, day_end = dt.int8,
        hour_end = dt.int8, minute_end = dt.int8, second_end = dt.float64
        )]

    # Extract columns needed for final calculations
    timeUTCs = dt_frame[:, f.timeUTC].to_list()[0] if 'timeUTC' in dt_frame.names else [None] * dt_frame.nrows
    timeUTC_starts = dt_frame[:, f.timeUTC_start].to_list()[0] # Already extracted
    timeUTC_ends = dt_frame[:, f.timeUTC_end].to_list()[0] # Use the newly updated column

    # Lists to store results
    list_diffs = [None] * dt_frame.nrows
    list_warnings = [None] * dt_frame.nrows
    list_y_end, list_m_end, list_d_end, list_h_end, list_min_end, list_s_end = ([None] * dt_frame.nrows for _ in range(6))

    for i in range(dt_frame.nrows):
        start_dt = timeUTC_starts[i]
        end_dt = timeUTC_ends[i]
        utc_dt = timeUTCs[i]

        # Calculate difference
        if isinstance(start_dt, pydt.datetime) and isinstance(end_dt, pydt.datetime):
            try:
                list_diffs[i] = (end_dt - start_dt).total_seconds()
            except TypeError: pass # Should not happen if inputs are datetimes

        # Check warning condition (timeUTC == timeUTC_end)
        if isinstance(utc_dt, pydt.datetime) and isinstance(end_dt, pydt.datetime):
             try:
                 # Use a small tolerance for comparison due to potential float precision
                 if abs((utc_dt - end_dt).total_seconds()) < 1e-6 :
                      list_warnings[i] = "warning, timeUTC == timeUTC_end"
                 else:
                      list_warnings[i] = "all good"
             except TypeError:
                  list_warnings[i] = "comparison error"
        else:
             # If either is None or not a datetime
             list_warnings[i] = "info, missing time data for comparison"


        # Extract end components
        if isinstance(end_dt, pydt.datetime):
             try:
                list_y_end[i] = end_dt.year
                list_m_end[i] = end_dt.month
                list_d_end[i] = end_dt.day
                list_h_end[i] = end_dt.hour
                list_min_end[i] = end_dt.minute
                list_s_end[i] = end_dt.second + end_dt.microsecond / 1e6 # Include microseconds
             except AttributeError: pass # Should not happen if it's a datetime


    # Update the frame with the results from the lists
    update_dict = {
        'dif_time_seconds': dt.Frame(list_diffs)[f[0]],
        'time_warning': dt.Frame(list_warnings)[f[0]],
        'year_end': dt.Frame(list_y_end)[f[0]],
        'month_end': dt.Frame(list_m_end)[f[0]],
        'day_end': dt.Frame(list_d_end)[f[0]],
        'hour_end': dt.Frame(list_h_end)[f[0]],
        'minute_end': dt.Frame(list_min_end)[f[0]],
        'second_end': dt.Frame(list_s_end)[f[0]],
    }
    dt_frame[:, update(**update_dict)]

    return dt_frame

# --- obs_agg ---
def obs_agg(
    dt_frame: dt.Frame,
    cols: List[str] = [ # Default aggregation columns from R
        "year", "month", "day", "hour", "minute", "second", "time",
        "time_decimal", "value", "latitude", "longitude", "altitude",
        "pressure", "u", "v", "temperature", "type_altitude"
    ],
    by_cols: List[str] = [ # Default grouping columns from R
        "key_time", "site_code", "altitude_final", "type_altitude",
        "lab_1_abbr", "dataset_calibration_scale"
    ],
    fn: str = "mean", # Function name as string ('mean', 'sum', 'median', etc.)
    na_rm: bool = True, # Corresponds to R's na.rm=T
    verbose: bool = True
) -> Optional[dt.Frame]:
    """
    Aggregates ObsPack data based on specified columns and function.

    Args:
        dt_frame: Input datatable Frame.
        cols: List of column names to aggregate.
        by_cols: List of column names to group by.
        fn: Aggregation function ('mean', 'median', 'sum', 'min', 'max', 'sd',
            'first', 'last', 'count', 'nunique'). Case-insensitive.
        na_rm: If True, ignore NA values during aggregation (default).
               Note: Datatable aggregations generally ignore NAs by default.
                     This flag is kept for conceptual parity with R.
        verbose: Print information.

    Returns:
        Aggregated datatable Frame, or None if errors occur.
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    # Check for required grouping columns
    missing_by = [c for c in by_cols if c not in dt_frame.names]
    if missing_by:
        # Allow run even if grouping cols are missing, but warn.
        warnings.warn(f"Grouping columns missing, aggregation might fail or produce unexpected results: {', '.join(missing_by)}")
        # Alternative: raise ValueError for stricter check
        # raise ValueError(f"Missing required grouping columns: {', '.join(missing_by)}")
        valid_by_cols = [c for c in by_cols if c in dt_frame.names]
        if not valid_by_cols:
             print("Error: No valid grouping columns found.")
             return None
    else:
        valid_by_cols = by_cols


    # Map function string to datatable aggregation function/expression
    # Datatable's aggregators generally handle NA removal implicitly.
    agg_map = {
        "mean": dt.mean, "median": dt.median, "sum": dt.sum,
        "min": dt.min, "max": dt.max, "sd": dt.sd,
        "first": dt.first, "last": dt.last,
        "count": dt.count, # Counts non-NA values
        "nunique": dt.nunique,
        # 'countna': lambda x: dt.countna(x) # Example if needed
    }
    fn_lower = fn.lower()
    if fn_lower not in agg_map:
        raise ValueError(f"Unsupported aggregation function: '{fn}'. Choose from {list(agg_map.keys())}")

    agg_func = agg_map[fn_lower]

    # Filter cols to those actually present in the frame
    valid_cols = [c for c in cols if c in dt_frame.names]
    skipped_cols = list(set(cols) - set(valid_cols))
    if skipped_cols:
        warnings.warn(f"Columns not found in data and skipped for aggregation: {', '.join(skipped_cols)}")
    if not valid_cols:
         print("Error: No valid columns found to aggregate.")
         return None

    # Build the aggregation expression dictionary for the j part
    # Apply the chosen aggregation function to each valid column
    agg_exprs = {col: agg_func(f[col]) for col in valid_cols}

    if verbose:
        print(f"Aggregating columns: {', '.join(valid_cols)}")
        print(f"Grouping by columns: {', '.join(valid_by_cols)}")
        print(f"Using function: {fn_lower} (NA removal is default in datatable)")

    try:
        # Perform aggregation using the dictionary expansion and by()
        agg_dt = dt_frame[:, agg_exprs, by(*valid_by_cols)]
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None

    # Add/Recalculate time components based on 'key_time' if it was a grouping key
    if 'key_time' in valid_by_cols and 'key_time' in agg_dt.names:
         if verbose: print("Adding/updating time components based on 'key_time'.")

         # --- Recalculate timeUTC ---
         # Helper function (can be defined outside or imported)
         def epoch_col_to_datetime_col_agg(epoch_col):
             # ... (same implementation as in obs_addtime) ...
             results = []
             if not epoch_col.stype in (dt.stype.int32, dt.stype.int64, dt.stype.float32, dt.stype.float64): return dt.Frame([None] * epoch_col.nrows)[f[0]]
             epoch_list = epoch_col.to_list()[0]
             for epoch in epoch_list:
                if epoch is None or (isinstance(epoch, float) and math.isnan(epoch)): results.append(None)
                else:
                    try: results.append(pydt.datetime.fromtimestamp(float(epoch), pydt.timezone.utc))
                    except (OSError, ValueError, TypeError): results.append(None)
             return dt.Frame(results)[f[0]]

         agg_dt[:, update(timeUTC = epoch_col_to_datetime_col_agg(f.key_time))]

         # --- Recalculate components and time_decimal ---
         list_y, list_m, list_d, list_h, list_min, list_s = ([None] * agg_dt.nrows for _ in range(6))
         list_time_dec = [None] * agg_dt.nrows
         timeUTCs = agg_dt['timeUTC'].to_list()[0] # Use the newly created/updated column

         # Decimal year calculation helper
         def to_decimal_year(date_obj):
             # ... (same implementation as in obs_agg example - make sure it handles timezones if needed) ...
            if date_obj is None: return None
            try:
                 year = date_obj.year
                 # Ensure start_of_year has the same timezone awareness
                 tz = date_obj.tzinfo
                 start_of_year = pydt.datetime(year, 1, 1, tzinfo=tz)
                 start_of_next_year = pydt.datetime(year + 1, 1, 1, tzinfo=tz)
                 # Use total_seconds() for accurate duration calculation
                 year_duration_seconds = (start_of_next_year - start_of_year).total_seconds()
                 if year_duration_seconds == 0: return float(year) # Avoid division by zero for edge cases
                 time_into_year_seconds = (date_obj - start_of_year).total_seconds()
                 return year + time_into_year_seconds / year_duration_seconds
            except Exception: # Catch potential errors during calculation
                 return None


         for i in range(agg_dt.nrows):
             utc_dt = timeUTCs[i]
             if isinstance(utc_dt, pydt.datetime):
                 try:
                     list_y[i] = utc_dt.year
                     list_m[i] = utc_dt.month
                     list_d[i] = utc_dt.day
                     list_h[i] = utc_dt.hour
                     list_min[i] = utc_dt.minute
                     list_s[i] = utc_dt.second + utc_dt.microsecond / 1e6
                     list_time_dec[i] = to_decimal_year(utc_dt)
                 except AttributeError: pass # Should not happen

         # Update frame - potentially overwrite columns if they existed from aggregation
         update_time_dict = {
             'year': dt.Frame(list_y)[f[0]], 'month': dt.Frame(list_m)[f[0]], 'day': dt.Frame(list_d)[f[0]],
             'hour': dt.Frame(list_h)[f[0]], 'minute': dt.Frame(list_min)[f[0]], 'second': dt.Frame(list_s)[f[0]],
             'time_decimal': dt.Frame(list_time_dec)[f[0]],
             'time': f.key_time # Update 'time' to be consistent with 'key_time' epoch
         }
         agg_dt[:, update(**update_time_dict)]

    elif verbose:
        print("Info: 'key_time' not in grouping columns, time components not recalculated.")


    # Sort results if possible
    sort_cols = ["site_code", "timeUTC"] # R code sorts by site_code, timeUTC
    valid_sort_cols = [c for c in sort_cols if c in agg_dt.names]
    if valid_sort_cols:
        if verbose: print(f"Sorting aggregated results by: {', '.join(valid_sort_cols)}")
        # Use datatable's sort()
        agg_dt = agg_dt[:, :, sort(*valid_sort_cols)]
    else:
         warnings.warn("Cannot sort aggregated results: Missing 'site_code' or 'timeUTC'.")

    return agg_dt

# --- obs_addltime ---
def obs_addltime(
    dt_frame: dt.Frame,
    time_utc_col: str = "timeUTC",
    utc2lt_col: str = "site_utc2lst",
    longitude_col: str = "longitude",
    # tz_str: str = "UTC" # Less relevant as calculations use timedelta
) -> dt.Frame:
    """
    Calculates approximate local time based on UTC time and longitude or a UTC offset.

    Args:
        dt_frame: Input datatable Frame.
        time_utc_col: Name of the column containing UTC datetime objects.
        utc2lt_col: Name of the column containing the UTC to Local Time offset
                    (in hours, e.g., -5 for EST). Priority is given to this.
        longitude_col: Name of the column containing longitude (degrees East).
                       Used if utc2lt_col is missing or NA.

    Returns:
        The input Frame with added 'local_time' (datetime object) and 'lh' (local hour)
        columns (modified in-place).
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    # Check required columns
    if time_utc_col not in dt_frame.names:
        raise ValueError(f"Required column '{time_utc_col}' not found.")

    has_utc_offset = utc2lt_col in dt_frame.names
    has_longitude = longitude_col in dt_frame.names

    if not has_utc_offset and not has_longitude:
        warnings.warn(f"Neither '{utc2lt_col}' nor '{longitude_col}' found. Cannot calculate local time.")
        dt_frame[:, update(local_time = dt.obj64, lh = dt.int8)] # Add empty columns
        return dt_frame

    # Initialize new columns
    dt_frame[:, update(local_time = dt.obj64, lh = dt.int8)] # obj64 for datetime, int8 for hour

    # Extract columns as lists for row-wise processing
    time_utc_list = dt_frame[:, f[time_utc_col]].to_list()[0]
    utc_offset_list = dt_frame[:, f[utc2lt_col]].to_list()[0] if has_utc_offset else [None] * dt_frame.nrows
    longitude_list = dt_frame[:, f[longitude_col]].to_list()[0] if has_longitude else [None] * dt_frame.nrows

    local_time_list = [None] * dt_frame.nrows
    local_hour_list = [None] * dt_frame.nrows

    print(f"Calculating local time. Priority to '{utc2lt_col}' if available.")
    for i in range(dt_frame.nrows):
        utc_dt = time_utc_list[i]
        if not isinstance(utc_dt, pydt.datetime):
            continue # Skip if base UTC time is invalid

        offset_hours = None
        # Try using UTC offset column first
        if has_utc_offset and utc_offset_list[i] is not None and isinstance(utc_offset_list[i], (int, float)) and not math.isnan(utc_offset_list[i]):
            offset_hours = float(utc_offset_list[i])
        # Fallback to longitude if UTC offset wasn't valid
        elif has_longitude and longitude_list[i] is not None and isinstance(longitude_list[i], (int, float)) and not math.isnan(longitude_list[i]):
            # John Miller approach: offset = longitude / 15
            offset_hours = float(longitude_list[i]) / 15.0
        # else: offset_hours remains None

        # Calculate local time if an offset was determined
        if offset_hours is not None:
            try:
                local_dt = utc_dt + pydt.timedelta(hours=offset_hours)
                local_time_list[i] = local_dt
                local_hour_list[i] = local_dt.hour
            except (OverflowError, TypeError):
                 # Handle potential errors with timedelta calculation
                 pass # Leave as None

    # Update the datatable Frame
    dt_frame[:, update(
        local_time = dt.Frame(local_time_list)[f[0]],
        lh = dt.Frame(local_hour_list)[f[0]]
    )]

    return dt_frame

# --- obs_addstime ---
def obs_addstime(
    dt_frame: dt.Frame,
    # tz: str = "UTC" # Assume components define UTC time
) -> dt.Frame:
    """
    Adds a 'timeUTC_st' column by combining solar time component columns.

    Assumes columns 'year_st', 'month_st', 'day_st', 'hour_st',
    'minute_st', 'second_st' exist.

    Args:
        dt_frame: Input datatable Frame.

    Returns:
        The input Frame with the added 'timeUTC_st' column (modified in-place).
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input must be a datatable Frame.")

    # Check for required solar time component columns
    st_cols = ['year_st', 'month_st', 'day_st', 'hour_st', 'minute_st', 'second_st']
    missing_cols = [c for c in st_cols if c not in dt_frame.names]
    if missing_cols:
        warnings.warn(f"Missing solar time component columns: {', '.join(missing_cols)}. Cannot create 'timeUTC_st'.")
        dt_frame[:, update(timeUTC_st = dt.obj64)] # Add empty column
        return dt_frame

    # Initialize new column
    dt_frame[:, update(timeUTC_st = dt.obj64)]

    # Extract component columns (handle potential non-numeric or NA)
    try:
        years = dt_frame[:, f.year_st].to_numpy().astype(float)
        months = dt_frame[:, f.month_st].to_numpy().astype(float)
        days = dt_frame[:, f.day_st].to_numpy().astype(float)
        hours = dt_frame[:, f.hour_st].to_numpy().astype(float)
        minutes = dt_frame[:, f.minute_st].to_numpy().astype(float)
        seconds = dt_frame[:, f.second_st].to_numpy().astype(float) # Keep float for potential fractions
    except Exception as e:
         warnings.warn(f"Error converting solar time components to numeric arrays: {e}")
         return dt_frame # Return without adding column if conversion fails

    timeUTC_st_list = [None] * dt_frame.nrows

    for i in range(dt_frame.nrows):
        # Check if any component is NA (NaN for float arrays)
        if np.isnan(years[i]) or np.isnan(months[i]) or np.isnan(days[i]) or \
           np.isnan(hours[i]) or np.isnan(minutes[i]) or np.isnan(seconds[i]):
            continue # Leave as None if any part is missing

        try:
            # Separate integer seconds and microseconds
            sec_int = int(seconds[i])
            usec = int((seconds[i] - sec_int) * 1_000_000)

            # Create datetime object (assume components define UTC)
            st_dt = pydt.datetime(
                int(years[i]), int(months[i]), int(days[i]),
                int(hours[i]), int(minutes[i]), sec_int, usec,
                tzinfo=pydt.timezone.utc # Make timezone-aware UTC
            )
            timeUTC_st_list[i] = st_dt
        except (ValueError, TypeError):
            # Handle invalid date/time combinations (e.g., month=13)
            pass # Leave as None

    # Update the datatable Frame
    dt_frame[:, update(timeUTC_st = dt.Frame(timeUTC_st_list)[f[0]])]

    return dt_frame