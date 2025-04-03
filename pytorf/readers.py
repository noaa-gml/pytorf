
"""
Functions for reading ObsPack files (.txt, .nc) and metadata.
"""

import datatable as dt
from datatable import f, by, count, ifelse, isna, sort, update, join, rbind
import os
import glob
import re
import math
import netCDF4
from pathlib import Path
import warnings
from typing import List, Union, Optional, Dict, Any, Tuple

# Import specific helpers
from .helpers import fex, sr, obs_out, obs_trunc, obs_write_csvy, obs_read_csvy# is only used in examples

# --- obs_summary ---
def obs_summary(
    obs_path: Union[str, Path],
    categories: List[str] = [
        "aircraft-pfp", "aircraft-insitu", "surface-insitu",
        "tower-insitu", "aircore", "surface-pfp",
        "shipboard-insitu", "flask"
    ],
    # lnchar: int = 11, # Less reliable, use regex instead
    out: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    file_pattern: str = '*.txt' # Default to txt, adjust if needed
) -> Optional[dt.Frame]:
    """
    Creates an index data frame (datatable Frame) of ObsPack files.

    Args:
        obs_path: Path to the directory containing ObsPack data files.
        categories: List of strings to identify file categories based on names.
        out: Optional path to save the index CSV file.
        verbose: If True, print summary information.
        file_pattern: Glob pattern to find files (e.g., '*.txt', '*.nc', '*').

    Returns:
        A datatable Frame containing the index, or None if path is invalid/empty.
    """
    obs_dir = Path(obs_path)
    if not obs_dir.is_dir():
        print(f"Error: Path '{obs_path}' is not a valid directory.")
        return None

    # Use glob to find files matching the pattern
    file_paths = sorted(list(obs_dir.glob(file_pattern)))
    if not file_paths:
        print(f"No files matching pattern '{file_pattern}' found in {obs_path}")
        return None

    file_names = [p.name for p in file_paths]
    file_ids = [str(p.resolve()) for p in file_paths] # Store full paths

    # Create index datatable Frame
    index = dt.Frame(id=file_ids, name=file_names)
    index = index[:, :, sort(f.name)] # Sort by name

    # Add row number
    temp_frame = dt.Frame(range(index.nrows))
    index[:, update(n = temp_frame[:, 'C0'])] # Assign the 'C0' column from the temp frame

    # Assign sectors based on categories in filename
    index[:, update(sector=dt.str64)] # Initialize sector column as string
    for category in categories:
        pattern = re.escape(category)
        index[dt.re.match(f.name, f".*{pattern}.*"), update(sector=category)]

    if verbose:
        print(f"Number of files found: {index.nrows}")
        # Check if 'sector' exists and has non-NA values before proceeding
    if 'sector' in index.names and index[:, dt.count(f.sector)][0, 0] > 0: # Check if *any* sectors assigned
        # Filter out rows where sector is NA *first*
        index_assigned = index[~isna(f.sector), :]

        if index_assigned.nrows > 0: # Check if anything remains after filtering
            # Now perform the aggregation on the filtered frame
            summary = index_assigned[:, count(), by(f.sector)]

            # Calculate total (no need for sum, just use nrows of filtered frame)
            total_assigned = index_assigned.nrows # Total is simply the count after filtering NA
            total_frame = dt.Frame(sector=["Total assigned sectors"], N=[total_assigned])
            print("\nFile counts by assigned sector:")
            # Use force=True in rbind just in case N column type differs slightly
            print(rbind(summary, total_frame, force=True))
        else:
            print("\nNo files were assigned a category (all sectors were NA).")


        # Calculate unassigned count (can do this on original index)
        unassigned_count = index[isna(f.sector), count()][0,0]
        if unassigned_count > 0:
            print(f"\nFiles without an assigned sector: {unassigned_count}")
    else:
         print("\nNo 'sector' column found or no sectors were assigned.")


    # Extract 'agl' value if 'magl' is in the filename using regex
    index[:, update(agl=dt.float64)] # Initialize agl column as float
    # Regex to capture the number before magl.ext (handles optional d-)
    # Requires knowing the extension. Let's try to make it more general.
    # Assume format like '...[d-]<number>magl.<ext>'
    # Use fex helper to get extension for each file
    num_pattern_template = r"(d-?)?(\d+)magl\.{ext}$"

    for i in range(index.nrows):
      fname = index[i, f.name]
      fext = fex(fname)
      if not fext: continue # Skip if no extension

      num_pattern = re.compile(num_pattern_template.format(ext=re.escape(fext)))
      match = num_pattern.search(fname)
      if match:
          try:
              agl_val = float(match.group(2))
              index[i, update(agl=agl_val)]
          except (ValueError, IndexError):
              if verbose:
                  print(f"Warning: Could not parse AGL number from {fname}")

    if verbose:
        agl_present = index[~isna(f.agl), count()][0, 0]
        agl_absent = index.nrows - agl_present
        print(f"\nDetected {agl_present} files possibly with AGL in name.")
        print(f"Detected {agl_absent} files possibly without AGL in name.")

    # Save index if 'out' path is provided
    if out:
        out_path = Path(out)
        try:
            index.to_csv(str(out_path))
            if verbose:
                print(f"\nIndex saved to: {out_path}")
        except Exception as e:
            print(f"\nError saving index to {out_path}: {e}")

    return index


# --- obs_read (.txt) ---
def obs_read(
    index: dt.Frame,
    categories: str = "flask", # Process ONE category
    expr: Optional[Any] = None, # Accepts a datatable f-expression
    verbose: bool = True,
    # Metadata extraction parameters (adjust for 0-based Python slicing)
    # R: substr(x, start, stop) -> Python: x[start-1 : stop]
    # We'll use regex patterns instead for robustness if possible,
    # but keep position parameters as fallback/alternative.
    meta_patterns: Optional[Dict[str, str]] = None, # e.g., {'site_code': r"site_code\s+(.+)"}
    meta_positions: Optional[Dict[str, int]] = None, # e.g., {'site_code': 15}
    fill_value: float = -1e+34,
    as_list: bool = False # Ignored, will return combined Frame
) -> Optional[dt.Frame]:
    """
    Reads ObsPack .txt files listed in the index for a specific category,
    extracts metadata from headers, and combines data.

    Args:
        index: A datatable Frame generated by obs_summary.
        categories: The single category of files to read (e.g., 'flask').
        expr: Optional datatable filter expression (e.g., f.altitude > 100).
        verbose: If True, print progress information.
        meta_patterns: Dictionary mapping metadata keys to regex patterns
                       for extraction from header lines. Group 1 should capture value.
        meta_positions: Dictionary mapping metadata keys to start character
                        positions (1-based) for simple substring extraction.
                        If both patterns and positions are given, patterns take precedence.
        fill_value: Value used for missing numeric data in files.
        as_list: Ignored, returns a single combined datatable Frame.

    Returns:
        A combined datatable Frame for the specified category, or None.
    """
    if not isinstance(index, dt.Frame) or index.nrows == 0:
        warnings.warn("Input index must be a non-empty datatable Frame.")
        return None
    if 'sector' not in index.names or 'id' not in index.names or 'name' not in index.names:
         warnings.warn("Index frame must contain 'id', 'name', and 'sector' columns.")
         return None

    # Define default extraction methods if none provided
    if meta_patterns is None and meta_positions is None:
         # Use default positions from R code (adjusted for 0-based)
         # R: substr(x, start, stop) -> Python: x[start-1 : stop]
         # R: substr(x, start) -> Python: x[start-1:]
         meta_positions = {
            "site_code": 15, "site_latitude": 18, "site_longitude": 19,
            "site_name": 15, "site_country": 18, "dataset_project": 21,
            "lab_1_abbr": 16, "dataset_calibration_scale": 31,
            "site_elevation": 20, # Note R code has leading space " site_elevation"
            "altitude_comment": 22, # Note R code has leading space " altitude:comment"
            "site_utc2lst": 18
         }
         # Corresponding keys used in grep/position extraction
         meta_lookup_keys = {
            "site_code": "site_code", "site_latitude": "site_latitude",
            "site_longitude": "site_longitude", "site_name": "site_name",
            "site_country": "site_country", "dataset_project": "dataset_project",
            "lab_1_abbr": "lab_1_abbr", "dataset_calibration_scale": "dataset_calibration_scale",
            "site_elevation": " site_elevation", # Keep leading space for matching R
            "altitude_comment": " altitude:comment", # Keep leading space
            "site_utc2lst": "site_utc2lst"
         }
         use_patterns = False
    elif meta_patterns is not None:
         meta_lookup_keys = {k: k for k in meta_patterns.keys()} # Keys are used directly
         use_patterns = True
         # Compile regex patterns for efficiency
         compiled_patterns = {k: re.compile(v) for k, v in meta_patterns.items()}
    else: # meta_positions is not None
         meta_lookup_keys = {k: k for k in meta_positions.keys()} # Assume key matches lookup
         use_patterns = False

    # Helper to extract metadata based on chosen method
    def get_metadata(lines: List[str]) -> Dict[str, Any]:
        metadata = {}
        found_keys = set()
        for key, lookup in meta_lookup_keys.items():
            for line in lines:
                value = None
                if use_patterns:
                    match = compiled_patterns[key].search(line)
                    if match:
                        try:
                            value = match.group(1).strip() # Assume value is in group 1
                        except IndexError:
                            warnings.warn(f"Regex pattern for '{key}' needs a capturing group.")
                        break # Found value for this key
                else: # Use positions
                    if lookup in line: # Find line containing the key identifier
                        pos = meta_positions[key]
                        if len(line) >= pos:
                            value = line[pos-1:].strip()
                        break # Found line for this key
            metadata[key] = value
            if value is not None:
                found_keys.add(key)
        if verbose and len(found_keys) < len(meta_lookup_keys):
            missing_meta = set(meta_lookup_keys.keys()) - found_keys
            #print(f"    Metadata keys not found: {', '.join(missing_meta)}")
        return metadata


    if verbose:
        print(f"Reading category: {categories}...")

    # Filter index for the specified category
    df_category = index[f.sector == categories, :]

    if df_category.nrows == 0:
        print(f"No files found for category: {categories}")
        return None

    list_of_frames = []

    for i in range(df_category.nrows):
        row = df_category[i, :] # Get the row as a Frame
        file_id = Path(row[0, f.id]) # Use Path object
        file_name = row[0, f.name]
        agl_from_name = row[0, f.agl] if 'agl' in row.names else None # Might be NA or column missing

        if verbose:
            print(f"  {i+1}/{df_category.nrows}: Reading {file_name}")

        try:
            # Read header size (assuming specific format V1 V2 V3 V4...)
            # This is fragile, consider alternative header detection if format varies
            with file_id.open('r', encoding='utf-8', errors='ignore') as f_in:
                first_line = f_in.readline().split()
                if len(first_line) < 4:
                     raise ValueError(f"Cannot determine header size from first line: {first_line}")
                num_header_lines = int(first_line[3]) # Assumes 4th element is header size

                # Read the header content
                f_in.seek(0) # Rewind to read header lines
                header_content = [f_in.readline() for _ in range(num_header_lines)]

            # Extract metadata
            metadata = get_metadata(header_content)

            # Read the data part
            dt_file = dt.fread(str(file_id), skip_to_line=num_header_lines + 1,
                               na_strings=[str(fill_value), "", "NA", "NaN"]) # Add common NA strings

            # Add extracted metadata and file info as new columns
            # Convert specific metadata fields to numeric safely
            def safe_float(val): return float(val) if val is not None else None
            def safe_int(val): return int(val) if val is not None else None

            # Check if metadata keys exist before trying to access
            site_elev = safe_float(metadata.get('site_elevation'))
            site_lat = safe_float(metadata.get('site_latitude'))
            site_lon = safe_float(metadata.get('site_longitude'))
            site_utc = safe_float(metadata.get('site_utc2lst'))

            dt_file[:, update(
                rtorf_filename=file_name,
                rtorf_sector=categories,
                rtorf_file_index=i+1, # Original loop index
                site_code=metadata.get('site_code'),
                site_name=metadata.get('site_name'),
                site_country=metadata.get('site_country'),
                site_latitude_meta=site_lat,
                site_longitude_meta=site_lon,
                site_elevation_meta=site_elev,
                site_utc2lst_meta=site_utc,
                dataset_project=metadata.get('dataset_project'),
                lab_1_abbr=metadata.get('lab_1_abbr'),
                dataset_calibration_scale=metadata.get('dataset_calibration_scale'),
                altitude_comment=metadata.get('altitude_comment'),
                agl_from_filename=agl_from_name # AGL from filename
            )]

            # Altitude Calculation Logic (from R code)
            if 'altitude' in dt_file.names:
                # Initialize derived columns
                dt_file[:, update(agl_calculated=dt.float64, type_altitude=dt.str64, altitude_final=dt.float64)]

                # Calculate agl = altitude - site_elevation where agl_from_filename is NA
                # Use the numeric site_elevation_meta column added above
                dt_file[ isna(f.agl_from_filename) & ~isna(f.site_elevation_meta) & ~isna(f.altitude),
                        update(agl_calculated = f.altitude - f.site_elevation_meta)]

                # Determine final altitude and type based on availability
                dt_file[:, update(
                    # Use agl_from_filename if present, else calculated, else altitude (asl)
                    altitude_final = ifelse(~isna(f.agl_from_filename), f.agl_from_filename,
                                           ifelse(~isna(f.agl_calculated), f.agl_calculated, f.altitude)),
                    # Assign type based on which altitude was used
                    type_altitude = ifelse(~isna(f.agl_from_filename), "magl", # From filename is MAGL
                                           ifelse(~isna(f.agl_calculated), "magl", "masl")) # Calculated or fallback to asl
                    )]

                # If altitude_final is still NA (e.g., all inputs NA), mark type as unavailable
                dt_file[isna(f.altitude_final), update(type_altitude="not available")]

            else:
                 warnings.warn(f"'altitude' column not found in {file_name}, cannot calculate final altitude/type.")
                 # Add columns with NAs if they don't exist
                 dt_file[:, update(type_altitude="not available", altitude_final=dt.float64)]


            # Apply filter expression if provided
            if expr is not None:
                try:
                    dt_file = dt_file[expr, :] # Filter rows
                    if verbose and dt_file.nrows == 0:
                         print(f"    Filter expression resulted in 0 rows for {file_name}")
                except Exception as filter_err:
                    warnings.warn(f"Could not apply filter expression to {file_name}: {filter_err}")

            if dt_file.nrows > 0:
                 list_of_frames.append(dt_file)
            elif verbose:
                 print(f"    No data rows read or remaining after filter for {file_name}")


        except FileNotFoundError:
             warnings.warn(f"File not found: {file_id}")
        except ValueError as ve: # Catch specific errors like header parsing
             warnings.warn(f"Error parsing file {file_name}: {ve}")
        except dt.DatatableWarning as dw: # Catch datatable specific warnings/errors
             warnings.warn(f"Datatable issue with file {file_name}: {dw}")
        except Exception as e:
            warnings.warn(f"Unexpected error processing file {file_name}: {type(e).__name__} - {e}")
            if verbose: import traceback; traceback.print_exc() # Print traceback if verbose
            continue # Skip to next file

    if not list_of_frames:
        print(f"No valid data frames were generated for category '{categories}'.")
        return None

    # Combine all frames using robust rbind logic
    if verbose: print(f"\nCombining {len(list_of_frames)} data frames...")
    try:
        # Using force=True helps align columns with potentially different types initially
        combined_dt = dt.rbind(*list_of_frames, force=True)
        if verbose: print(f"Combined frame shape: {combined_dt.shape}")
        return combined_dt
    except Exception as bind_err:
        print(f"Error combining data frames: {bind_err}")
        # Fallback implemented previously is complex and might hide issues.
        # Returning None is safer if direct rbind fails.
        return None

# --- obs_read_nc ---
def obs_read_nc(
    index: dt.Frame,
    categories: str = "flask",
    att: bool = False, # Add global attributes as columns
    expr: Optional[Any] = None, # Datatable filter expression
    solar_time: Optional[bool] = None, # Default logic based on category
    as_list: bool = False, # Ignored
    verbose: bool = False,
    show_warnings: bool = False # Renamed from 'warnings'
) -> Optional[dt.Frame]:
    """
    Reads ObsPack .nc files listed in the index for a specific category.

    Args:
        index: Frame from obs_summary.
        categories: Single category to read.
        att: If True, add global attributes as columns to the data.
        expr: Optional datatable filter expression (e.g., f.altitude > 1000).
        solar_time: If True, include solar time components. Defaults based on category.
        as_list: Ignored.
        verbose: Print progress.
        show_warnings: Print warnings from netCDF4 library during attribute reading.

    Returns:
        Combined datatable Frame for the category, or None.
    """
    if not isinstance(index, dt.Frame) or index.nrows == 0:
        warnings.warn("Input index must be a non-empty datatable Frame.")
        return None
    if 'sector' not in index.names or 'id' not in index.names or 'name' not in index.names:
         warnings.warn("Index frame must contain 'id', 'name', and 'sector' columns.")
         return None

    # Default solar_time logic
    if solar_time is None:
        solar_time = False if "aircraft" in categories else True

    if verbose:
        print(f"Reading NetCDF category: {categories}...")

    df_category = index[f.sector == categories, :]

    if df_category.nrows == 0:
        print(f"No files found for category: {categories}")
        return None

    list_of_frames = []

    for i in range(df_category.nrows):
        row = df_category[i, :]
        file_id = Path(row[0, f.id]) # Use Path
        file_name = row[0, f.name]
        # agl_from_name = row[0, f.agl] # Available if needed

        if verbose:
            print(f"  {i+1}/{df_category.nrows}: Reading {file_name}")

        try:
            # Use netCDF4.Dataset within a 'with' block for automatic closing
            with netCDF4.Dataset(str(file_id), 'r') as nc:
                # --- Read Essential Variables ---
                if "time_components" not in nc.variables:
                    warnings.warn(f"'time_components' not found in {file_name}. Skipping file.")
                    continue
                # Read time components [nobs, 6] -> transpose to [6, nobs] -> Frame
                time_comps = nc.variables["time_components"][:] # Reads data as numpy array
                if time_comps.ndim != 2 or time_comps.shape[1] != 6:
                     warnings.warn(f"Unexpected shape for 'time_components' in {file_name}: {time_comps.shape}. Skipping.")
                     continue
                dt_file = dt.Frame(time_comps, names=["year", "month", "day", "hour", "minute", "second"])
                n_obs = dt_file.nrows
                if n_obs == 0:
                    if verbose: print(f"    No observations found (based on time_components) in {file_name}.")
                    continue # Skip empty files

                # --- Read Optional Variables ---
                # Solar time components
                if solar_time:
                    if "solartime_components" in nc.variables:
                        solar_comps = nc.variables["solartime_components"][:]
                        if solar_comps.ndim == 2 and solar_comps.shape[0] == n_obs and solar_comps.shape[1] == 6:
                            dt_solar = dt.Frame(solar_comps, names=[f"{n}_st" for n in dt_file.names])
                            dt_file = dt.cbind(dt_file, dt_solar)
                        else:
                             warnings.warn(f"Solar time dimension mismatch or wrong shape in {file_name}: expected ({n_obs}, 6), got {solar_comps.shape}")
                    elif verbose:
                        warnings.warn(f"'solartime_components' not found in {file_name}, but solar_time=True")


                # Read other 1D variables matching the primary time dimension (n_obs)
                for var_name, variable in nc.variables.items():
                    if var_name not in ["time_components", "solartime_components"]:
                         # Check if variable is 1D and has the correct length
                         if variable.ndim == 1 and variable.shape[0] == n_obs:
                             try:
                                 # Read data and add as new column
                                 var_data = variable[:]
                                 # Convert MaskedArray to numpy array with NaNs if needed
                                 if isinstance(var_data, np.ma.MaskedArray):
                                      var_data = np.ma.filled(var_data, np.nan)
                                 dt_file[:, var_name] = dt.Frame(var_data)[f[0]] # Add column
                             except Exception as var_read_err:
                                 warnings.warn(f"Error reading variable '{var_name}' from {file_name}: {var_read_err}")
                         # Optional: Handle variables with other dimensions if needed

                # --- Read Attributes ---
                # Read specific variable attributes (e.g., scale_comment for 'value')
                scale_comment = None
                if 'value' in nc.variables and hasattr(nc.variables['value'], 'scale_comment'):
                     scale_comment = nc.variables['value'].getncattr('scale_comment')
                dt_file[:, update(scale_comment = scale_comment)] # Add column even if None

                # Read global attributes
                global_attrs_dict = {}
                nc_attrs = nc.ncattrs() # Get list of global attribute names
                for attr_name in nc_attrs:
                    try:
                        global_attrs_dict[attr_name] = nc.getncattr(attr_name)
                    except Exception as glob_att_err:
                        # Only show warning if flag is set
                        if show_warnings:
                            print(f"Warning reading global attribute '{attr_name}' in {file_name}: {glob_att_err}")

                # Add specific global attributes as columns, using .get for safety
                manual_attrs = [
                    "dataset_intake_ht_unit", "site_elevation_unit", "dataset_project",
                    "dataset_selection_tag", "site_name", "site_elevation",
                    "site_latitude", "site_longitude", "site_country", "site_code",
                    "site_utc2lst", "lab_1_abbr", "dataset_calibration_scale"
                ]
                update_dict_manual = {
                    attr_name: global_attrs_dict.get(attr_name, None) for attr_name in manual_attrs
                }
                # Convert numeric attributes if needed (they might be read as numpy types)
                for key in ['site_elevation', 'site_latitude', 'site_longitude', 'site_utc2lst']:
                    if update_dict_manual[key] is not None:
                        try:
                           update_dict_manual[key] = float(update_dict_manual[key])
                        except (ValueError, TypeError):
                            if verbose: warnings.warn(f"Could not convert global attribute '{key}' to float.")
                            update_dict_manual[key] = None # Set to None if conversion fails


                dt_file[:, update(**update_dict_manual)]

                # Add ALL global attributes if att=True
                if att:
                    update_dict_all_globals = {}
                    for attr_name, attr_value in global_attrs_dict.items():
                         # Avoid overwriting existing columns unless intended
                         # Also avoid overwriting manually added ones if they differ
                         if attr_name not in dt_file.names and attr_name not in manual_attrs:
                             # Handle potential type issues (attributes can be arrays, etc.)
                             # For simplicity, store as is (datatable might handle basic types)
                             # Complex types might need conversion to string or be skipped
                             if isinstance(attr_value, (str, int, float, bool)):
                                 update_dict_all_globals[attr_name] = attr_value
                             else:
                                 if verbose: warnings.warn(f"Skipping non-scalar global attribute '{attr_name}' of type {type(attr_value)}")

                    if update_dict_all_globals:
                        dt_file[:, update(**update_dict_all_globals)]

                # --- Altitude Logic (NetCDF specific) ---
                # Use the columns added from global attributes
                project = dt_file[0, f.dataset_project] if dt_file.nrows > 0 and 'dataset_project' in dt_file.names and not isna(dt_file[0, f.dataset_project]) else ""
                intake_unit = dt_file[0, f.dataset_intake_ht_unit] if dt_file.nrows > 0 and 'dataset_intake_ht_unit' in dt_file.names and not isna(dt_file[0, f.dataset_intake_ht_unit]) else ""

                dt_file[:, update(altitude_final=dt.float64, type_altitude=dt.str64)] # Initialize

                if "airc" in project.lower(): # Aircraft or aircore
                    if 'altitude' in dt_file.names:
                        # Assume 'altitude' variable is ASL for aircraft
                        dt_file[:, update(altitude_final = f.altitude, type_altitude = "masl")]
                    else:
                         warnings.warn(f"'altitude' variable needed for aircraft altitude in {file_name}. Setting to NA.")
                         dt_file[:, update(type_altitude = "not available")] # altitude_final remains NA
                else: # Surface, tower, etc. -> use intake height
                    if 'intake_height' in dt_file.names:
                        dt_file[:, update(altitude_final = f.intake_height)]
                        # Determine type based on unit attribute
                        dt_file[:, update(type_altitude = ifelse(f.dataset_intake_ht_unit == "magl", "magl", "masl"))]
                        # Handle cases where unit attribute is missing -> assume ASL
                        dt_file[isna(f.dataset_intake_ht_unit), update(type_altitude = "masl")]
                    else:
                        warnings.warn(f"'intake_height' variable needed for non-aircraft altitude in {file_name}. Setting to NA.")
                        dt_file[:, update(type_altitude = "not available")]

                # Final check if altitude_final is still NA
                dt_file[isna(f.altitude_final), update(type_altitude = "not available")]

                # Apply filter expression if provided
                if expr is not None:
                    try:
                        dt_file = dt_file[expr, :]
                        if verbose and dt_file.nrows == 0:
                             print(f"    Filter expression resulted in 0 rows for {file_name}")
                    except Exception as filter_err:
                        warnings.warn(f"Could not apply filter expression to {file_name}: {filter_err}")

                if dt_file.nrows > 0:
                    list_of_frames.append(dt_file)
                elif verbose:
                    print(f"    No data rows read or remaining after filter for {file_name}")


        except FileNotFoundError:
             warnings.warn(f"NetCDF file not found: {file_id}")
        except OSError as oe: # Catch common netCDF4 file access errors
             warnings.warn(f"Error opening/reading NetCDF file {file_name}: {oe}")
        except Exception as e:
            warnings.warn(f"Unexpected error processing NetCDF file {file_name}: {type(e).__name__} - {e}")
            if verbose: import traceback; traceback.print_exc()
            continue # Skip to next file

    if not list_of_frames:
        print(f"No valid data frames were generated for NetCDF category '{categories}'.")
        return None

    # Combine all frames
    if verbose: print(f"\nCombining {len(list_of_frames)} data frames...")
    try:
        combined_dt = dt.rbind(*list_of_frames, force=True)
        if verbose: print(f"Combined frame shape: {combined_dt.shape}")
        return combined_dt
    except Exception as bind_err:
        print(f"Error combining NetCDF data frames: {bind_err}")
        return None

# --- obs_meta ---
def obs_meta(
    index: dt.Frame,
    verbose: bool = True,
    # Re-use extraction parameters from obs_read
    meta_patterns: Optional[Dict[str, str]] = None,
    meta_positions: Optional[Dict[str, int]] = None,
    fill_value: float = -1e+34, # Needed if converting elevation
    as_list: bool = False # Ignored
) -> Optional[dt.Frame]:
    """
    Reads only the metadata from the header of each ObsPack .txt file in the index.

    Args:
        index: A datatable Frame generated by obs_summary.
        verbose: If True, print progress information.
        meta_patterns: Dictionary mapping metadata keys to regex patterns.
        meta_positions: Dictionary mapping metadata keys to start positions (1-based).
        fill_value: Value representing NA for numeric conversion (e.g., elevation).
        as_list: Ignored.

    Returns:
        A datatable Frame containing metadata for each file, or None.
    """
    # Basic validation
    if not isinstance(index, dt.Frame) or index.nrows == 0:
        warnings.warn("Input index must be a non-empty datatable Frame.")
        return None
    if not all(c in index.names for c in ['id', 'name', 'sector']):
         warnings.warn("Index frame must contain 'id', 'name', and 'sector' columns.")
         return None

    # Define default extraction methods (same logic as obs_read)
    if meta_patterns is None and meta_positions is None:
         meta_positions = {
            "site_code": 15, "site_latitude": 18, "site_longitude": 19,
            "site_name": 15, "site_country": 18, "dataset_project": 21,
            "lab_1_abbr": 16, "dataset_calibration_scale": 31,
            "site_elevation": 20, "altitude_comment": 22, "site_utc2lst": 18
         }
         meta_lookup_keys = {
            "site_code": "site_code", "site_latitude": "site_latitude",
            "site_longitude": "site_longitude", "site_name": "site_name",
            "site_country": "site_country", "dataset_project": "dataset_project",
            "lab_1_abbr": "lab_1_abbr", "dataset_calibration_scale": "dataset_calibration_scale",
            "site_elevation": " site_elevation", "altitude_comment": " altitude:comment",
            "site_utc2lst": "site_utc2lst"
         }
         use_patterns = False
    elif meta_patterns is not None:
         meta_lookup_keys = {k: k for k in meta_patterns.keys()}
         use_patterns = True
         compiled_patterns = {k: re.compile(v) for k, v in meta_patterns.items()}
    else: # meta_positions is not None
         meta_lookup_keys = {k: k for k in meta_positions.keys()}
         use_patterns = False

     # Metadata extraction helper (same as obs_read)
    def get_metadata(lines: List[str]) -> Dict[str, Any]:
        metadata = {}
        for key, lookup in meta_lookup_keys.items():
            value = None
            for line in lines:
                if use_patterns:
                    match = compiled_patterns[key].search(line)
                    if match:
                        try: value = match.group(1).strip(); break
                        except IndexError: pass
                else:
                    if lookup in line:
                        pos = meta_positions[key]
                        if len(line) >= pos: value = line[pos-1:].strip(); break
            metadata[key] = value
        return metadata

    metadata_list = [] # List to store dictionaries of metadata per file

    if verbose: print("Extracting metadata...")
    for i in range(index.nrows):
        row = index[i, :]
        file_id = Path(row[0, f.id])
        file_name = row[0, f.name]
        sector = row[0, f.sector]
        agl_val = row[0, f.agl] if 'agl' in row.names else None

        if verbose:
            print(f"  {i+1}/{index.nrows}: Reading header {file_name}")

        try:
            # Read header size and content (same logic as obs_read)
            with file_id.open('r', encoding='utf-8', errors='ignore') as f_in:
                first_line = f_in.readline().split()
                if len(first_line) < 4: raise ValueError("Cannot determine header size")
                num_header_lines = int(first_line[3])
                f_in.seek(0)
                header_content = [f_in.readline() for _ in range(num_header_lines)]

            # Extract metadata
            file_meta = get_metadata(header_content)

            # Add file info from index
            file_meta['rtorf_filename'] = file_name
            file_meta['rtorf_filepath'] = str(file_id)
            file_meta['rtorf_sector'] = sector
            file_meta['rtorf_agl_filename'] = agl_val # AGL inferred from filename
            file_meta['rtorf_file_index'] = i + 1

            # Convert numeric fields, handling fill_value for elevation
            def safe_float_meta(key, fill_val=None):
                 val_str = file_meta.get(key)
                 if val_str is None: return None
                 try:
                     val_num = float(val_str)
                     # Check against fill_value *after* conversion
                     return None if fill_val is not None and abs(val_num - fill_val) < 1e-9 else val_num
                 except (ValueError, TypeError):
                     return None

            file_meta['site_elevation'] = safe_float_meta('site_elevation', fill_value)
            file_meta['site_latitude'] = safe_float_meta('site_latitude')
            file_meta['site_longitude'] = safe_float_meta('site_longitude')
            file_meta['site_utc2lst'] = safe_float_meta('site_utc2lst')

            metadata_list.append(file_meta)

        except FileNotFoundError:
             warnings.warn(f"File not found: {file_id}. Skipping metadata.")
        except ValueError as ve:
             warnings.warn(f"Error parsing header for {file_name}: {ve}. Skipping metadata.")
        except Exception as e:
            warnings.warn(f"Unexpected error processing header for {file_name}: {type(e).__name__} - {e}. Skipping metadata.")
            if verbose: import traceback; traceback.print_exc()

    if not metadata_list:
        print("No metadata could be extracted.")
        return None

    # Convert list of dictionaries to datatable Frame
    try:
        meta_dt = dt.Frame(metadata_list)
        if verbose: print(f"\nMetadata frame shape: {meta_dt.shape}")
        return meta_dt
    except Exception as frame_err:
        print(f"Error creating final metadata frame: {frame_err}")
        return None

# Placeholder for obs_read_nc_att if needed - similar logic to obs_read_nc but only extracts attributes
def obs_read_nc_att(
    index: dt.Frame,
    as_list: bool = False, # Ignored
    verbose: bool = False,
    show_warnings: bool = False
) -> Optional[dt.Frame]:
     """Reads only the global attributes from NetCDF files in the index."""
     warnings.warn("obs_read_nc_att implementation is basic. Review needed.")
     if not isinstance(index, dt.Frame) or index.nrows == 0: return None
     if not all(c in index.names for c in ['id', 'name']): return None

     attribute_list = []
     if verbose: print("Extracting NetCDF global attributes...")
     for i in range(index.nrows):
        row = index[i, :]
        file_id = Path(row[0, f.id])
        file_name = row[0, f.name]
        if verbose: print(f"  {i+1}/{index.nrows}: Reading attributes {file_name}")

        try:
            with netCDF4.Dataset(str(file_id), 'r') as nc:
                 file_attrs = {attr: nc.getncattr(attr) for attr in nc.ncattrs()}
                 file_attrs['rtorf_filename'] = file_name
                 file_attrs['rtorf_filepath'] = str(file_id)
                 attribute_list.append(file_attrs)
        except Exception as e:
             warnings.warn(f"Error reading attributes from {file_name}: {e}")

     if not attribute_list: return None
     try:
         # This will fail if attributes have non-scalar types or differing keys
         attr_dt = dt.Frame(attribute_list)
         if verbose: print(f"\nAttribute frame shape: {attr_dt.shape}")
         return attr_dt
     except Exception as frame_err:
         print(f"Error creating attribute frame (attributes might be complex or inconsistent): {frame_err}")
         # Consider returning the list of dicts instead or handling complex types
         return None # Or return attribute_list