"""
Functions for reading ObsPack files (.txt, .nc) and metadata.
"""
import datatable as dt
from datatable import f, by, count, ifelse, isna, sort, update, join, rbind
import os
import re
import netCDF4
from pathlib import Path
import warnings
import yaml
from typing import List, Union, Optional, Dict, Any, Tuple

# --- obs_summary ---
def obs_summary(
    obs_path: Union[str, Path],
    categories: List[str] = [
        "aircraft-pfp", "aircraft-insitu", "surface-insitu",
        "tower-insitu", "aircore", "surface-pfp",
        "shipboard-insitu", "flask"
    ],
    out: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    file_pattern: str = '*.txt'
) -> Optional[dt.Frame]:
    """
    Creates an index data frame (datatable Frame) of ObsPack files.
    """
    obs_dir = Path(obs_path)
    if not obs_dir.is_dir():
        print(f"Error: Path '{obs_path}' is not a valid directory.")
        return None

    file_paths = sorted(list(obs_dir.glob(file_pattern)))
    if not file_paths:
        print(f"No files matching pattern '{file_pattern}' found in {obs_path}")
        return None

    index = dt.Frame(
        id=[str(p.resolve()) for p in file_paths],
        name=[p.name for p in file_paths]
    )
    index[:, 'n'] = range(1, index.nrows + 1)

    # --- Vectorized assignment of sectors from filenames (CORRECTED) ---
    if verbose:
        print("Vectorizing assignment of sectors from filenames...")

    # Use a standard Python list comprehension for maximum compatibility.
    file_names_list = index['name'].to_list()[0]
    sectors = []
    for name in file_names_list:
        found_sector = None
        for category in categories:
            if category in name:
                found_sector = category
                break # Stop at the first match
        sectors.append(found_sector)

    # Assign the list of found sectors to the new 'sector' column.
    index[:, 'sector'] = dt.Frame(sectors)
    # --- End of Correction ---

    if verbose:
        print(f"Number of files found: {index.nrows}")
        if 'sector' in index.names and index[~isna(f.sector), :].nrows > 0:
            summary = index[:, count(), by(f.sector)]
            
            # Sort in ascending order first
            summary.sort('count')
            # Then reverse the frame to get descending order
            summary = summary[::-1, :]

            total_assigned = index[~isna(f.sector), dt.count()][0,0]
            total_frame = dt.Frame(sector=["Total assigned sectors"], N=[total_assigned])
            print("\nFile counts by assigned sector:")
            summary.names = {'count': 'N'}
            print(rbind(summary, total_frame, force=True))
        else:
            print("\nNo files were assigned a category based on the provided patterns.")

        unassigned_count = index[isna(f.sector), dt.count()][0,0]
        if unassigned_count > 0:
            print(f"\nFiles without an assigned sector: {unassigned_count}")

    if out:
        try:
            out_path = Path(out)
            index.to_csv(str(out_path))
            if verbose:
                print(f"\nIndex saved to: {out_path.resolve()}")
        except Exception as e:
            print(f"\nError saving index to {out}: {e}")

    return index

# --- obs_read (.txt) ---
def obs_read(
    index: dt.Frame,
    category: str = "flask",
    expr: Optional[Any] = None,
    verbose: bool = True,
    meta_positions: Optional[Dict[str, int]] = None,
    fill_value: float = -1e+34
) -> Optional[dt.Frame]:
    """
    Reads ObsPack .txt files for a category, extracts metadata, and combines data.
    """
    if not isinstance(index, dt.Frame) or not all(c in index.names for c in ['id', 'name', 'sector']):
         warnings.warn("Index must be a datatable Frame from obs_summary.")
         return None

    if meta_positions is None:
         meta_positions = {
            "site_code": ("site_code", 15), "site_latitude": ("site_latitude", 18),
            "site_longitude": ("site_longitude", 19), "site_name": ("site_name", 15),
            "site_country": ("site_country", 18), "dataset_project": ("dataset_project", 21),
            "lab_1_abbr": ("lab_1_abbr", 16), "dataset_calibration_scale": ("dataset_calibration_scale", 31),
            "site_elevation": (" site_elevation", 20), "altitude_comment": (" altitude:comment", 22),
            "site_utc2lst": ("site_utc2lst", 18)
         }

    def get_metadata(lines: List[str]) -> Dict[str, Any]:
        metadata = {}
        for key, (lookup, pos) in meta_positions.items():
            val = None
            for line in lines:
                if lookup in line and len(line) >= pos:
                    val = line[pos-1:].strip()
                    break
            metadata[key] = val
        return metadata

    if verbose:
        print(f"Reading category: {category}...")

    df_category = index[f.sector == category, :]
    if df_category.nrows == 0:
        print(f"No files found for category: {category}")
        return None

    list_of_frames = []
    for i in range(df_category.nrows):
        # FIX: Extract scalar value from Frame before passing to Path()
        file_id_str = df_category[i, 'id'].to_list()[0]
        file_name = df_category[i, 'name'].to_list()[0]
        file_id = Path(file_id_str)

        if verbose:
            print(f"  {i+1}/{df_category.nrows}: Reading {file_name}")

        try:
            with file_id.open('r', encoding='utf-8', errors='ignore') as f_in:
                first_line = f_in.readline().split()
                num_header_lines = int(first_line[3])
                f_in.seek(0)
                header_content = [f_in.readline() for _ in range(num_header_lines)]

            metadata = get_metadata(header_content)
            dt_file = dt.fread(
                str(file_id),
                skip_to_line=num_header_lines + 1,
                na_strings=[str(fill_value), "NA", "NaN"]
            )

            def safe_float(val): return float(val) if val is not None else None
            
            # Get the agl value if the column exists
            row = df_category[i,:]
            agl_val = row[0, f.agl] if 'agl' in row.names else None

            dt_file[:, update(
                rtorf_filename=file_name,
                rtorf_sector=category,
                site_code=metadata.get('site_code'),
                site_name=metadata.get('site_name'),
                site_country=metadata.get('site_country'),
                site_latitude_meta=safe_float(metadata.get('site_latitude')),
                site_longitude_meta=safe_float(metadata.get('site_longitude')),
                site_elevation_meta=safe_float(metadata.get('site_elevation')),
                site_utc2lst_meta=safe_float(metadata.get('site_utc2lst')),
                dataset_project=metadata.get('dataset_project'),
                lab_1_abbr=metadata.get('lab_1_abbr'),
                dataset_calibration_scale=metadata.get('dataset_calibration_scale'),
                altitude_comment=metadata.get('altitude_comment'),
                agl_from_filename=agl_val
            )]

            if 'altitude' in dt_file.names:
                dt_file[:, 'agl_calculated'] = f.altitude - f.site_elevation_meta
                dt_file[:, 'altitude_final'] = ifelse(
                    ~isna(f.agl_from_filename), f.agl_from_filename,
                    ifelse(~isna(f.agl_calculated), f.agl_calculated, f.altitude)
                )
                dt_file[:, 'type_altitude'] = ifelse(
                    ~isna(f.agl_from_filename), "magl",
                    ifelse(~isna(f.agl_calculated), "magl", "masl")
                )
                dt_file[isna(f.altitude_final), update(type_altitude="not available")]
            else:
                 dt_file[:, update(type_altitude="not available", altitude_final=dt.float64)]

            if expr is not None:
                dt_file = dt_file[expr, :]

            if dt_file.nrows > 0:
                 list_of_frames.append(dt_file)

        except Exception as e:
            warnings.warn(f"Skipping file {file_name} due to error: {e}")
            continue

    if not list_of_frames:
        print(f"No valid data frames were generated for category '{category}'.")
        return None

    try:
        combined_dt = dt.rbind(*list_of_frames, force=True)
        if verbose: print(f"\nCombined frame shape: {combined_dt.shape}")
        return combined_dt
    except Exception as e:
        print(f"Error combining data frames: {e}")
        return None

# --- obs_read_nc ---
def obs_read_nc(
    index: dt.Frame,
    category: str = "flask",
    att: bool = False,
    expr: Optional[Any] = None,
    solar_time: Optional[bool] = None,
    verbose: bool = False
) -> Optional[dt.Frame]:
    """
    Reads ObsPack .nc files for a specific category.
    """
    if not isinstance(index, dt.Frame) or not all(c in index.names for c in ['id', 'name', 'sector']):
         warnings.warn("Index must be a datatable Frame from obs_summary.")
         return None

    if solar_time is None:
        solar_time = "aircraft" not in category

    if verbose:
        print(f"Reading NetCDF category: {category}...")

    df_category = index[f.sector == category, :]
    if df_category.nrows == 0:
        print(f"No files found for category: {category}")
        return None

    list_of_frames = []
    for i in range(df_category.nrows):
        # FIX: Extract scalar value from Frame before passing to Path()
        file_id_str = df_category[i, 'id'].to_list()[0]
        file_name = df_category[i, 'name'].to_list()[0]
        file_id = Path(file_id_str)

        if verbose:
            print(f"  {i+1}/{df_category.nrows}: Reading {file_name}")

        try:
            with netCDF4.Dataset(str(file_id), 'r') as nc:
                if "time_components" not in nc.variables or nc.variables["time_components"].shape[1] != 6:
                    warnings.warn(f"Skipping {file_name}: invalid 'time_components'.")
                    continue
                
                time_comps = nc.variables["time_components"][:]
                dt_file = dt.Frame(time_comps, names=["year", "month", "day", "hour", "minute", "second"])
                n_obs = dt_file.nrows
                if n_obs == 0: continue

                for var_name, var in nc.variables.items():
                    if var.ndim == 1 and var.shape[0] == n_obs and var_name != "time_components":
                         dt_file[:, var_name] = dt.Frame(var[:])
                
                global_attrs = {attr: nc.getncattr(attr) for attr in nc.ncattrs()}
                
                # Add attributes as columns
                if att:
                    for key, value in global_attrs.items():
                        # Ensure value is scalar before adding as a column
                        if isinstance(value, (str, int, float, bool)):
                            dt_file[:, key] = value
                        else:
                            if verbose:
                                warnings.warn(f"Skipping non-scalar global attribute '{key}' of type {type(value)} in file {file_name}")

                if expr is not None:
                    dt_file = dt_file[expr, :]
                
                if dt_file.nrows > 0:
                    list_of_frames.append(dt_file)

        except Exception as e:
            warnings.warn(f"Skipping file {file_name} due to error: {e}")
            continue

    if not list_of_frames:
        print(f"No valid data frames were generated for NetCDF category '{category}'.")
        return None

    try:
        combined_dt = dt.rbind(*list_of_frames, force=True)
        if verbose: print(f"\nCombined frame shape: {combined_dt.shape}")
        return combined_dt
    except Exception as e:
        print(f"Error combining NetCDF data frames: {e}")
        return None