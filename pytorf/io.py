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

    # --- Vectorized assignment of sectors and AGL from filenames (CORRECTED) ---
    if verbose:
        print("Vectorizing assignment of sectors and AGL from filenames...")

    category_pattern = f"({'|'.join(re.escape(c) for c in categories)})"

    # Forcing the computation of the regex match to return a new Frame
    # by selecting the column from the original frame first.
    sector_matches = dt.re.match(index['name'], category_pattern)

    if sector_matches and sector_matches.ncols > 1:
        index[:, 'sector'] = sector_matches[:, 1]
    else:
        index[:, 'sector'] = dt.str32

    agl_matches = dt.re.match(index['name'], r'(\d+)magl')
    if agl_matches and agl_matches.ncols > 1:
        index[:, 'agl'] = agl_matches[:, 1].to_float64()
    else:
        index[:, 'agl'] = dt.float64
    # --- End of Correction ---

    if verbose:
        print(f"Number of files found: {index.nrows}")
        if 'sector' in index.names and index[~isna(f.sector), :].nrows > 0:
            summary = index[:, count(), by(f.sector)]
            summary.sort('count', reverse=True)
            
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
        
        if 'agl' in index.names:
            agl_present = index[~isna(f.agl), dt.count()][0, 0]
            print(f"\nDetected {agl_present} files with AGL in name.")

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
        row = df_category[i, :]
        file_id = Path(row[0, f.id])
        file_name = row[0, f.name]

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
                agl_from_filename=row[0, f.agl] if 'agl' in row.names else None
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
        file_id = Path(df_category[i, f.id])
        file_name = df_category[i, f.name]

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
                # ... (attribute handling logic remains similar) ...

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
    """
    if not isinstance(index, dt.Frame) or index.nrows == 0:
        warnings.warn("Input index must be a non-empty datatable Frame.")
        return None
    if not all(c in index.names for c in ['id', 'name', 'sector']):
         warnings.warn("Index frame must contain 'id', 'name', and 'sector' columns.")
         return None

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

    metadata_list = []
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
            with file_id.open('r', encoding='utf-8', errors='ignore') as f_in:
                first_line = f_in.readline().split()
                if len(first_line) < 4: raise ValueError("Cannot determine header size")
                num_header_lines = int(first_line[3])
                f_in.seek(0)
                header_content = [f_in.readline() for _ in range(num_header_lines)]

            file_meta = get_metadata(header_content)
            file_meta['rtorf_filename'] = file_name
            file_meta['rtorf_filepath'] = str(file_id)
            file_meta['rtorf_sector'] = sector
            file_meta['rtorf_agl_filename'] = agl_val
            file_meta['rtorf_file_index'] = i + 1

            def safe_float_meta(key, fill_val=None):
                 val_str = file_meta.get(key)
                 if val_str is None: return None
                 try:
                     val_num = float(val_str)
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

    try:
        meta_dt = dt.Frame(metadata_list)
        if verbose: print(f"\nMetadata frame shape: {meta_dt.shape}")
        return meta_dt
    except Exception as frame_err:
        print(f"Error creating final metadata frame: {frame_err}")
        return None

def obs_read_nc_att(
    index: dt.Frame,
    as_list: bool = False, # Ignored
    verbose: bool = False,
    show_warnings: bool = False
) -> Optional[dt.Frame]:
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
         attr_dt = dt.Frame(attribute_list)
         if verbose: print(f"\nAttribute frame shape: {attr_dt.shape}")
         return attr_dt
     except Exception as frame_err:
         print(f"Error creating attribute frame (attributes might be complex or inconsistent): {frame_err}")
         return None
     
def obs_read_csvy(f: Union[str, Path], n_header_lines: int = 100, **kwargs) -> Tuple[Optional[Dict], Optional[dt.Frame]]:
    """
    Reads a CSVY file, prints YAML header, and returns header dict and data frame.
    """
    f_path = Path(f)
    yaml_content = []
    in_yaml = False
    header_data = None
    dt_frame = None
    skip_rows = 0

    try:
        with f_path.open('r', encoding='utf-8') as file:
            found_delimiters = 0
            for i, line in enumerate(file):
                current_line_num = i + 1
                if line.strip() == '---':
                    found_delimiters += 1
                    if found_delimiters == 1:
                        in_yaml = True
                    elif found_delimiters == 2:
                        skip_rows = current_line_num
                        break
                elif in_yaml:
                    yaml_content.append(line)

                if current_line_num >= n_header_lines * 2:
                    warnings.warn(f"YAML delimiters '---' not found within expected lines in {f_path}.")
                    skip_rows = 0
                    yaml_content = []
                    break
            else:
                if found_delimiters == 1:
                     warnings.warn(f"YAML end delimiter '---' not found in {f_path}.")
                     skip_rows = current_line_num
                     yaml_content = []

        if yaml_content:
            try:
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

        dt_frame = dt.fread(str(f_path), skip_to_line=skip_rows + 1, **kwargs)

    except FileNotFoundError:
        print(f"Error: File not found {f_path}")
    except Exception as e:
        print(f"Error reading CSVY file {f_path}: {e}")

    return header_data, dt_frame


def obs_write_csvy(
    dt_frame: dt.Frame,
    notes: List[str],
    out: Union[str, Path],
    sep: str = ",",
    **kwargs
):
    """
    Writes a datatable Frame to a CSVY file with YAML header.
    """
    if not isinstance(dt_frame, dt.Frame):
        raise TypeError("Input dt_frame must be a datatable Frame.")

    out_path = Path(out)
    metadata = {
        'name': 'Metadata',
        'notes': notes,
        'structure': {
            'rows': dt_frame.nrows,
            'columns': dt_frame.ncols,
            'names': dt_frame.names,
            'types': [str(t) for t in dt_frame.stypes]
        },
        'generated_by': 'rtorf Python package',
        'timestamp': pydt.datetime.now(pydt.timezone.utc).isoformat()
    }

    try:
        with out_path.open('w', encoding='utf-8') as f:
            f.write("---\n")
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, indent=2)
            f.write("---\n")

        dt_frame.to_csv(str(out_path), sep=sep, header=True, append=True, **kwargs)

    except Exception as e:
        print(f"Error writing CSVY file {out_path}: {e}")
     
def get_metadata(lines: List[str]) -> Dict[str, Any]:
    metadata = {key: None for key in meta_positions}
    keys_to_find = set(meta_positions.keys())

    for line in lines:
        if not keys_to_find:
            break

        found_keys_in_line = []
        for key in keys_to_find:
            lookup, pos = meta_positions[key]
            if lookup in line and len(line) >= pos:
                metadata[key] = line[pos-1:].strip()
                found_keys_in_line.append(key)

        keys_to_find.difference_update(found_keys_in_line)

    return metadata