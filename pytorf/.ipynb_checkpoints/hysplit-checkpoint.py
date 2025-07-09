
"""
Functions for generating HYSPLIT model configuration files (CONTROL, SETUP.CFG, ASCDATA.CFG).
"""
import os
from pathlib import Path
import datetime as pydt
import math
import warnings
from typing import List, Union, Optional, Dict, Any, Tuple, Sequence

# Optional datatable import if reading input from Frame
try:
    import datatable as dt
    from datatable import f, Frame
    DT_AVAILABLE = True
except ImportError:
    DT_AVAILABLE = False
    class Frame: pass # Dummy

# --- obs_hysplit_control ---
def obs_hysplit_control(
    # Input receptor info (either DataFrame row or individual args)
    df_row: Optional[Frame] = None, # A single row datatable Frame
    year: Optional[int] = None, month: Optional[int] = None, day: Optional[int] = None,
    hour: Optional[int] = None, minute: Optional[int] = None,
    lat: Optional[float] = None, lon: Optional[float] = None, alt: Optional[float] = None,
    # HYSPLIT Parameters
    nlocations: int = 1,
    duration: int = -240, # Hours, negative for backward
    vertical_motion: int = 0,
    top_model_domain: float = 20000.0, # Meters AGL
    met: Union[str, List[str]] = ["gfs0p25"], # Met file type identifiers
    # nmet: Optional[int] = None, # Number of met files per type (calculated if None)
    met_days_buffer: int = 1, # Extra days of met data before/after run
    metpath: Union[str, List[str]] = ["/default/met/path/"], # Paths corresponding to met types
    ngases: int = 1,
    gas: str = "Foot", # Pollutant name
    emissions_rate: float = 0.0, # kg/hr
    hour_emissions: float = 0.01, # Duration of emission (hr) - 0.01 for ~instantaneous puff
    # release_start: Optional[str] = None, # Start time string YYMMDDHHmm (calculated)
    nsim_grids: int = 1,
    center_conc_grids: Tuple[float, float] = (0.0, 0.0), # Lat, Lon
    grid_spacing: Tuple[float, float] = (0.1, 0.1), # Deg Lat, Deg Lon
    grid_span: Tuple[float, float] = (30.0, 30.0), # Deg Lat, Deg Lon
    output_dir: str = "./", # Directory for concentration output file
    nconc: str = "cdump", # Concentration output filename
    nvert_levels: int = 1,
    height_vert_levels: Sequence[float] = (50.0,), # Heights of concentration levels (m)
    sampling_start_time: Optional[Sequence[int]] = None, # YYMMDDHHmm or [0,0,0,0,0] for relative start
    sampling_end_time: Optional[Sequence[int]] = None, # YYMMDDHHmm or [0,0,0,0,0] for relative end
    sampling_interval_type: Sequence[int] = (0, 0, 0), # Type (0=avg), Hour, Minute interval
    npol_depositing: int = 1, # Number of pollutants with deposition
    particle_params: Sequence[float] = (0.0, 0.0, 0.0), # Diameter(um), Density(g/cc), Shape(0-1)
    dmwsrdre: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0), # DepVel(m/s), MW(g/mol), SurfRea(0-1), Diffus(cm2/s), Henry(M/a)
    wrhcicbc: Sequence[float] = (0.0, 0.0, 0.0), # Wet removal: Henry's Const, InCloud(1/s), BelowCloud(1/s)
    radiactive_decay: float = 0.0, # Half-life (days), 0 for none
    pol_res: float = 0.0, # Pollutant Resuspension (1/m)
    control_filename: str = "CONTROL" # Output filename
) -> None:
    """
    Creates a HYSPLIT CONTROL file based on input parameters.

    Args:
        df_row: A single-row datatable Frame containing receptor info
                (requires columns like 'year', 'month', ..., 'altitude').
                If provided, overrides individual time/location arguments.
        year, month,... alt: Individual receptor info (used if df_row is None).
        nlocations: Number of starting locations (usually 1 per file).
        duration: Run duration in hours (negative for backward).
        vertical_motion: HYSPLIT vertical motion option (0=data, 3=isentropic, ...).
        top_model_domain: Top of the model domain in meters AGL.
        met: List of meteorological data type identifiers (e.g., 'gfs0p25', 'nam').
        met_days_buffer: Number of extra met days to include beyond run duration.
        metpath: List of base paths corresponding to the 'met' identifiers.
        ngases: Number of pollutant species.
        gas: Name(s) of the pollutant species.
        emissions_rate: Emission rate (kg/hr) for each species.
        hour_emissions: Duration of emission (hours) for each species. 0.01 ~ puff.
        nsim_grids: Number of concentration grids.
        center_conc_grids: Center lat/lon for the concentration grid.
        grid_spacing: Grid resolution (deg lat, deg lon).
        grid_span: Grid extent (deg lat, deg lon).
        output_dir: Output directory for concentration files.
        nconc: Base name for concentration output files (e.g., 'cdump').
        nvert_levels: Number of vertical concentration levels.
        height_vert_levels: Heights (m AGL) of the top of each vertical level.
        sampling_start_time: YYMMDDHHmm or [0,0,0,0,0]. Default uses run start.
        sampling_end_time: YYMMDDHHmm or [0,0,0,0,0]. Default uses run end.
        sampling_interval_type: [Type(0=avg,1=snap), HH, MM] interval. Default (0,0,0) -> single output sum/avg.
        npol_depositing: Number of pollutants undergoing deposition.
        particle_params: Particle characteristics [diameter, density, shape].
        dmwsrdre: Deposition parameters [DepVel, MW, SurfRea, Diffus, Henry].
        wrhcicbc: Wet removal parameters [Henry, InCloud, BelowCloud].
        radiactive_decay: Radioactive half-life in days (0=none).
        pol_res: Resuspension factor (1/m).
        control_filename: Name of the output CONTROL file.

    Returns:
        None. Writes the file directly.
    """

    # --- Extract Receptor Info ---
    if df_row is not None:
        if not DT_AVAILABLE: raise ImportError("datatable needed to process df_row")
        if not isinstance(df_row, Frame) or df_row.nrows != 1:
            raise ValueError("df_row must be a single-row datatable Frame.")
        # Assume column names match R example usage
        row_dict = df_row.to_dict()
        yr = int(row_dict['year'][0])
        mo = int(row_dict['month'][0])
        dy = int(row_dict['day'][0])
        hr = int(row_dict['hour'][0])
        mi = int(row_dict['minute'][0]) if 'minute' in row_dict else 0 # Default minute if missing
        lat_val = float(row_dict['latitude'][0])
        lon_val = float(row_dict['longitude'][0])
        alt_val = float(row_dict['altitude'][0]) # Ensure altitude column name matches
    elif all(v is not None for v in [year, month, day, hour, lat, lon, alt]):
        yr, mo, dy, hr, mi = year, month, day, hour, minute if minute is not None else 0
        lat_val, lon_val, alt_val = lat, lon, alt
    else:
        raise ValueError("Must provide either df_row or individual year, month, day, hour, lat, lon, alt.")

    # --- Prepare Time Information ---
    try:
        start_datetime_utc = pydt.datetime(yr, mo, dy, hr, mi, tzinfo=pydt.timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid start date/time components: {yr}-{mo}-{dy} {hr}:{mi} - {e}")

    start_yr_2digit = start_datetime_utc.strftime('%y') # 2-digit year
    start_mo_2digit = start_datetime_utc.strftime('%m')
    start_dy_2digit = start_datetime_utc.strftime('%d')
    start_hr_2digit = start_datetime_utc.strftime('%H')
    start_mi_2digit = start_datetime_utc.strftime('%M')

    # --- Prepare Location Information ---
    lat_str = f"{lat_val: .4f}" # Format with space padding and 4 decimals
    lon_str = f"{lon_val: .4f}"
    alt_str = f"{alt_val:.1f}"   # Format with 1 decimal

    # --- Prepare Met File Information ---
    if isinstance(met, str): met = [met]
    if isinstance(metpath, str): metpath = [metpath]
    if len(met) != len(metpath):
        raise ValueError("Number of 'met' types must match number of 'metpath' entries.")

    num_met_types = len(met)
    # Calculate number of files needed per type
    abs_duration_days = math.ceil(abs(duration) / 24)
    # Need data from start_time + duration back/forward + buffer
    nmet_files_per_type = abs_duration_days + met_days_buffer

    # --- Prepare Output Strings ---
    lines = []
    # Line 1: Start Time (YY MM DD HH)
    lines.append(f"{start_yr_2digit} {start_mo_2digit} {start_dy_2digit} {start_hr_2digit}")
    # Line 2: Number of starting locations
    lines.append(f"{nlocations}")
    # Line(s) 3+: Starting Location(s) (LAT LON HGT(m))
    # Assuming nlocations = 1 for this example based on R code usage
    lines.append(f"{lat_str} {lon_str} {alt_str}")
    # Line 4: Run duration (hours)
    lines.append(f"{duration}")
    # Line 5: Vertical motion option (0=data, 3=isentropic, ...)
    lines.append(f"{vertical_motion}")
    # Line 6: Top of model domain (meters)
    lines.append(f"{top_model_domain:.1f}")

    # Line(s) 7+: Meteorology File List
    lines.append(f"{num_met_types}") # Number of met file types
    met_file_entries = []
    current_date = start_datetime_utc if duration >= 0 else start_datetime_utc + pydt.timedelta(hours=duration) # Start date for met file search
    # Go back/forward in time to collect necessary met file names
    # HYSPLIT reads met files backwards from the run start time for back trajectories
    met_search_start_date = start_datetime_utc if duration >=0 else start_datetime_utc + pydt.timedelta(hours=duration)
    # Search direction depends on trajectory direction
    day_step = pydt.timedelta(days=1) if duration >=0 else pydt.timedelta(days=-1)

    for type_idx in range(num_met_types):
         met_type_name = met[type_idx]
         met_type_path = Path(metpath[type_idx])
         lines.append(str(met_type_path / '%Y%m%d')) # Base path structure for the type
         lines.append(f"{met_type_name}_%Y%m%d") # Base filename structure for the type
         # Generate file list for this type (this is a placeholder, HYSPLIT might use these base paths/names directly)
         # HYSPLIT often expects you to just provide the base path and filename pattern.
         # The R code *explicitly* lists files - this might be needed for some setups.
         # Let's replicate the R code's explicit listing:
         # met_listing_date = start_datetime_utc # Date used for listing files
         # for day_offset in range(nmet_files_per_type):
         #      file_date = met_listing_date + day_offset * day_step # Adjust date based on offset
         #      year_str = file_date.strftime('%Y')
         #      date_str = file_date.strftime('%Y%m%d')
         #      # Construct path and filename based on common patterns
         #      # These patterns need to match your actual met file naming convention!
         #      met_dir = met_type_path / year_str
         #      if met_type_name == "nams": # Example from R
         #          met_fname = f"{date_str}_hysplit.t00z.namsa"
         #      elif met_type_name == "gfs0p25": # Example from R
         #          met_fname = f"{date_str}_gfs0p25"
         #      else: # Generic fallback
         #           met_fname = f"{met_type_name}_{date_str}" # Guessing
         #      met_file_entries.append(str(met_dir.resolve())) # Full directory path
         #      met_file_entries.append(met_fname)               # Filename

    # Add the explicit file list if generated
    # lines.extend(met_file_entries) # Commented out: Use base path/name method first

    # Line 8: Number of pollutant species
    lines.append(f"{ngases}")
    # Line 9: Pollutant name(s)
    lines.append(f"{gas}") # Assuming single gas for simplicity based on default
    # Line 10: Emission rate (kg/hr)
    lines.append(f"{emissions_rate:.1f}") # Format emission rate
    # Line 11: Emission duration (hours)
    lines.append(f"{hour_emissions:.2f}")
    # Line 12: Release start time (YYMMDDHHmm) - relative to simulation start
    release_start_str = f"{start_yr_2digit}{start_mo_2digit}{start_dy_2digit}{start_hr_2digit}{start_mi_2digit}"
    lines.append(release_start_str)

    # --- Concentration Grid Definition ---
    lines.append(f"{nsim_grids}")
    # Grid 1 definition (repeat if nsim_grids > 1)
    lines.append(f"{center_conc_grids[0]:.1f} {center_conc_grids[1]:.1f}") # Center Lat Lon
    lines.append(f"{grid_spacing[0]:.2f} {grid_spacing[1]:.2f}") # Spacing Lat Lon
    lines.append(f"{grid_span[0]:.1f} {grid_span[1]:.1f}") # Span Lat Lon
    lines.append(str(Path(output_dir).resolve())) # Output directory path
    lines.append(f"{nconc}") # Output filename base

    # --- Vertical Levels ---
    lines.append(f"{nvert_levels}")
    lines.append(" ".join(f"{h:.1f}" for h in height_vert_levels)) # Heights of levels

    # --- Sampling Timing ---
    # Default uses run start/end if None provided
    def format_sampling_time(time_spec):
        if time_spec is None or all(t == 0 for t in time_spec): return "00 00 00 00 00"
        if len(time_spec) == 5: return f"{time_spec[0]:02d} {time_spec[1]:02d} {time_spec[2]:02d} {time_spec[3]:02d} {time_spec[4]:02d}"
        raise ValueError("Invalid sampling_time format. Use [YY,MM,DD,HH,MM] or None/[0,0,0,0,0].")

    lines.append(format_sampling_time(sampling_start_time))
    lines.append(format_sampling_time(sampling_end_time))
    if len(sampling_interval_type) != 3: raise ValueError("sampling_interval_type needs 3 elements [Type, HH, MM]")
    lines.append(f"{sampling_interval_type[0]:02d} {sampling_interval_type[1]:02d} {sampling_interval_type[2]:02d}")

    # --- Deposition ---
    lines.append(f"{npol_depositing}")
    # Parameters for the *first* depositing pollutant (repeat if npol > 1)
    lines.append(" ".join(f"{p:.1f}" for p in particle_params)) # Particle: Dia(um) Dens(g/cc) Shape(0-1)
    lines.append(" ".join(f"{p:.1f}" for p in dmwsrdre))      # Gas Dry Dep: DepVel(m/s) MW(g/mol) SurfRea(0-1) Diffus(cm2/s) Henry(M/a)
    lines.append(" ".join(f"{p:.1f}" for p in wrhcicbc))      # Gas Wet Dep: Henry InCloud(1/s) BelowCloud(1/s)
    lines.append(f"{radiactive_decay:.1f}")                   # Radioactive Decay: Half-life (days)
    lines.append(f"{pol_res:.1f}")                             # Resuspension Factor (1/m)

    # --- Write to File ---
    try:
        control_path = Path(control_filename)
        with control_path.open('w') as f_out:
            for line in lines:
                f_out.write(line + '\n')
        print(f"Successfully wrote CONTROL file to: {control_path.resolve()}")
    except IOError as e:
        print(f"Error writing CONTROL file {control_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred writing CONTROL file: {e}")


# --- obs_hysplit_setup ---
def obs_hysplit_setup(
    # Key STILT/Dispersion parameters
    idsp: int = 2,          # 1:HYSPLIT, 2:STILT particle dispersion
    ichem: int = 8,         # 8:STILT mode (density+varying layer), 0:none, 6:density only, 9:varying layer only
    veght: float = 0.5,     # Footprint tally height (<1: frac PBL, >1: m AGL)
    outdt: int = 15,        # Particle output interval (min), 0=each step, <0=none
    krand: int = 4,         # Random number generator option
    # Other dispersion/physics parameters
    capemin: float = 500.0, # Convection trigger CAPE (J/kg), -1=none, -2=Grell
    vscales: float = -1.0,  # Vertical Lagrangian time scale (sec) for stable PBL (-1 uses default calc)
    kbls: int = 1,          # PBL Stability source (1:fluxes, 2:wind/temp)
    kblt: int = 5,          # PBL Turbulence scheme (1:Beljaars, 2:Kanthar, 3:TKE, 5:Hanna)
    kmixd: int = 0,         # Mixed Layer Depth source (0:input, 1:temp, 2:TKE, 3:modRich)
    kmix0: float = 150.0,   # Minimum mixing depth (m)
    # Particle/Puff parameters
    initd: int = 0,         # Initial distribution (0:particle, 2:puff, 4:3D particle)
    numpar: int = 500,      # Number of particles/puffs released per cycle
    maxpar: int = 10000,    # Max particles/puffs allowed in simulation (adjust based on memory)
    # Output variables for PARTICLE_STILT.DAT
    varsiwant: List[str] = ['TIME','INDX', 'LATI','LONG','ZAGL', 'ZSFC', # Default STILT-like vars
                            'FOOT','SAMT', 'TEMP','DSWF','MLHT','DENS',
                            'DMAS','SIGW','TLGR'],
    # ivmax: Optional[int] = None, # Number of vars (calculated from varsiwant if None)
    # Extra parameters to add
    extra_params: Optional[Dict[str, Any]] = None,
    setup_filename: str = "SETUP.CFG" # Output filename
) -> None:
    """
    Creates a HYSPLIT SETUP.CFG namelist file.

    Args:
        idsp: Particle dispersion scheme (1: HYSPLIT, 2: STILT).
        ichem: Chemistry/output mode (8 for STILT mixing ratio & varying layer).
        veght: Height for footprint calculation in PARTICLE_STILT.DAT.
        outdt: Output frequency (minutes) for PARTICLE.DAT/PARTICLE_STILT.DAT.
        krand: Random number generator option.
        capemin: CAPE threshold for enhanced mixing or convection scheme.
        vscales: Vertical Lagrangian time scale factor.
        kbls: Boundary layer stability source.
        kblt: Boundary layer turbulence parameterization.
        kmixd: Mixed layer depth source.
        kmix0: Minimum mixing depth (m).
        initd: Initial particle distribution type.
        numpar: Number of particles released per cycle.
        maxpar: Maximum number of particles in simulation.
        varsiwant: List of variable names to write to PARTICLE_STILT.DAT.
        extra_params: Dictionary of additional namelist parameters to include.
        setup_filename: Name of the output SETUP.CFG file.

    Returns:
        None. Writes the file directly.
    """
    ivmax = len(varsiwant) # Number of variables is determined by the list length

    # Format the varsiwant list for the namelist file
    vars_str = ", ".join(f"'{v.upper()}'" for v in varsiwant) # Uppercase and quote

    lines = []
    lines.append("&SETUP") # Start of namelist

    # Add standard parameters with comments (optional)
    lines.append(f" idsp    = {idsp},       ! Dispersion: 1=HYSPLIT, 2=STILT")
    lines.append(f" ichem   = {ichem},       ! Output mode: 0=none, 6=dens, 8=STILT, 9=varlayer")
    lines.append(f" veght   = {veght:.1f},    ! Footprint height: <1 frac PBL, >1 m AGL")
    lines.append(f" outdt   = {outdt},       ! Particle output interval (min)")
    lines.append(f" krand   = {krand},       ! Random number generator option")
    lines.append(f" capemin = {capemin:.1f}, ! Convection CAPE threshold (J/kg) or scheme (-1,-2)")
    lines.append(f" vscales = {vscales:.1f},  ! Vertical Lagrangian time scale factor")
    lines.append(f" kbls    = {kbls},       ! PBL Stability source (1=flux, 2=wind/T)")
    lines.append(f" kblt    = {kblt},       ! PBL Turbulence scheme (1=Belj, 2=Kant, 3=TKE, 5=Hanna)")
    lines.append(f" kmixd   = {kmixd},       ! Mixed Layer source (0=input, 1=T, 2=TKE, 3=Ri)")
    lines.append(f" kmix0   = {kmix0:.1f},  ! Minimum Mixing Depth (m)")
    lines.append(f" initd   = {initd},       ! Initial distribution (0=part, 2=puff)")
    lines.append(f" numpar  = {numpar},    ! Particles released per cycle")
    lines.append(f" maxpar  = {maxpar},    ! Maximum particles in simulation")
    lines.append(f" ivmax   = {ivmax},       ! Number of variables for PARTICLE_STILT.DAT")
    lines.append(f" varsiwant = {vars_str},") # Add the formatted variable list

    # Add extra parameters if provided
    if extra_params:
        for key, value in extra_params.items():
            # Basic formatting (handle strings vs numbers)
            if isinstance(value, str):
                 lines.append(f" {key.lower()} = '{value}',")
            elif isinstance(value, bool):
                 lines.append(f" {key.lower()} = .{ 'TRUE.' if value else 'FALSE.'},") # Fortran logical
            else: # Assume numeric
                 lines.append(f" {key.lower()} = {value},")

    # Ensure the last non-comment line ends correctly (remove trailing comma if needed)
    # Find last non-comment line
    last_param_line_index = -1
    for i in range(len(lines) - 1, 0, -1):
        if not lines[i].strip().startswith('!'):
            last_param_line_index = i
            break
    if last_param_line_index > 0:
         lines[last_param_line_index] = lines[last_param_line_index].rstrip(',') # Remove trailing comma

    lines.append("/") # End of namelist

    # --- Write to File ---
    try:
        setup_path = Path(setup_filename)
        with setup_path.open('w') as f_out:
            for line in lines:
                f_out.write(line + '\n')
        print(f"Successfully wrote SETUP.CFG file to: {setup_path.resolve()}")
    except IOError as e:
        print(f"Error writing SETUP.CFG file {setup_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred writing SETUP.CFG file: {e}")


# --- obs_hysplit_ascdata ---
def obs_hysplit_ascdata(
    llc: Tuple[float, float] = (-90.0, -180.0), # Lower Left Corner Lat, Lon
    spacing: Tuple[float, float] = (1.0, 1.0),   # Grid spacing Deg Lat, Deg Lon
    n: Tuple[int, int] = (180, 360),             # Number of grid points Lat, Lon
    landusecat: int = 2,                         # Default land use category
    rough: float = 0.2,                          # Default roughness length (m)
    bdyfiles_dir: str = '../bdyfiles/',          # Directory for boundary files (land use, roughness)
    ascdata_filename: str = "ASCDATA.CFG"        # Output filename
) -> None:
    """
    Creates a HYSPLIT ASCDATA.CFG file, defining default surface characteristics.

    Args:
        llc: Tuple of (latitude, longitude) for the lower-left corner of the default grid.
        spacing: Tuple of (latitude spacing, longitude spacing) in degrees.
        n: Tuple of (number of latitude points, number of longitude points).
        landusecat: Default land use category if file not found.
        rough: Default roughness length (meters) if file not found.
        bdyfiles_dir: Path (relative or absolute) to the directory containing
                      landuse and roughness data files (e.g., LANDUSE.ASC, ROUGHLEN.ASC).
                      Needs the trailing slash in HYSPLIT.
        ascdata_filename: Name of the output ASCDATA.CFG file.

    Returns:
        None. Writes the file directly.
    """
    lines = []
    # Line 1: Lower Left Corner
    lines.append(f"{llc[0]:<8.1f} {llc[1]:<8.1f}   lat/lon of lower left corner (last record in file)")
    # Line 2: Grid Spacing
    lines.append(f"{spacing[0]:<8.1f} {spacing[1]:<8.1f}   lat/lon spacing in degrees between data points")
    # Line 3: Number of Points
    lines.append(f"{n[0]:<8d} {n[1]:<8d}   lat/lon number of data points")
    # Line 4: Default Land Use Category
    lines.append(f"{landusecat:<15d} default land use category")
    # Line 5: Default Roughness Length
    lines.append(f"{rough:<15.1f} default roughness length (meters)")
    # Line 6: Boundary Files Directory (ensure trailing slash)
    bdy_path = Path(bdyfiles_dir)
    # HYSPLIT often expects paths relative to the working dir or specific structure.
    # Using the provided string directly, ensuring trailing slash.
    bdy_dir_str = str(bdy_path)
    if not bdy_dir_str.endswith(('/', '\\')):
        bdy_dir_str += os.sep
    lines.append(f"'{bdy_dir_str}'" + " "*(15-len(bdy_dir_str)-2) + " directory location of data files") # Pad for alignment

    # --- Write to File ---
    try:
        asc_path = Path(ascdata_filename)
        with asc_path.open('w') as f_out:
            for line in lines:
                f_out.write(line + '\n')
        print(f"Successfully wrote ASCDATA.CFG file to: {asc_path.resolve()}")
    except IOError as e:
        print(f"Error writing ASCDATA.CFG file {ascdata_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred writing ASCDATA.CFG file: {e}")