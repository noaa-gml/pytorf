
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
    
