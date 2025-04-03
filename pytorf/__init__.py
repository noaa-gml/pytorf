# src/rtorf/__init__.py
"""
rtorf: A Python package for reading and processing NOAA ObsPack data.
"""

__version__ = "0.1.0" # Define package version

# Import key functions/classes to make them available at the package level
from .readers import obs_summary, obs_read, obs_read_nc, obs_meta, obs_read_csvy
from .processing import obs_addtime, obs_agg, obs_addltime, obs_addstime
from .hysplit import obs_hysplit_control, obs_hysplit_setup, obs_hysplit_ascdata
from .plotting import obs_plot
from .helpers import (
    fex, sr, obs_out, obs_trunc, obs_footname, obs_write_csvy,
    obs_format, obs_rbind, obs_freq, obs_roundtime,
    obs_index, obs_addzero # Include deprecated ones if desired
)
# Removed: from .invfile import obs_invfiles

# Optionally define __all__ to specify explicit public API for 'from rtorf import *'
# This controls what gets imported when a user does 'from rtorf import *'
__all__ = [
    # Readers
    'obs_summary', 'obs_read', 'obs_read_nc', 'obs_meta', 'obs_read_csvy',
    # Processing
    'obs_addtime', 'obs_agg', 'obs_addltime', 'obs_addstime',
    # HYSPLIT
    'obs_hysplit_control', 'obs_hysplit_setup', 'obs_hysplit_ascdata',
    # Plotting
    'obs_plot',
    # Helpers
    'fex', 'sr', 'obs_out', 'obs_trunc', 'obs_footname', 'obs_write_csvy',
    'obs_format', 'obs_rbind', 'obs_freq', 'obs_roundtime',
    # Inverse Modeling - Removed
    # Package Info
    '__version__'
]