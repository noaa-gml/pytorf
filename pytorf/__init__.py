# src/rtorf/__init__.py
"""
rtorf: A Python package for reading and processing NOAA ObsPack data.
"""

__version__ = "0.1.0" # Define package version

# Import key functions/classes to make them available at the package level
from .io import (
    obs_summary, obs_read, obs_read_nc, obs_meta, obs_read_csvy, obs_write_csvy
)
from .processing import (
    obs_agg, obs_freq, obs_rbind, obs_format, obs_id2pos
)
from .helpers import (
    fex, sr, obs_out, obs_trunc, obs_roundtime, obs_julian_py,
    obs_addtime, obs_addltime, obs_addstime, obs_footname
)
from .hysplit import obs_hysplit_control, obs_hysplit_setup, obs_hysplit_ascdata
from .plotting import obs_plot

# This controls what gets imported when a user does 'from rtorf import *'
__all__ = [
    # io
    'obs_summary', 'obs_read', 'obs_read_nc', 'obs_meta', 'obs_read_csvy',
    'obs_write_csvy',
    # processing
    'obs_agg', 'obs_freq', 'obs_rbind', 'obs_format', 'obs_id2pos',
    # helpers
    'fex', 'sr', 'obs_out', 'obs_trunc', 'obs_roundtime', 'obs_julian_py',
    'obs_addtime', 'obs_addltime', 'obs_addstime', 'obs_footname',
    # hysplit
    'obs_hysplit_control', 'obs_hysplit_setup', 'obs_hysplit_ascdata',
    # plotting
    'obs_plot',
    # Package Info
    '__version__'
]