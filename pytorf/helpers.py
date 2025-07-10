"""
Helper utility functions for the pytorf package.
"""

import os
import re
import math
import datetime as pydt
from pathlib import Path
import yaml
import warnings
import numpy as np
from typing import List, Union, Optional, Tuple, Any, Dict

# Import datatable conditionally or where needed
try:
    import datatable as dt
    from datatable import f, isna, Frame # f needs to be imported for obs_format
    DT_AVAILABLE = True
except ImportError:
    DT_AVAILABLE = False
    # Define dummy types if datatable not installed, useful for type hinting
    class Frame: pass # Dummy class
    class f: pass      # Dummy class

# Assuming pandas is used for obs_id2pos based on pd.DataFrame
try:
    import pandas as pd
except ImportError:
    class pd: # Dummy class if pandas is not available but code uses pd.DataFrame
        @staticmethod
        def DataFrame(*args, **kwargs):
            warnings.warn("pandas is not installed. DataFrame functionality in obs_id2pos will not work as expected.")
            return args[0] if args else {} # Return first arg (expected to be list of dicts) or empty dict

# --- File/String Helpers ---

def fex(filepath: Union[str, Path]) -> str:
    """Extracts the file extension without the leading dot."""
    return os.path.splitext(filepath)[1][1:]



# --- Numeric Helpers ---

