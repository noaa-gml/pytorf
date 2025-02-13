import pytest
import pandas as pd
from pytorf.obs_read_nc import obs_read_nc

def test_obs_read_nc_empty_index():
    with pytest.raises(ValueError, match="empty index"):
        obs_read_nc(pd.DataFrame(), categories="flask")

def test_obs_read_nc_valid_input():
    # Create a mock index DataFrame
    data = {
        "sector": ["flask", "flask"],
        "id": ["file1.nc", "file2.nc"],
        "agl": [100, 200],
        "n": [1, 2]
    }
    index = pd.DataFrame(data)

    # Mock the Dataset class from netCDF4
    class MockDataset:
        def __init__(self, file):
            self.variables = {
                "time_components": pd.DataFrame({
                    "year": [2023, 2023],
                    "month": [3, 3],
                    "day": [8, 8],
                    "hour": [0, 0],
                    "minute": [0, 0],
                    "second": [0, 0]
                }).values,
                "value": {"scale_comment": "some_scale"},
            }
            self.ncattrs = lambda: ["global_attr1", "global_attr2"]

    # Patch the Dataset class
    import netCDF4
    netCDF4.Dataset = MockDataset

    result = obs_read_nc(index, categories="flask", solar_time=False, as_list=False, verbose=False)

    assert isinstance(result, pd.DataFrame)
    assert "scale" in result.columns
    assert result["scale"].iloc[0] == "some_scale"