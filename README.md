# Python Tools for Observations, Receptors and Footprints (pytorf) - in development

<img src="https://github.com/noaa-gml/rtorf/blob/main/man/figures/logo.png?raw=true" align="right" alt="" width="220" />

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/noaa-gml/pytorf)
[![python-check](https://github.com/noaa-gml/pytorf/actions/workflows/python-app.yml/badge.svg)](https://github.com/noaa-gml/pytorf/actions/workflows/python-app.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/noaa-gml/pytorf)

[NOAA Obspack](https://gml.noaa.gov/ccgg/obspack/) is a collection of greenhouse gases observations

pytorf is a Python package designed for reading NetCDF files into Pandas DataFrames. It provides a convenient function, `obs_read_nc`, which allows users to extract and manipulate observational data from NetCDF files efficiently.

Python package for reading and processing atmospheric observation data, particularly NOAA ObsPack files. This package is a port of functionality from the R package [rtorf](https://github.com/noaa-gml/rtorf).

## Description
pytorf provides tools to:

- Index and summarize ObsPack data files (.txt, .nc).
- Read observation data and associated metadata from ObsPack files.
- Process time information (add UTC timestamps, calculate local time, etc.).
- Aggregate observation data based on specified criteria.
- Generate HYSPLIT configuration files (CONTROL, SETUP.CFG, ASCDATA.CFG).
- Helper functions for common data manipulation and formatting tasks.

## Installation

It is highly recommended to install pytorf within a virtual environment.


```bash
git clone https://github.com/noaa-gml/pytorf.git
cd pytorf
pip install .
```
Then

### Dependencies:

The package relies on several core Python libraries, which will be installed automatically via pip:

- datatable
- netCDF4
- numpy
- PyYAML
- matplotlib

Note: Installing python-datatable might require specific system prerequisites on some platforms (like Linux). Please refer to the official datatable documentation if you encounter installation issues.
Usage
To use the obs_read_nc function, you can import it from the package and call it with the appropriate parameters. Here is a basic example:

Assuming you have an index DataFrame ready

`obs_summary` takes the input of the obspack directory and return a DataFrame


```python
import pytorf

categories = [
    "aircraft-pfp",
    "aircraft-insitu", #"aircraft-flask", # If flask is not in CH4 data
    "surface-insitu", #"surface-flask",
    "surface-pfp",
    "tower-insitu",
    "aircore",
    "shipboard-insitu" #, "shipboard-flask"
]
print(f"categories:\n {categories}")


obspack_ch4_dir = Path("/PATH/obspack/obspack_ch4_1_GLOBALVIEWplus_v5.1_2023-03-08/data/nc/")

print(f"Directory exists? {obspack_ch4_dir.is_dir()}")

index_ch4 = pytorf.obs_summary(
    obs_path=obspack_ch4_dir,
    categories=categories,
    file_pattern='*.nc', # Specify NetCDF files
    verbose=True
)

```
Number of files found: 429

File counts by assigned sector:
   | sector                  count      N
   | str64                   int64  int32
-- + ----------------------  -----  -----
 0 | aircore                     1     NA
 1 | aircraft-insitu            15     NA
 2 | aircraft-pfp               40     NA
 3 | shipboard-insitu            1     NA
 4 | surface-insitu            174     NA
 5 | surface-pfp                33     NA
 6 | tower-insitu               51     NA
 7 | Total assigned sectors     NA    315
[8 rows x 3 columns]

```

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


## Special thanks to all the 

**contributors**

[![Contributors](https://contrib.rocks/image?repo=noaa-gml/pytorf)](https://github.com/noaa-gml/pytorf/graphs/contributors)

and

**Stargazers**

<p>
  <a href="https://github.com/noaa-gml/pytorf/stargazers">
    <img src="http://reporoster.com/stars/dark/noaa-gml/pytorf"/>
  </a>
</p>
