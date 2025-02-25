# Python Tools for Obspack, Receptors and Footprints (pytorf) - in development

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/noaa-gml/pytorf)
[![python-check](https://github.com/noaa-gml/pytorf/actions/workflows/python-app.yml/badge.svg)](https://github.com/noaa-gml/pytorf/actions/workflows/python-app.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/noaa-gml/pytorf)

[NOAA Obspack](https://gml.noaa.gov/ccgg/obspack/) is a collection of greenhouse gases observations


<img src="https://github.com/noaa-gml/rtorf/blob/main/man/figures/logo.png?raw=true" align="right" alt="" width="220" />

pytorf is a Python package designed for reading NetCDF files into Pandas DataFrames. It provides a convenient function, `obs_read_nc`, which allows users to extract and manipulate observational data from NetCDF files efficiently.


## Installation

`pytorf` only depends on `pandas` and `netCDF4`, which is basically parallel C, 
so it can be installed in any machine.

To install the package, you can clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/noaa-gml/pytorf.git
cd pytorf
pip install -r requirements.txt
```

Alternatively, you can install the package directly from the source:

```bash
pip install .
```
## R version 

Check the R version [rtorf]( https://github.com/noaa-gml/rtorf)

## Usage

To use the `obs_read_nc` function, you can import it from the package and call it with the appropriate parameters. Here is a basic example:

```python
import pytorf

# Assuming you have an index DataFrame ready
`obs_summary` takes the input of the obspack directory and return a DataFrame
categories=["aircraft-pfp",
            "aircraft-insitu", 
            "surface-insitu",
            "tower-insitu", 
            "aircore",
            "surface-pfp",
            "shipboard-insitu",
            "flask"]
obs = "Z:/torf/obspack_ch4_1_GLOBALVIEWplus_v5.1_2023-03-08/data/nc/"
index = obs_summary(obs = obs)

data = obs_read_nc(index, categories="flask", solar_time=False, as_list=False, verbose=True)
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

## Special thanks to all the contributors

[![Contributors](https://contrib.rocks/image?repo=noaa-gml/pytorf)](https://github.com/noaa-gml/pytorf/graphs/contributors)