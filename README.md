# README.md

# pytorf

pytorf is a Python package designed for reading NetCDF files into Pandas DataFrames. It provides a convenient function, `obs_read_nc`, which allows users to extract and manipulate observational data from NetCDF files efficiently.

## Installation

To install the package, you can clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/yourusername/pytorf.git
cd pytorf
pip install -r requirements.txt
```

Alternatively, you can install the package directly from the source:

```bash
pip install .
```

## Usage

To use the `obs_read_nc` function, you can import it from the package and call it with the appropriate parameters. Here is a basic example:

```python
from pytorf.obs_read_nc import obs_read_nc

# Assuming you have an index DataFrame ready
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