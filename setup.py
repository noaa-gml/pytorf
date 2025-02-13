from setuptools import setup, find_packages

setup(
    name='pytorf',
    version='0.1.0',
    author='Sergio Ibarra-Espinosa',
    author_email='sergio.ibarra-espinosa@noaa.gov',
    description='A package for reading Obspack NetCDF files into DataFrames',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'netCDF4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)