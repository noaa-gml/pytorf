from setuptools import setup, find_packages

setup(
    name='pytorf',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for reading NetCDF files into DataFrames',
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