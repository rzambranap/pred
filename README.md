# PREDICT
Precipitation Estimation via Data Fusion of Integrated Technologies

## Install

### On local, create conda environment
For some reason I cannot deduct, it only works on coda environments. We must first create an environment and at the same time install the dask module. it only works ok this way.
```bash
conda create --name envname
```

Activate it, and manually install the following dependencies
```bash
conda activate envname
conda install -c conda-forge ipykernel
conda install -c conda-forge xradar=0.2.0 lxml=4.9.2 matplotlib=3.7.1 cartopy=0.21.1
```

Then, once you've installed all of those libraries in your environment and cloned the precipitation_estimation package, go to the cloned directory and from there launch the following command

```bash
pip install -r requirements.txt
```
If the previous command fails, run it again to check everything is ok

And now you can install your package. For normal usage:

```bash
pip install .
```

### To update package

```bash
git pull
pip install .
```

For development usage:

```bash
pip install -e .
```

To check documentation enter following command with your environment activated
```bash
python -m pydoc -b
```
And then enter the package precipitation_estimation, there you can browse each module, and doc is being written as we speak.

## Data Format Conventions

### L0, L1, L2... Convention

We are loosely based on these NASA conventions: https://www.earthdata.nasa.gov/engage/open-data-services-and-software/data-information-policy/data-levels 

However we don't deal, as of yet, with that amount of data and their precision so we stay within the following confines:

- L0: Raw data files, with the nomenclatures, as defined by their creators, their own data types as well and pre-determined time aggregations
- L1R, L1S: These are easy to read files with the data in known formats. They still have their instrument-given variables, variable names and data structure but they are easy to read and have geo-referenced coordinates. Think of a radar file loaded into a netcdf with lat/lon coordinates for every range/azimuth pair. L1R are the most basi of L1 files, depending on the instrument, it will have its' whole data or a single time-step. L1S on the other hand are series of data of a predetermined length, usually one single day. Think of concatenated files for a whole day for one of the radar's elevations. These files are usually NETCDFs or CSVs.
- L2A, L2S: Derived geophysical variables at the same resolution and location as L1 source data. Same distinction for R, S.
- L3: Variables mapped on uniform space-time grid scales, usually with some completeness and consistency.
- L4: Merged products with attributes on how they were merged.

### CML Networks

- L0 files: Raw data for data as well as metadata files, they are in proprietary structures in varying formats (csv, xml, gzip...) our first goal is always to format them into our L1 data structures.
- L1 files: CSV files, *one-per-link-and-direction* with the following structure:

|                            time                            |            rsl            | ... |
|:----------------------------------------------------------:|:-------------------------:|:---:|
| _date time value for step in format "YYYY-MM-DDTHH:MM:SS"_ | _rsl level values in dBm_ | ... |
|                     2023-01-20T02:20:15                    |           -23.4           | ... |
|                             ...                            |            ...            | ... |

Their nomenclature will be as follows *L1R_unique_link_id.csv*
This may seem like a lot of files for the task at hand but this will ensure capped-size data files, in the worst imaginable case, 15 years of data on a 20 seconds link with 6 64-bit values and complete datetimes would weight 1.6 Gb, whereas a 1 year csv with 4000 links with single values would clock at 50.5 Gb. Furthermore this would allow us to select the files to read based on bounding boxes applied to our metadata file, instead of reading loads of data for no reason.
- L2 files: CSV files, *one-per-link-and-direction* with the following structure:

|                            time                            |                rain_rate               |                                   uncertainty                                  |
|:----------------------------------------------------------:|:--------------------------------------:|:------------------------------------------------------------------------------:|
| _date time value for step in format "YYYY-MM-DDTHH:MM:SS"_ | _calculated rain rate levels in mm/hr_ | _uncertainty in dBm (calculated based on quantization or min v max levels...)_ |
|                     2023-01-20T02:20:15                    |                  2.35                  |                                      0.078                                     |
|                             ...                            |                   ...                  |                                       ...                                      |

Their nomenclature will be as follows *L2R_unique_link_id.csv*

- Metadata file structure (fichier réseau):

|                   unique_link_id                  |      direction     |     length     |      frequency     |      lon_a      |      lon_b      |      lat_a      |      lat_b      |        quantization        |                          timestep_length                          | ... |
|:-------------------------------------------------:|:------------------:|:--------------:|:------------------:|:---------------:|:---------------:|:---------------:|:---------------:|:--------------------------:|:-----------------------------------------------------------------:|:---:|
| _unambiguous and unique machine readable link id_ | _A to B or B to A_ | _length in km_ | _frequency in GHz_ | _lon A antenna_ | _lon B antenna_ | _lat A antenna_ | _lat B antenna_ | _quantization step in dBm_ | _lenght between time steps in "PnYnMnDTnHnMnS" time delta format_ | ... |
|             12321.545.562_1235.321.212            |       545 A-B      |      17.35     |        18.0        |     -35.536     |      -35.23     |      5.086      |      4.9502     |             0.1            |                               PD15M                               | ... |

## Example Notebook

In the folder notebooks there is a notebook which explains how to do a few things with it, the How_to_use notebook

## Test

By default, python provide a unittest package, which allows you to create unit tests. Unit tests are a way for you to test pieces of your code by scripting a process ( within your unit test ) where you give inputs to one of your function you want to test, make it run with it, collect outputs, and assert that this output matches your expectations. You can have a look on the unit test example in the tests folder.

To launch unit testing, hit the following command:

```bash
python -m unittest
```

This will launch all unit tests that are presents in tests/ folder.

Here is a documentation about it: [unittest documentation](https://docs.python.org/fr/3/library/unittest.html).

## Versioning

If you want to version it with git, go to gitlab account, click on the new repository button, and follow the instructions in the "already existing folder part".


### git commands you should know

to have an up to date version of code, pull sources from gitlab using:

```bash
git pull
```

to add your changes into a commit:

```bash
git add <file>
```

to commit :

```bash
git commit -m "<insert your commit message here>"
```
and to push on gitlab

```bash
git push
```
