from setuptools import setup, find_packages

setup(
    name='PREDICT-Precipitation Estimation via Data Fusion of Integrated Technologies',
    version='0.0.1',
    description='Package to fuse data from Satellites, weather radars and commercial microwave links for more precise precipitation estimations',
    author='Rodrigo Zambrana',
    author_email='rodrizp@gmail.com',
    url='https://https://github.com/rainsmore/PREDICT',
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
   'pandas',
   'matplotlib',
   'numpy',
   'cython',
   'xarray',
   'scipy'
],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)