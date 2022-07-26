#!/usr/bin/env python

from distutils.core import setup

setup(
    name='pyseis',
    version='1.0',
    description='Generate synthetic seismic data',
    author='Stuart Farris',
    author_email='sfarris@sep.stanford.edu',
    url='http://cees-gitlab.stanford.edu/sfarris/pyseis',
    packages=['pyseis'],
    install_requires=[
        'holoviews==1.14.6', 'opensimplex==0.4.2', 'numpy==1.20.1',
        'numba==0.54.1', 'matplotlib==3.3.4', 'scipy==1.6.2', 'h5py==2.10.0',
        'selenium==4.3.0', 'tbb==2021.1.1'
    ],
)
