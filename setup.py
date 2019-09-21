#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

version = "0.1.0"

requirements = ['Click>=6.0', ]

setup(name='spiketag',
      version=version,
      description='spike-sorting packages for project Xike',
      author=['Chongxi Lai'],
      author_email='chongxi.lai@gmail.com',
      # url='http://github.com/all-umass/metric-learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      entry_points={
          'console_scripts': [
              'spiketag=spiketag.command:main',
          ],
      },
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'six',
          'cython',
          'numba',
          'numexpr',
          'vispy',
          'phy',
          'seaborn',
          'hdbscan',
          'ipyparallel'
      ],
      extras_require=dict(
          docs=['sphinx', 'numpydoc'],
          demo=['vispy'],
      ),
      packages=find_packages(include=['spiketag']),
      test_suite='test',
      keywords=[
          'Spike Sorting', 
          'Clustering'
      ])
