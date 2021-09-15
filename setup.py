#!/usr/bin/env python

import os
from setuptools import setup
from setuptools import find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(name='orth_optim',
      author='Mark Tuddenham',
      author_email='mark.tuddenham@southampton.ac.uk',
      version='1.0.0',
      description='Add option to orthogonalise gradients to pytorch optimisers',
      long_description=long_description,
      long_description_context_type='text/markdown',
      license='MIT',
      url='https://github.com/MarkTuddenham/Orthogonal-Optimisers',
      keywords=['optimisation', 'optimzation', 'pytorch'],
      packages=find_packages(where='src'),
      package_dir={"": "src"},
      install_requires=['torch'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
      )
