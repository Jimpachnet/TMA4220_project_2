from setuptools import find_packages
from distutils.core import setup

setup(name='Project_2',
      version='0.1dev',
      description='Our solution to project 2 of the course TMA4220 in autumn semester 2018 at NTNU Trondheim',
      author='Leon Sievers, Thibault Gaudet',
      author_email='leon.sievers@tum.de',
      url='https://github.com/Jimpachnet/TMA4220_project_2',
      packages=find_packages(),
      install_requires=[
          'scipy',
          'matplotlib',
          'numpy',
          'tqdm',
      ],
     )
