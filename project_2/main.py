#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np

from project_2.mesh.mesh import Mesh


def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    args = parser.parse_args()

    if args.mesh:
        mesh = Mesh()
        mesh.loadTest()


if __name__ == "__main__":
    main()
