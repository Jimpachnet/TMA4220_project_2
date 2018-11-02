#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np

from project_2.infrastructure.mesh import Mesh


def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    args = parser.parse_args()

    if args.mesh:
        mesh = Mesh()
        mesh.loadMesh()
        mesh.plotMesh()


if __name__ == "__main__":
    main()
