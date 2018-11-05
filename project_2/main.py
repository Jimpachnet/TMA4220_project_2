#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np

from project_2.infrastructure.mesh import Mesh
from project_2.solvers.solver_quadpy import solve_quadpy
from project_2.visualization.scatter_structure import plot_scatter_structure


def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    parser.add_argument('-sqp', "--solvequadpy", help="Uses quadpy to solve the problem", action='store_true')
    args = parser.parse_args()

    if args.mesh:
        mesh = Mesh()
        mesh.loadMesh()
        mesh.plotMesh()
    if args.solvequadpy:
        mesh = Mesh()
        mesh.loadMesh("/home/leon/Documents/RCI/TMA4220_NumPDE/models/export/cube_2.med")
        mesh.plotMesh()
        ux,uy,uz = solve_quadpy(mesh)
        plot_scatter_structure(mesh,ux,uy,uz)


if __name__ == "__main__":
    main()
