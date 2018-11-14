#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np

from project_2.infrastructure.mesh import Mesh
from project_2.solvers.solver_quadpy import solve_quadpy
from project_2.visualization.scatter_structure import plot_scatter_structure,plot_stress,plot_stress_meshed,export_matlab,trimeshit,trisurfit
from project_2.infrastructure.configuration import Configuration

def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    parser.add_argument('-sqp', "--solvequadpy", help="Uses quadpy to solve the problem", action='store_true')
    args = parser.parse_args()
    basepath = "/home/leon/Documents/RCI/TMA4220_NumPDE/models/netgen/"

    if args.mesh:
        mesh = Mesh()
        mesh.loadMesh()
        mesh.plotMesh()
    if args.solvequadpy:
        mesh = Mesh()
        configuration = Configuration()
        configuration.loadconfig(basepath+"harbourbridge.ini")
        mesh.loadMesh(basepath+"harbourbridge.geo")
        mesh.plotMesh()
        stress, ux,uy,uz = solve_quadpy(mesh,configuration)
        #plot_scatter_structure(mesh,ux,uy,uz)
        #plot_stress(mesh,stress)
        #plot_stress_meshed(mesh,stress)
        export_matlab(mesh,stress,ux,uy,uz)
        #trimeshit(mesh,0)
        #trisurfit(mesh,stress)


if __name__ == "__main__":
    main()
