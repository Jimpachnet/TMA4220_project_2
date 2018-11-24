#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np
import time

from project_2.infrastructure.mesh import Mesh
from project_2.solvers.solver import solve
from project_2.visualization.scatter_structure import plot_scatter_structure,plot_stress,plot_stress_meshed,export_matlab,trimeshit,trisurfit
from project_2.infrastructure.configuration import Configuration
from project_2.infrastructure.filewriter import generate_vtf


def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    parser.add_argument('-s', "--solve", help="Solve the problem", action='store_true')
    args = parser.parse_args()
    basepath = "/home/leon/Documents/RCI/TMA4220_NumPDE/models/netgen/"

    if args.mesh:
        mesh = Mesh()
        mesh.loadMesh()
        mesh.plotMesh()
    if args.solve:
        time_start = time.time()
        mesh = Mesh()
        configuration = Configuration()
        configuration.loadconfig(basepath+"harbourbridge.ini")
        mesh.loadMesh(basepath+"harbourbridge_fine_fine.geo")
        #mesh.loadMesh(basepath + "cube.geo")
        mesh.plotMesh()
        stress, ux,uy,uz = solve(mesh, configuration)
        #plot_scatter_structure(mesh,ux,uy,uz)
        #plot_stress(mesh,stress)
        #plot_stress_meshed(mesh,stress)
        #export_matlab(mesh,stress,ux,uy,uz)
        time_vtf_begin = time.time()
        generate_vtf("/home/leon/Documents/RCI/TMA4220_NumPDE/results/output.vtf",mesh,stress,ux,uy,uz)
        time_vtf = time.time()-time_vtf_begin
        print("[Info] Writing output took "+str(time_vtf)+"s")
        #trimeshit(mesh,0)
        #trisurfit(mesh,stress)
        time_total = time.time()-time_start
        print("[Info] Total time: "+str(time_total)+"s")


if __name__ == "__main__":
    main()
