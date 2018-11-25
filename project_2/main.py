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
from project_2.infrastructure.configuration import Configuration
from project_2.infrastructure.filewriter import generate_vtf


def main():
    """
    The main functionality of the project
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mesh", help="Load mesh", action='store_true')
    parser.add_argument('-s', "--solve", help="Solve the problem")
    parser.add_argument('-c', "--config", help="Path to the config")
    parser.add_argument('-o', "--output", help="Path to output file")
    args = parser.parse_args()

    if args.mesh:
        mesh = Mesh()
        mesh.loadMesh()
        mesh.plotMesh()
    if args.solve:
        print(args.config)
        time_start = time.time()
        mesh = Mesh()
        configuration = Configuration()
        configuration.loadconfig(args.config)
        mesh.loadMesh(args.solve)
        mesh.plotMesh()
        stress, ux, uy, uz = solve(mesh, configuration)
        time_vtf_begin = time.time()
        generate_vtf(args.output,mesh,stress,ux,uy,uz)
        time_vtf = time.time()-time_vtf_begin
        print("[Info] Writing output took "+str(time_vtf)+"s")
        time_total = time.time()-time_start
        print("[Info] Total time: "+str(time_total)+"s")


if __name__ == "__main__":
    main()
