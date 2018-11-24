#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a infrastructure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio
import time


class Mesh:
    """
    Represents a topology
    """

    def loadMesh(self, path="/home/leon/Documents/RCI/TMA4220_NumPDE/models/export/cube.med"):
        """
        Loads a mesh file
        :param path: The path to the file
        """
        print("[Info] Loading infrastructure")
        mesh_load_start = time.time()
        mesh = meshio.read(path, file_format='gmsh2')
        self.tetraeders = mesh.cells['tetra']
        self.triangles = mesh.cells['triangle']
        self.supports = mesh.points
        self.supports = self.supports - np.min(self.supports, axis=0)
        mesh_load_time = time.time() - mesh_load_start

        print("[Info] Loaded " + str(self.supports.shape[0]) + " supports")
        print("[Info] Loaded " + str(self.tetraeders.shape[0]) + " simplices")
        print("[Info] Loading mesh took " + str(mesh_load_time) + "s")

    def loadexamplemesh(self):
        """
        Load a hardcoded toy mesh
        """
        self.supports = np.array([
            (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
            (0, 0, 12), (2, 0, 12), (2, 2, 12), (0, 2, 12),
        ])
        self.tetraeders = np.array([
            [21, 39, 38, 52],
            [9, 50, 2, 3],
            [12, 45, 15, 54],
            [39, 43, 20, 52],
            [41, 45, 24, 54]
        ])
        print(np.shape(self.supports))

    def plotMesh(self):
        """
        Plots the nodes
        """
        print("[Info] Plotting nodes")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.supports[:, 0], self.supports[:, 1], self.supports[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
