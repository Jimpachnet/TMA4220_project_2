#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a infrastructure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio

class Mesh:
    """
    Represents a infrastructure
    """

    def loadMesh(self, path = "/home/leon/Documents/RCI/TMA4220_NumPDE/models/export/cube.med"):
        print("[Info] Loading infrastructure")
        mesh = meshio.read(path,file_format='gmsh2')
        self.tetraeders = mesh.cells['tetra']
        self.triangles = mesh.cells['triangle']
        self.supports = -mesh.points
        print("[Info] Loaded " + str(self.supports.shape[0]) + " supports")
        print("[Info] Loaded " + str(self.tetraeders.shape[0]) + " simplices")
    def loadexamplemesh(self):
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
        print("[Info] Plotting infrastructure")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.supports[:,0], self.supports[:,1], self.supports[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
