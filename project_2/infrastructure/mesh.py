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

    def loadMesh(self, path = "/home/leon/Documents/RCI/TMA4220_NumPDE/models/export/gate.med"):
        print("[Info] Loading infrastructure")
        mesh = meshio.read(path)
        self.tetraeders = mesh.cells['tetra']
        self.supports = mesh.points
        print("[Info] Loaded " + str(self.supports.shape[0]) + " supports")
        print("[Info] Loaded " + str(self.tetraeders.shape[0]) + " simplices")

    def plotMesh(self):
        print("[Info] Plotting infrastructure")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.supports[:,0], self.supports[:,1], self.supports[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
