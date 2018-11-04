"""
Tools to plot scattered structures
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter_structure(mesh,ux,uy,uz):
    print(mesh.supports.shape)
    x, y, z, v = (np.random.random((4, 100)) - 0.5) * 15
    x = mesh.supports[:,0]
    y = mesh.supports[:, 1]
    z = mesh.supports[:, 2]
    c = v = ux

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap("hot")
    cax = ax.scatter(x, y, z, v, s=50, c=c, cmap=cmhot)

    plt.show()
