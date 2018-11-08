"""
Tools to plot scattered structures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import scipy.io as sio
import meshio
import trimesh

def plot_scatter_structure(mesh,ux,uy,uz):
    x = mesh.supports[:,0]
    y = mesh.supports[:, 1]
    z = mesh.supports[:, 2]
    c = v = np.squeeze(uz)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap("hot")
    cax = ax.scatter(x, y, z, v, s=50, c=c, cmap=cmhot)
    fig.colorbar(cax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def plot_stress(mesh,stress):
    varnr = mesh.tetraeders.shape[0]
    x = np.zeros(varnr)
    y = np.zeros(varnr)
    z = np.zeros(varnr)
    v = np.linalg.norm(stress[:,:],axis=1)
    #v = stress[:,2]
    for n in range(varnr):
        v0_coord = (mesh.supports[mesh.tetraeders[n,0],0],mesh.supports[mesh.tetraeders[n,0],1],mesh.supports[mesh.tetraeders[n,0],2])
        v1_coord = (mesh.supports[mesh.tetraeders[n,1],0],mesh.supports[mesh.tetraeders[n,1],1],mesh.supports[mesh.tetraeders[n,1],2])
        v2_coord = (mesh.supports[mesh.tetraeders[n,2],0],mesh.supports[mesh.tetraeders[n,2],1],mesh.supports[mesh.tetraeders[n,2],2])
        v3_coord = (mesh.supports[mesh.tetraeders[n,3],0],mesh.supports[mesh.tetraeders[n,3],1],mesh.supports[mesh.tetraeders[n,3],2])
        x[n] = (np.asarray(v0_coord)+np.asarray(v1_coord)+np.asarray(v2_coord)+np.asarray(v3_coord))[0]/4
        y[n] =  (np.asarray(v0_coord)+np.asarray(v1_coord)+np.asarray(v2_coord)+np.asarray(v3_coord))[1]/4
        z[n] =  (np.asarray(v0_coord)+np.asarray(v1_coord)+np.asarray(v2_coord)+np.asarray(v3_coord))[2]/4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap("hot")
    cax = ax.scatter(x, y, z, v, s=50, c=v, cmap=cmhot)
    fig.colorbar(cax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def plot_stress_meshed(mesh,stress):
    data = [
        go.Mesh3d(
            x=[0, 1, 2, 0],
            y=[0, 0, 1, 2],
            z=[0, 2, 0, 1],
            colorbar=go.ColorBar(
                title='z'
            ),
            colorscale=[[0, 'rgb(255, 0, 0)'],
                        [0.5, 'rgb(0, 255, 0)'],
                        [1, 'rgb(0, 0, 255)']],
            intensity=[0, 0.33, 0.66, 1],
            i=[0, 0, 0, 1],
            j=[1, 2, 3, 2],
            k=[2, 3, 1, 3],
            name='y',
            showscale=True
        )
    ]
    layout = go.Layout(
        xaxis=go.XAxis(
            title='x'
        ),
        yaxis=go.YAxis(
            title='y'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='/home/leon/Documents/RCI/TMA4220_NumPDE/3d-mesh-tetrahedron-python')


def export_matlab(mesh,stress):
    tetras = mesh.tetraeders
    points = mesh.supports
    values = stress
    cells = {
        "triangle": mesh.triangles
    }
    sio.savemat('solution.mat', {'tet': tetras,'X': points,'val': values})
    meshio.write_points_cells("testtest.stl", mesh.supports,cells)

def trimeshit(calc_mesh,stress):
    mesh = trimesh.load('testtest.stl')
    #v = np.linalg.norm(stress[:, :], axis=1)

    i = 0
    #print(np.shape(mesh.))
    print(mesh.faces[0,:])
    print(calc_mesh.triangles[0,:])
    for face in mesh.faces.T:
        mesh.visual.face_colors[face] = 10000
        i+=1
    mesh.show()

def trisurfit(mesh,stress):
    print("[Info] Plotting")
    plotly.tools.set_credentials_file(username='Jimpachnet', api_key='')
    outers = mesh.triangles
    ptz = mesh.supports
    print(np.shape(outers))

    xer = ptz[:,0]
    yer = ptz[:,1]
    zer = ptz[:,2]

    ier = outers[:,0]
    jer = outers[:,1]
    ker = outers[:,2]

    v = np.linalg.norm(stress[:, :], axis=1)
    ver = np.zeros((np.shape(xer)[0], 1))

    for i in range(mesh.supports.shape[0]):
        founds = np.argwhere(outers == i)
        founds = founds[:,0]
        nrr = np.shape(founds)[0]
        #if nrr >0:
        #    ver[i] = np.max(v[founds])
        for j in range(nrr):
            ver[i] += v[founds[j]]/nrr

    ver = np.squeeze(ver)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap("hot")
    cax = ax.scatter(xer, yer, zer, ver, s=50, c=ver, cmap=cmhot)
    fig.colorbar(cax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


    exit(0)




    data = [
        go.Mesh3d(
            x=xer,
            y=yer,
            z=zer,
            colorbar=go.ColorBar(
                title='z'
            ),
            colorscale=[[0, 'rgb(255, 0, 0)'],
                        [0.5, 'rgb(0, 255, 0)'],
                        [1, 'rgb(0, 0, 255)']],
            intensity=ver,
            i=ier,
            j=jer,
            k=ker,
            name='y',
            showscale=True
        )
    ]
    layout = go.Layout(
        xaxis=go.XAxis(
            title='x'
        ),
        yaxis=go.YAxis(
            title='y'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='3d-mesh-tetrahedron-python')