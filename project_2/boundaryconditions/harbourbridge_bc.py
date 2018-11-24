"""
Applies boundary conditions for the harbour bridge
"""

import numpy as np
from project_2.infrastructure.affine_transformation_2d import AffineTransformation2D


def apply_bc(mesh, K, b):
    """
    Applies the BC to the structure
    :param mesh: The mesh
    :param K: The stiffness matrix
    :param b: the linear form
    :return: Linear form, Stiffness matrix and counts
    """

    nr = np.shape(mesh.supports)[0]
    bc_count = 0
    nm_count = 0

    #Activate to apply Neumann
    if True:
        F = 1000000000 * 9.81
        refVal = 0.16666666666666666

        #Different Meshes
        pointsarr = np.array([[95751, 83578, 50820]])

        # pointsarr = np.array([[6138,6332,6334 ],
        #                      [6138, 6242,6334],
        #                      [6402,6242,6334],
        #                      [6402,6465,6334],
        #                      [6470,6334,6465],
        #                      [6332,6334,6470]])

        atraf2d = AffineTransformation2D()
        A = 0
        for ft in pointsarr:
            v0_coord = (mesh.supports[ft[0], 0], mesh.supports[ft[0], 1])
            v1_coord = (mesh.supports[ft[1], 0], mesh.supports[ft[1], 1])
            v2_coord = (mesh.supports[ft[2], 0], mesh.supports[ft[2], 1])
            atraf2d.set_target_cell(v0_coord, v1_coord, v2_coord)
            A += np.abs(atraf2d.get_determinant()) / 2

        p = F / A

        for ft in pointsarr:
            v0_coord = (mesh.supports[ft[0], 0], mesh.supports[ft[0], 1])
            v1_coord = (mesh.supports[ft[1], 0], mesh.supports[ft[1], 1])
            v2_coord = (mesh.supports[ft[2], 0], mesh.supports[ft[2], 1])
            atraf2d.set_target_cell(v0_coord, v1_coord, v2_coord)
            for brr in ft:
                b[brr * 3 + 2] += p * refVal * np.abs(atraf2d.get_determinant())

    # BC Dirichlet

    for i in range(nr):
        if True:
            if mesh.supports[i, 2] == 0:
                bc_count += 1
                K[i * 3, :] = np.zeros((1, nr * 3))
                K[i * 3, i * 3] = 1
                K[i * 3 + 1, :] = np.zeros((1, nr * 3))
                K[i * 3 + 1, i * 3 + 1] = 1
                K[i * 3 + 2, :] = np.zeros((1, nr * 3))
                K[i * 3 + 2, i * 3 + 2] = 1
                b[i * 3] = 0
                b[i * 3 + 1] = 0
                b[i * 3 + 2] = 0

    return b, K, bc_count, nm_count
