"""
Implements the solver
"""

import numpy as np
import quadpy as qp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, bsr_matrix
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse.linalg
import scipy.io as sio
import time
from project_2.infrastructure.p1_reference_element import P1ReferenceElement
from project_2.infrastructure.affine_transformation_3d import AffineTransformation3D
from project_2.infrastructure.affine_transformation_2d import AffineTransformation2D
from project_2.boundaryconditions import bending_beam_bc


def solve(mesh, config, showstats=False):
    """
    Solves the system
    :param mesh: The mesh on which to work
    :param config: The configuration file
    :param showstats: Should stats be displayed
    :return: The solution
    """
    Youngs_E_Modulus = config.emodulus
    v = config.poissonratio
    lame_lambda = Youngs_E_Modulus * v / ((1 + v) * (1 - 2 * v))
    mue = Youngs_E_Modulus / (2 * (1 + v))
    D = np.array([[lame_lambda + 2 * mue, lame_lambda, lame_lambda, 0, 0, 0],
                  [lame_lambda, lame_lambda + 2 * mue, lame_lambda, 0, 0, 0],
                  [lame_lambda, lame_lambda, lame_lambda + 2 * mue, 0, 0, 0], [0, 0, 0, mue, 0, 0],
                  [0, 0, 0, 0, mue, 0], [0, 0, 0, 0, 0, mue]])

    print("[Info] Generating stiffness matrix")
    time_K_start = time.time()
    K = generate_stiffness_matrix(mesh, D)
    time_K = time.time() - time_K_start
    print("[Info] Generating stiffness matrix took " + str(time_K) + "s")

    if showstats:
        print("[Info] Data before applying BC")
        print("[Info] Condition number:" + str(np.linalg.cond(K)))
        print("[Info] Rank:" + str(np.linalg.matrix_rank(K)) + "/" + str(K.shape[0]))

    print("[Info] Generating linear form")
    time_lf_start = time.time()
    b = generate_linear_form(mesh, density=config.density)
    time_lf = time.time() - time_lf_start
    print("[Info] Generating linear form took " + str(time_lf) + "s")

    time_bc_start = time.time()
    b, K, bc_count, nm_count = bending_beam_bc.apply_bc(mesh, K, b)
    time_bc = time.time() - time_bc_start
    print("[Info] Applying BC took " + str(time_bc) + "s")

    print("[Info] Imposed Dirichlet BC on " + str(bc_count) + " points")
    print("[Info] Imposed Neumann BC on " + str(nm_count) + " points")
    if showstats:
        print("[Info] Data after applying BC")
        print("[Info] Condition number:" + str(np.linalg.cond(K)))
        print("[Info] Rank:" + str(np.linalg.matrix_rank(K)) + "/" + str(K.shape[0]))

    print("[Info] Solving system")
    time_solve_start = time.time()
    u = scipy.sparse.linalg.spsolve(K, b)
    time_solve = time.time() - time_solve_start
    print("[Info] Solving system took " + str(time_solve) + "s")
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]

    print("[Info] Calculating stress")
    time_stress_Start = time.time()
    stress = generate_stress(ux, uy, uz, mesh, D).T
    time_stress = time.time() - time_stress_Start
    print("[Info] Calculating stress took " + str(time_stress) + "s")

    return stress, ux, uy, uz


def generate_stress(ux, uy, uz, mesh, D):
    """
    Calculate the stress
    :param ux: Deformation in x
    :param uy: Deformation in y
    :param uz: Deformation in z
    :param mesh: The mesh
    :param D: The stiffness matrix
    :return: The stress
    """
    varnr = mesh.tetraeders.shape[0]
    strain = np.zeros((varnr, 6))
    u = np.array([ux, uy, uz])

    for n in range(varnr):
        v0_coord = (mesh.supports[mesh.tetraeders[n, 0], 0], mesh.supports[mesh.tetraeders[n, 0], 1],
                    mesh.supports[mesh.tetraeders[n, 0], 2])
        v1_coord = (mesh.supports[mesh.tetraeders[n, 1], 0], mesh.supports[mesh.tetraeders[n, 1], 1],
                    mesh.supports[mesh.tetraeders[n, 1], 2])
        v2_coord = (mesh.supports[mesh.tetraeders[n, 2], 0], mesh.supports[mesh.tetraeders[n, 2], 1],
                    mesh.supports[mesh.tetraeders[n, 2], 2])
        v3_coord = (mesh.supports[mesh.tetraeders[n, 3], 0], mesh.supports[mesh.tetraeders[n, 3], 1],
                    mesh.supports[mesh.tetraeders[n, 3], 2])

        PhiGrad = np.ones((4, 4))
        vertices_ = np.squeeze(np.array([v0_coord, v1_coord, v2_coord, v3_coord]))
        PhiGrad[1:4, 0:] = vertices_.T
        PhiGrad = np.linalg.solve(PhiGrad, np.squeeze(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])))

        valmat = np.zeros((3, 4))

        for i in range(3):
            for j in range(4):
                valmat[i, j] = u[i, mesh.tetraeders[n, j]]

        distortion = valmat.dot(PhiGrad)
        distortion = 0.5 * (distortion + distortion.T)

        eps = np.zeros((6))
        eps[0:3] = np.diagonal(distortion)
        eps[3] = distortion[0, 1]
        eps[4] = distortion[0, 2]
        eps[5] = distortion[1, 2]
        strain[n, :] = eps

    return D.dot(strain.T)


def generate_linear_form(mesh, density):
    """
    Calculates the linear form
    :param mesh: The mesh
    :param density: The density of the material
    :return: The linear form
    """

    def f1(x=0):
        return 0

    def f2(x=0):
        return 0

    def f3(density, x=0):
        return -9.81 * density

    atraf = AffineTransformation3D()
    varnr = mesh.supports.shape[0]
    b = np.zeros((varnr * 3, 1))
    for n in range(mesh.tetraeders.shape[0]):
        v0_coord = (mesh.supports[mesh.tetraeders[n, 0], 0], mesh.supports[mesh.tetraeders[n, 0], 1],
                    mesh.supports[mesh.tetraeders[n, 0], 2])
        v1_coord = (mesh.supports[mesh.tetraeders[n, 1], 0], mesh.supports[mesh.tetraeders[n, 1], 1],
                    mesh.supports[mesh.tetraeders[n, 1], 2])
        v2_coord = (mesh.supports[mesh.tetraeders[n, 2], 0], mesh.supports[mesh.tetraeders[n, 2], 1],
                    mesh.supports[mesh.tetraeders[n, 2], 2])
        v3_coord = (mesh.supports[mesh.tetraeders[n, 3], 0], mesh.supports[mesh.tetraeders[n, 3], 1],
                    mesh.supports[mesh.tetraeders[n, 3], 2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord, v3_coord)
        j = atraf.get_jacobian()
        det = atraf.get_determinant()
        for i in range(4):
            b[mesh.tetraeders[n, i] * 3] += f1() * det / 6
            b[mesh.tetraeders[n, i] * 3 + 1] += f2() * det / 6
            b[mesh.tetraeders[n, i] * 3 + 2] += f3(density) * det / 6
    return b


def generate_stiffness_matrix(mesh, D):
    """
    Generates the global stiffness matrix
    :param mesh: The mesh
    :param D: The stiffness matrix
    :return: The global stiffness matrix
    """
    varnr = mesh.supports.shape[0]
    K = lil_matrix((varnr * 3, varnr * 3))

    for n in range(mesh.tetraeders.shape[0]):
        v0_coord = (mesh.supports[mesh.tetraeders[n, 0], 0], mesh.supports[mesh.tetraeders[n, 0], 1],
                    mesh.supports[mesh.tetraeders[n, 0], 2])
        v1_coord = (mesh.supports[mesh.tetraeders[n, 1], 0], mesh.supports[mesh.tetraeders[n, 1], 1],
                    mesh.supports[mesh.tetraeders[n, 1], 2])
        v2_coord = (mesh.supports[mesh.tetraeders[n, 2], 0], mesh.supports[mesh.tetraeders[n, 2], 1],
                    mesh.supports[mesh.tetraeders[n, 2], 2])
        v3_coord = (mesh.supports[mesh.tetraeders[n, 3], 0], mesh.supports[mesh.tetraeders[n, 3], 1],
                    mesh.supports[mesh.tetraeders[n, 3], 2])

        PhiGrad = np.ones((4, 4))
        vertices_ = np.squeeze(np.array([v0_coord, v1_coord, v2_coord, v3_coord]))
        PhiGrad[1:4, 0:] = vertices_.T
        detgiv = PhiGrad
        PhiGrad = np.linalg.solve(PhiGrad, np.squeeze(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        R = np.zeros((6, 12))

        R[[0, 3, 4], 0] = PhiGrad.T[:, 0]
        R[[0, 3, 4], 3] = PhiGrad.T[:, 1]
        R[[0, 3, 4], 6] = PhiGrad.T[:, 2]
        R[[0, 3, 4], 9] = PhiGrad.T[:, 3]

        R[[3, 1, 5], 1] = PhiGrad.T[:, 0]
        R[[3, 1, 5], 4] = PhiGrad.T[:, 1]
        R[[3, 1, 5], 7] = PhiGrad.T[:, 2]
        R[[3, 1, 5], 10] = PhiGrad.T[:, 3]

        R[[4, 5, 2], 2] = PhiGrad.T[:, 0]
        R[[4, 5, 2], 5] = PhiGrad.T[:, 1]
        R[[4, 5, 2], 8] = PhiGrad.T[:, 2]
        R[[4, 5, 2], 11] = PhiGrad.T[:, 3]

        local_K = np.linalg.det(detgiv) / 6 * R.T.dot(D).dot(R)

        for i in range(4):
            for j in range(4):
                K[mesh.tetraeders[n, i] * 3, mesh.tetraeders[n, j] * 3] += local_K[i * 3, j * 3]
                K[mesh.tetraeders[n, i] * 3 + 1, mesh.tetraeders[n, j] * 3 + 1] += local_K[i * 3 + 1, j * 3 + 1]
                K[mesh.tetraeders[n, i] * 3 + 2, mesh.tetraeders[n, j] * 3 + 2] += local_K[i * 3 + 2, j * 3 + 2]

    return K
