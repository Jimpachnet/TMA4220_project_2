"""
Implements the solver using the quadpy package for integration
"""

import numpy as np
import quadpy as qp
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import scipy.io
from project_2.infrastructure.p1_reference_element import P1ReferenceElement
from project_2.infrastructure.affine_transformation import AffineTransformation




def solve_quadpy(mesh):
    """
    Solves the system using the quadpy pack
    :param mesh: The mesh on which to work
    :return: The solution
    """
    print("[Info] Solving system using quadpy")
    print("[Info] Generating stiffness matrix")
    K = generate_stiffness_matrix_paper(mesh)



    print(np.linalg.cond(K))
    print(np.shape(K))
    print(np.linalg.matrix_rank(K))

    print("[Info] Generating linear form")
    b = generate_linear_form(mesh)

    # BC Dirichlet
    nr = np.shape(mesh.supports)[0]
    bc_count = 0
    for i in range(nr):
        if mesh.supports[i, 2] == -1 or mesh.supports[i, 2] == 1:
            bc_count+=1
            K[i*3, :] = np.zeros((1, nr*3))
            K[i*3, i*3] = 1
            K[i*3+1, :] = np.zeros((1, nr*3))
            K[i*3+1, i*3+1] = 1
            K[i*3+2, :] = np.zeros((1, nr*3))
            K[i*3+2, i*3+2] = 1
            b[i*3] = 0
            b[i*3+1] = 0
            b[i*3+2] = 0
        if True:
            if mesh.supports[i, 1] == -1 or mesh.supports[i, 1] == 1:
                bc_count+=1
                K[i*3, :] = np.zeros((1, nr*3))
                K[i*3, i*3] = 1
                K[i*3+1, :] = np.zeros((1, nr*3))
                K[i*3+1, i*3+1] = 1
                K[i*3+2, :] = np.zeros((1, nr*3))
                K[i*3+2, i*3+2] = 1
                b[i*3] = 0
                b[i*3+1] = 0
                b[i*3+2] = 0
        if True:
            if mesh.supports[i, 0] == -1 or mesh.supports[i, 0] == 1:
                bc_count+=1
                K[i*3, :] = np.zeros((1, nr*3))
                K[i*3, i*3] = 1
                K[i*3+1, :] = np.zeros((1, nr*3))
                K[i*3+1, i*3+1] = 1
                K[i*3+2, :] = np.zeros((1, nr*3))
                K[i*3+2, i*3+2] = 1
                b[i*3] = 0
                b[i*3+1] = 0
                b[i*3+2] = 0

    print("[Info] Imposed Dirichlet BC on "+str(bc_count)+" points")
    scipy.io.savemat('/home/leon/Desktop/data.dat', dict(K=K))

    absk = np.sin(K)
    plt.imshow(absk, interpolation='nearest', cmap=plt.cm.plasma,
               extent=(0.5, np.shape(absk)[0] + 0.5, 0.5, np.shape(absk)[1] + 0.5))
    plt.colorbar()
    plt.show()

    print(np.linalg.det(K))
    print(np.linalg.cond(K))
    print(np.shape(K))
    print(np.linalg.matrix_rank(K))

    print("[Info] Solving system")
    u = np.linalg.solve(K,b)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]

    return ux, uy, uz


def generate_linear_form(mesh):
    def f1(x=0):
        return 0
    def f2(x=0):
        return 0
    def f3(x=0):
        return -9.81

    atraf = AffineTransformation()
    varnr = mesh.supports.shape[0]
    b = np.zeros((varnr * 3, 1))
    for n in range(mesh.tetraeders.shape[0]):
        v0_coord = (mesh.supports[mesh.tetraeders[n,0],0],mesh.supports[mesh.tetraeders[n,0],1],mesh.supports[mesh.tetraeders[n,0],2])
        v1_coord = (mesh.supports[mesh.tetraeders[n,1],0],mesh.supports[mesh.tetraeders[n,1],1],mesh.supports[mesh.tetraeders[n,1],2])
        v2_coord = (mesh.supports[mesh.tetraeders[n,2],0],mesh.supports[mesh.tetraeders[n,2],1],mesh.supports[mesh.tetraeders[n,2],2])
        v3_coord = (mesh.supports[mesh.tetraeders[n,3],0],mesh.supports[mesh.tetraeders[n,3],1],mesh.supports[mesh.tetraeders[n,3],2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord, v3_coord)
        j = atraf.get_jacobian()
        det = atraf.get_determinant()
        for i in range(4):
            b[mesh.tetraeders[n,i]*3] += f1()*0.5*det
            b[mesh.tetraeders[n,i]*3 + 1] += f1() * 0.5 * det
            b[mesh.tetraeders[n,i]*3 + 2] += f3() * 0.5 * det
    return b


def generate_stiffness_matrix(mesh):
    varnr = mesh.supports.shape[0]
    p1_ref = P1ReferenceElement()
    K = np.zeros((varnr*3, varnr*3))
    atraf = AffineTransformation()
    for n in range(mesh.tetraeders.shape[0]):
        v0_coord = (mesh.supports[mesh.tetraeders[n,0],0],mesh.supports[mesh.tetraeders[n,0],1],mesh.supports[mesh.tetraeders[n,0],2])
        v1_coord = (mesh.supports[mesh.tetraeders[n,1],0],mesh.supports[mesh.tetraeders[n,1],1],mesh.supports[mesh.tetraeders[n,1],2])
        v2_coord = (mesh.supports[mesh.tetraeders[n,2],0],mesh.supports[mesh.tetraeders[n,2],1],mesh.supports[mesh.tetraeders[n,2],2])
        v3_coord = (mesh.supports[mesh.tetraeders[n,3],0],mesh.supports[mesh.tetraeders[n,3],1],mesh.supports[mesh.tetraeders[n,3],2])

        atraf.set_target_cell(v0_coord, v1_coord, v2_coord,v3_coord)
        jinvt = atraf.get_inverse_jacobian().T


        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3

        D_0 = np.array([[1, v, v, 0, 0, 0], [v, 1, v, 0, 0, 0], [v, v, 1, 0, 0, 0],[0,0,0,(1 - v) / 2,0,0],[0,0,0,0,(1 - v) / 2,0],[0,0,0,0,0,(1 - v) / 2]])

        D = Youngs_E_Modulus / (1 - v ** 2) * D_0

        #D_0_2 = np.array([[1, v/(1-v), 0], [v/(1-v), 1, 0], [0, 0, (1 - 2*v) / 2*(1-v)]])

        #D = Youngs_E_Modulus*(1-v)/((1+v)*(1-2*v))

        local_K = np.zeros((12,12))
        for i in range(12):
            for j in range(12):
                co = (0.1, 0.1, 0.1)

                B_i= np.zeros((6, 1))
                B_j = np.zeros((6, 1))

                if i % 3 == 0:
                    B_i[0, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[0]
                    B_i[4, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[1]
                    B_i[5, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[2]
                elif i%3 == 1:
                    B_i[1, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[1]
                    B_i[3, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[0]
                    B_i[5, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[2]
                else:
                    B_i[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[2]
                    B_i[3, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[0]
                    B_i[4, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i // 3])[1]

                if j % 3 == 0:
                    B_j[0, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[0]
                    B_j[4, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[1]
                    B_j[5, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[2]
                elif j % 3 == 1:
                    B_j[1, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[1]
                    B_j[3, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[0]
                    B_j[5, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[2]
                else:
                    B_j[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[2]
                    B_j[3, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[0]
                    B_j[4, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j // 3])[1]


                result = np.asscalar(B_i.T.dot(D).dot(B_j))





                    #ans, err = gauss_legendre_reference(stiffness_matrix_integrant_fast,
                    #                                    args=(p1_ref, i, j, jinvt, result))
                ans = result/2

                local_K[i,j] =atraf.get_determinant() * ans



        if ~np.isclose(np.linalg.det(local_K),0):
            print("ERROR! Determinant of local Matrix should be zero bit is "+str(np.linalg.det(local_K)))

        for i in range(4):
            for j in range(4):
                K[mesh.tetraeders[n,i]*3, mesh.tetraeders[n,j]*3] += local_K[i*3,j*3]
                K[mesh.tetraeders[n,i] * 3 + 1, mesh.tetraeders[n,i] * 3 + 1] += local_K[i * 3+1, j * 3+1]
                K[mesh.tetraeders[n, i] * 3 + 2, mesh.tetraeders[n, i] * 3 + 2] += local_K[i * 3 + 2, j * 3 + 2]



    return K

def generate_stiffness_matrix_paper(mesh):
    varnr = mesh.supports.shape[0]
    p1_ref = P1ReferenceElement()
    K = np.zeros((varnr*3, varnr*3))
    atraf = AffineTransformation()

    for n in range(mesh.tetraeders.shape[0]):
        v0_coord = (mesh.supports[mesh.tetraeders[n,0],0],mesh.supports[mesh.tetraeders[n,0],1],mesh.supports[mesh.tetraeders[n,0],2])
        v1_coord = (mesh.supports[mesh.tetraeders[n,1],0],mesh.supports[mesh.tetraeders[n,1],1],mesh.supports[mesh.tetraeders[n,1],2])
        v2_coord = (mesh.supports[mesh.tetraeders[n,2],0],mesh.supports[mesh.tetraeders[n,2],1],mesh.supports[mesh.tetraeders[n,2],2])
        v3_coord = (mesh.supports[mesh.tetraeders[n,3],0],mesh.supports[mesh.tetraeders[n,3],1],mesh.supports[mesh.tetraeders[n,3],2])

        atraf.set_target_cell(v0_coord, v1_coord, v2_coord,v3_coord)
        jinvt = atraf.get_inverse_jacobian().T


        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3

        D_0 = np.array([[1, v, v, 0, 0, 0], [v, 1, v, 0, 0, 0], [v, v, 1, 0, 0, 0],[0,0,0,(1 - v) / 2,0,0],[0,0,0,0,(1 - v) / 2,0],[0,0,0,0,0,(1 - v) / 2]])

        D = Youngs_E_Modulus / (1 - v ** 2) * D_0


        #D_0_2 = np.array([[1, v/(1-v), 0], [v/(1-v), 1, 0], [0, 0, (1 - 2*v) / 2*(1-v)]])

        #D = Youngs_E_Modulus*(1-v)/((1+v)*(1-2*v))

        local_K = np.zeros((12,12))

        PhiGrad = np.ones((4,4))
        vertices_ = np.squeeze(np.array([v0_coord, v1_coord, v2_coord,v3_coord]))
        PhiGrad[1:4,0:] = vertices_.T
        detgiv = PhiGrad
        PhiGrad = np.linalg.inv(PhiGrad).dot(np.squeeze(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])))
        R = np.zeros((6,12))

        R[[0,3,4],0] = PhiGrad.T[:, 0]
        R[[0, 3, 4], 3] = PhiGrad.T[:, 1]
        R[[0, 3, 4], 6] = PhiGrad.T[:, 2]
        R[[0, 3, 4], 9] = PhiGrad.T[:, 3]

        R[[3,1,5],1] = PhiGrad.T[:, 0]
        R[[3, 1, 5], 4] = PhiGrad.T[:, 1]
        R[[3, 1, 5], 7] = PhiGrad.T[:, 2]
        R[[3, 1, 5], 10] = PhiGrad.T[:, 3]

        R[[4,5,2],2] = PhiGrad.T[:, 0]
        R[[4, 5, 2], 5] = PhiGrad.T[:, 1]
        R[[4, 5, 2], 8] = PhiGrad.T[:, 2]
        R[[4, 5, 2], 11] = PhiGrad.T[:, 3]

        C = D

        stima3 = np.linalg.det(detgiv)/6*R.T.dot(C).dot(R)
        local_K = stima3



        if ~np.isclose(np.linalg.det(local_K),0):
            print("ERROR! Determinant of local Matrix should be zero bit is "+str(np.linalg.det(local_K)))

        for i in range(4):
            for j in range(4):
                K[mesh.tetraeders[n,i]*3, mesh.tetraeders[n,j]*3] += local_K[i*3,j*3]
                K[mesh.tetraeders[n,i] * 3 + 1, mesh.tetraeders[n,j] * 3 + 1] += local_K[i * 3+1, j * 3+1]
                K[mesh.tetraeders[n, i] * 3 + 2, mesh.tetraeders[n, j] * 3 + 2] += local_K[i * 3 + 2, j * 3 + 2]



    return K