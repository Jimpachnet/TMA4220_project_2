#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements a file writer that generates output files
"""

import numpy as np
import datetime

def generate_vtf(path,mesh,stress,ux,uy,uz):
    """
    Generates a vtf file for visualization
    :param path: Where should the file be exported to
    :param mesh: The mesh
    :param stress: The stress in vector form
    :param ux: The deformation in x
    :param uy: The deformation in y
    :param uz: The deformation in z
    """
    print("[Info] Generating vtf file")

    linelist = []

    #File header
    linelist.append('*VTF-1.00')
    linelist.append('')
    linelist.append('*INTERNALSTRING 40001')
    linelist.append('VTF Writer Version info:')
    linelist.append('APP_INFO: GLview Express Writer: 1.1-12')
    linelist.append('GLVIEW_API_VER: 2.1-22')
    linelist.append('EXPORT_DATE: '+str(datetime.datetime.now()))
    linelist.append('')

    #Points
    linelist.append('*NODES 1')
    for point in mesh.supports:
        linelist.append(str('%.6f' % point[0])+' '+str('%.6f' % point[1])+' '+str('%.6f' % point[2]))

    linelist.append('')
    #Elements
    linelist.append('*ELEMENTS 1')
    linelist.append('%NODES #1')
    linelist.append('%NAME "Patch 1"')
    linelist.append('%NO_ID')
    linelist.append('%MAP_NODE_INDICES')
    linelist.append('%PART_ID')
    linelist.append('%TETRAHEDRONS')
    for tet in mesh.tetraeders:
        linelist.append(str(tet[0]+1)+' '+str(tet[1]+1)+' '+str(tet[2]+1)+' '+str(tet[3]+1))
    linelist.append('')

    #Stress
    stressnorm = np.linalg.norm(stress[:, :], axis=1)
    linelist.append('*RESULTS 2')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stressnorm:
        linelist.append(str('%.6f' % s))
    linelist.append('')

    #Displacement
    linelist.append('*RESULTS 3')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 3')
    linelist.append('%PER_NODE #1')
    for i in range(np.shape(uz)[0]):
        linelist.append(str('%.6f' % ux[i])+' '+str('%.6f' % uy[i])+' '+str('%.6f' % uz[i]))

    #Stress xx
    linelist.append('*RESULTS 4')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,0]:
        linelist.append(str('%.6f' % s))
    linelist.append('')
    #Stress yy
    linelist.append('*RESULTS 5')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,1]:
        linelist.append(str('%.6f' % s))
    linelist.append('')
    #Stress zz
    linelist.append('*RESULTS 6')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,2]:
        linelist.append(str('%.6f' % s))
    linelist.append('')
    #Stress xy
    linelist.append('*RESULTS 7')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,3]:
        linelist.append(str('%.6f' % s))
    linelist.append('')
    #Stress xz
    linelist.append('*RESULTS 8')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,4]:
        linelist.append(str('%.6f' % s))
    linelist.append('')
    #Stress yz
    linelist.append('*RESULTS 9')
    linelist.append('%NO_ID')
    linelist.append('%DIMENSION 1')
    linelist.append('%PER_ELEMENT #1')
    for s in stress[:,5]:
        linelist.append(str('%.6f' % s))
    linelist.append('')

    #Footer
    linelist.append('')
    linelist.append('*GLVIEWGEOMETRY 1')
    linelist.append('%STEP 1')
    linelist.append('%ELEMENTS')
    linelist.append('1')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 1')
    linelist.append('%NAME "stress_norm_from_simulation"')
    linelist.append('%STEP 1')
    linelist.append('2')
    linelist.append('')
    linelist.append('*GLVIEWVECTOR 3')
    linelist.append('%NAME "deformation"')
    linelist.append('%STEP 1')
    linelist.append('3')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 5')
    linelist.append('%NAME "stress_xx"')
    linelist.append('%STEP 1')
    linelist.append('4')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 7')
    linelist.append('%NAME "stress_yy"')
    linelist.append('%STEP 1')
    linelist.append('5')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 9')
    linelist.append('%NAME "stress_zz"')
    linelist.append('%STEP 1')
    linelist.append('6')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 11')
    linelist.append('%NAME "stress_xy"')
    linelist.append('%STEP 1')
    linelist.append('7')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 13')
    linelist.append('%NAME "stress_xz"')
    linelist.append('%STEP 1')
    linelist.append('8')
    linelist.append('')
    linelist.append('*GLVIEWSCALAR 15')
    linelist.append('%NAME "stress_yz"')
    linelist.append('%STEP 1')
    linelist.append('9')


    with open(path, 'w') as f:
        for item in linelist:
            f.write("%s\n" % item)
