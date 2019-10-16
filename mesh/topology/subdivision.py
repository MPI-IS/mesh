#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-10-28.

from ..utils import row
from .linear_mesh_transform import LinearMeshTransform
from .connectivity import get_vert_connectivity, get_vertices_per_edge, get_vert_opposites_per_edge

import numpy as np
import scipy.sparse as sp


def loop_subdivider(mesh):

    IS = []
    JS = []
    data = []

    vc = get_vert_connectivity(mesh)
    ve = get_vertices_per_edge(mesh)
    vo = get_vert_opposites_per_edge(mesh)

    if hasattr(mesh, 'ft') and hasattr(mesh, 'vt'):
        from ..mesh import Mesh
        flat_mesh = Mesh(v=mesh.vt, f=mesh.ft)
        vt_start = len(flat_mesh.v)
        vt_edge_to_midpoint = {}
        vt_e = get_vertices_per_edge(flat_mesh)
        vt = flat_mesh.v[:, :2].tolist()
        for idx, vs in enumerate(vt_e):
            vsl = list(vs)
            vsl.sort()
            vt_edge_to_midpoint[(vsl[0], vsl[1])] = vt_start + idx
            vt_edge_to_midpoint[(vsl[1], vsl[0])] = vt_start + idx
            vt.append((np.array(vt[vsl[0]]) + np.array(vt[vsl[1]])) / 2.)
        vt = np.array(vt)

    if True:
        # New values for each vertex
        for idx in range(len(mesh.v)):

            # find neighboring vertices
            nbrs = np.nonzero(vc[:, idx])[0]

            nn = len(nbrs)

            #if nn <=3: # ==3 might give problems when meshes are not water-tight
            if nn == 3:
                wt = 3. / 16.
            elif nn > 3:
                wt = 3. / (8. * nn)
            else:
                raise Exception('nn should be 3 or more')
            for nbr in nbrs:
                IS.append(idx)
                JS.append(nbr)
                data.append(wt)

            JS.append(idx)
            IS.append(idx)
            data.append(1. - (wt * nn))

    start = len(mesh.v)
    edge_to_midpoint = {}

    if True:
        # New values for each edge:
        # new edge verts depend on the verts they span
        for idx, vs in enumerate(ve):

            vsl = list(vs)
            vsl.sort()
            IS.append(start + idx)
            IS.append(start + idx)
            JS.append(vsl[0])
            JS.append(vsl[1])
            data.append(3. / 8)
            data.append(3. / 8)

            opposites = vo[(vsl[0], vsl[1])]
            IS.append(start + idx)
            IS.append(start + idx)
            JS.append(opposites[0])
            JS.append(opposites[1])
            data.append(1. / 8)
            data.append(1. / 8)

            edge_to_midpoint[(vsl[0], vsl[1])] = start + idx
            edge_to_midpoint[(vsl[1], vsl[0])] = start + idx

    f = []
    if hasattr(mesh, 'ft'):
        ft = []

    for f_i, old_f in enumerate(mesh.f):
        ff = np.concatenate((old_f, old_f))
        ftft = np.concatenate((mesh.ft[f_i], mesh.ft[f_i])) if hasattr(mesh, 'ft') else []

        for i in range(3):
            v0 = edge_to_midpoint[(ff[i], ff[i + 1])]
            v1 = ff[i + 1]
            v2 = edge_to_midpoint[(ff[i + 1], ff[i + 2])]
            f.append(row(np.array([v0, v1, v2])))
            if len(ftft):
                if len(np.unique(mesh.ft[f_i])) != len(mesh.ft[f_i]):
                    # anomalous face
                    ft.append(row(np.array([0, 0, 0])))
                else:
                    e_v0 = vt_edge_to_midpoint[(ftft[i], ftft[i + 1])]
                    e_v1 = ftft[i + 1]
                    e_v2 = vt_edge_to_midpoint[(ftft[i + 1], ftft[i + 2])]
                    ft.append(row(np.array([e_v0, e_v1, e_v2])))

        v0 = edge_to_midpoint[(ff[0], ff[1])]
        v1 = edge_to_midpoint[(ff[1], ff[2])]
        v2 = edge_to_midpoint[(ff[2], ff[3])]
        f.append(row(np.array([v0, v1, v2])))
        if len(ftft):
            if len(np.unique(mesh.ft[f_i])) != len(mesh.ft[f_i]):
                # anomalous face
                ft.append(row(np.array([0, 0, 0])))
            else:
                e_v0 = vt_edge_to_midpoint[(ftft[0], ftft[1])]
                e_v1 = vt_edge_to_midpoint[(ftft[1], ftft[2])]
                e_v2 = vt_edge_to_midpoint[(ftft[2], ftft[3])]
                ft.append(row(np.array([e_v0, e_v1, e_v2])))

    f = np.vstack(f)
    if hasattr(mesh, 'ft'):
        ft = np.vstack(ft)

    IS = np.array(IS, dtype=np.uint32)
    JS = np.array(JS, dtype=np.uint32)

    if True:  # for x,y,z coords
        IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
        JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))
        data = np.concatenate((data, data, data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij))

    if hasattr(mesh, 'ft'):
        return LinearMeshTransform(mtx, f, vt=vt, ft=ft)
    else:
        return LinearMeshTransform(mtx, f)
