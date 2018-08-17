#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-10-30.


import numpy as np

from ..mesh import Mesh
from ..utils import row, col
from .connectivity import vertices_to_edges_matrix


class LinearMeshTransform(object):
    def __init__(self, mtx, faces, vt=None, ft=None):
        self.mtx = mtx
        self.faces = faces
        self.remeshed_vtx_to_remeshed_edge_mtx = vertices_to_edges_matrix(Mesh(f=faces, v=np.zeros((mtx.shape[0], 3))), want_xyz=True)
        self.vtx_to_edge_mtx = self.remeshed_vtx_to_remeshed_edge_mtx.dot(self.mtx)
        if vt is not None:
            self.vt = vt
        if ft is not None:
            self.ft = ft

    def __call__(self, a, want_edges=False):

        if not isinstance(a, Mesh):
            return self.chained_obj_for(a, want_edges)

        a_is_subdivided = (a.v.size == self.mtx.shape[0])

        # if we get here, "a" is a mesh
        if want_edges:
            if a_is_subdivided:
                return self.remeshed_vtx_to_remeshed_edge_mtx.dot(col(a.v)).reshape((-1, 3))
            else:
                return self.vtx_to_edge_mtx.dot(col(a.v)).reshape((-1, 3))
        else:
            if a_is_subdivided:
                return a  # nothing to do!
            else:
                result = Mesh(v=self.mtx.dot(col(a.v)).reshape((-1, 3)), f=self.faces.copy())
                if hasattr(a, 'segm'):
                    result.transfer_segm(a)
                if hasattr(a, 'landm'):
                    result.landm = dict([(k, np.argmin(np.sum((result.v - row(a.v[v])) ** 2, axis=1))) for k, v in a.landm.items()])
                if hasattr(self, 'ft'):
                    result.ft = self.ft
                if hasattr(self, 'vt'):
                    result.vt = self.vt

                return result

    def chained_obj_for(self, a, want_edges):

        from ..geometry.vert_normals import MatVecMult

        if hasattr(a, 'r'):
            a_len = len(a.r)
        else:
            a_len = a.size

        a_is_subdivided = a_len == self.mtx.shape[0]
        if a_is_subdivided and not want_edges:
            return a

        if not want_edges:
            mtx = self.mtx
        elif a_is_subdivided:
            mtx = self.remeshed_vtx_to_remeshed_edge_mtx
        else:
            mtx = self.vtx_to_edge_mtx

        return MatVecMult(mtx, a)
