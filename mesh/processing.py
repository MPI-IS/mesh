#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2013-02-20.


"""
Mesh processing backend
=======================

"""

import numpy as np


def reset_normals(self, face_to_verts_sparse_matrix=None, reset_face_normals=False):
    self.vn = self.estimate_vertex_normals(face_to_verts_sparse_matrix=None)
    if reset_face_normals:
        self.fn = self.f.copy()
    return self


def reset_face_normals(self):
    if not hasattr(self, 'vn'):
        self.reset_normals()
    self.fn = self.f
    return self


def uniquified_mesh(self):
    """This function returns a copy of the mesh in which vertices are copied such that
    each vertex appears in only one face, and hence has only one texture"""
    from mesh import Mesh
    new_mesh = Mesh(v=self.v[self.f.flatten()], f=np.array(range(len(self.f.flatten()))).reshape(-1, 3))

    if not hasattr(self, 'vn'):
        self.reset_normals()
    new_mesh.vn = self.vn[self.f.flatten()]

    if hasattr(self, 'vt'):
        new_mesh.vt = self.vt[self.ft.flatten()]
        new_mesh.ft = new_mesh.f.copy()
    return new_mesh


def keep_vertices(self, keep_list):
    trans = dict((v, i) for i, v in enumerate(keep_list))
    trans_f = np.array([trans[v] if v in trans else -1 for row in self.f for v in row], dtype=np.uint32).reshape(-1, 3)
    if hasattr(self, 'vn') and self.vn.shape[0] == self.vn.shape[0]:
        self.vn = self.vn.reshape(-1, 3)[keep_list]
    if hasattr(self, 'vc') and self.vc.shape[0] == self.v.shape[0]:
        self.vc = self.vc.reshape(-1, 3)[keep_list]
    if hasattr(self, 'landm_raw_xyz'):
        self.recompute_landmark_indices()

    self.v = self.v.reshape(-1, 3)[keep_list]
    self.f = trans_f[(trans_f != np.uint32(-1)).all(axis=1)]
    return self


def point_cloud(self):
    from .mesh import Mesh
    return Mesh(v=self.v, f=[], vc=self.vc) if hasattr(self, 'vc') else Mesh(v=self.v, f=[])


def remove_faces(self, face_indices_to_remove):

    def arr_replace(arr_in, lookup_dict):
        arr_out = arr_in.copy()
        for k, v in lookup_dict.iteritems():
            arr_out[arr_in == k] = v
        return arr_out

    f = np.delete(self.f, face_indices_to_remove, 0)
    v2keep = np.unique(f)
    self.v = self.v[v2keep]
    self.f = arr_replace(f, dict((v, i) for i, v in enumerate(v2keep)))

    if hasattr(self, 'fc'):
        self.fc = np.delete(self.fc, face_indices_to_remove, 0)
    if hasattr(self, 'vn') and self.vn.shape[0] == self.vn.shape[0]:
        self.vn = self.vn.reshape(-1, 3)[v2keep]
    if hasattr(self, 'vc') and self.vc.shape[0] == self.v.shape[0]:
        self.vc = self.vc.reshape(-1, 3)[v2keep]
    if hasattr(self, 'landm_raw_xyz'):
        self.recompute_landmark_indices()

    if hasattr(self, 'ft'):
        ft = np.delete(self.ft, face_indices_to_remove, 0)
        vt2keep = np.unique(ft)
        self.vt = self.vt[vt2keep]
        self.ft = arr_replace(ft, dict((v, i) for i, v in enumerate(vt2keep)))

    return self


def flip_faces(self):
    self.f = self.f.copy()
    for i in range(len(self.f)):
        self.f[i] = self.f[i][::-1]
    if hasattr(self, 'ft'):
        for i in range(len(self.f)):
            self.ft[i] = self.ft[i][::-1]
    return self


def scale_vertices(self, scale_factor):
    self.v *= scale_factor
    return self


def rotate_vertices(self, rotation_matrix):
    import cv2
    rotation_matrix = np.matrix(cv2.Rodrigues(np.array(rotation_matrix))[0] if (np.array(rotation_matrix).shape != (3, 3)) else rotation_matrix)
    self.v = np.array(self.v * rotation_matrix.T)
    return self


def translate_vertices(self, translation):
    self.v += translation
    return self


def subdivide_triangles(self):
    new_faces = []
    new_vertices = self.v.copy()
    for face in self.f:
        face_vertices = np.array([self.v[face[0], :], self.v[face[1], :], self.v[face[2], :]])
        new_vertex = np.mean(face_vertices, axis=0)
        new_vertices = np.vstack([new_vertices, new_vertex])
        new_vertex_index = len(new_vertices) - 1
        if len(new_faces):
            new_faces = np.vstack([new_faces, [face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
        else:
            new_faces = np.array([[face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
    self.v = new_vertices
    self.f = new_faces

    if hasattr(self, 'vt'):
        new_ft = []
        new_texture_coordinates = self.vt.copy()
        for face_texture in self.ft:
            face_texture_coordinates = np.array([self.vt[face_texture[0], :], self.vt[face_texture[1], :], self.vt[face_texture[2], :]])
            new_texture_coordinate = np.mean(face_texture_coordinates, axis=0)
            new_texture_coordinates = np.vstack([new_texture_coordinates, new_texture_coordinate])
            new_texture_index = len(new_texture_coordinates) - 1
            if len(new_ft):
                new_ft = np.vstack([new_ft, [face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
            else:
                new_ft = np.array([[face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
        self.vt = new_texture_coordinates
        self.ft = new_ft
    return self


def concatenate_mesh(self, mesh):
    if len(self.v) == 0:
        self.f = mesh.f.copy()
        self.v = mesh.v.copy()
        self.vc = mesh.vc.copy() if hasattr(mesh, 'vc') else None
    elif len(mesh.v):
        self.f = np.concatenate([self.f, mesh.f.copy() + len(self.v)])
        self.v = np.concatenate([self.v, mesh.v])
        self.vc = np.concatenate([self.vc, mesh.vc]) if (hasattr(mesh, 'vc') and hasattr(self, 'vc')) else None
    return self


# new_ordering specifies the new index of each vertex. If new_ordering[i] = j,
# vertex i should now be the j^th vertex. As such, each entry in new_ordering should be unique.
def reorder_vertices(self, new_ordering, new_normal_ordering=None):
    if new_normal_ordering is None:
        new_normal_ordering = new_ordering
    inverse_ordering = np.zeros(len(new_ordering), dtype=int)
    for i, j in enumerate(new_ordering):
        inverse_ordering[j] = i
    inverse_normal_ordering = np.zeros(len(new_normal_ordering), dtype=int)
    for i, j in enumerate(new_normal_ordering):
        inverse_normal_ordering[j] = i
    self.v = self.v[inverse_ordering]
    if hasattr(self, 'vn'):
        self.vn = self.vn[inverse_normal_ordering]
    for i in range(len(self.f)):
        self.f[i] = np.array([new_ordering[vertex_index] for vertex_index in self.f[i]])
        if hasattr(self, 'fn'):
            self.fn[i] = np.array([new_normal_ordering[normal_index] for normal_index in self.fn[i]])
