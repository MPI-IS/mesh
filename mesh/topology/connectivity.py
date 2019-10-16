#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-10-24.

import os
import zlib
import numpy as np
import pickle
import scipy.sparse as sp

from ..utils import row, col
from .. import mesh_package_cache_folder


def get_vert_opposites_per_edge(mesh):
    """Returns a dictionary from vertidx-pairs to opposites.
    For example, a key consist of [4,5)] meaning the edge between
    vertices 4 and 5, and a value might be [10,11] which are the indices
    of the vertices opposing this edge."""
    result = {}
    for f in mesh.f:
        for i in range(3):
            key = [f[i], f[(i + 1) % 3]]
            key.sort()
            key = tuple(key)
            val = f[(i + 2) % 3]

            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]
    return result


def get_vert_connectivity(mesh):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh.v), len(mesh.v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh.f[:, i]
        JS = mesh.f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def vertices_to_edges_matrix(mesh, want_xyz=True):
    """Returns a matrix M, which if multiplied by vertices,
    gives back edges (so "e = M.dot(v)"). Note that this generates
    one edge per edge, *not* two edges per triangle.

    :param mesh: the mesh to process
    :param want_xyz: if true, takes and returns xyz coordinates, otherwise
                      takes and returns x *or* y *or* z coordinates
    """

    vpe = get_vertices_per_edge(mesh)
    IS = np.repeat(np.arange(len(vpe)), 2)
    JS = vpe.flatten()
    data = np.ones_like(vpe)
    data[:, 1] = -1
    data = data.flatten()

    if want_xyz:
        IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
        JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))
        data = np.concatenate((data, data, data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    return sp.csc_matrix((data, ij))


def vertices_in_common(face_1, face_2):
    """Returns the two vertices shared by two faces,
    optimized for the case of triangular faces with two vertices in common"""
    if len(face_1) == 3 and len(face_2) == 3:
        vertices_in_common = [None, None]
        i = 0
        if (face_1[0] == face_2[0]) or (face_1[0] == face_2[1]) or (face_1[0] == face_2[2]):
            vertices_in_common[i] = face_1[0]
            i += 1
        if (face_1[1] == face_2[0]) or (face_1[1] == face_2[1]) or (face_1[1] == face_2[2]):
            vertices_in_common[i] = face_1[1]
            i += 1
        if (face_1[2] == face_2[0]) or (face_1[2] == face_2[1]) or (face_1[2] == face_2[2]):
            vertices_in_common[i] = face_1[2]
            i += 1
        if i == 2:
            if vertices_in_common[0] > vertices_in_common[1]:
                vertices_in_common = [vertices_in_common[1], vertices_in_common[0]]
            return vertices_in_common
        elif i < 2:
            return [vertices_in_common[0]] if i else []
    else:
        return np.intersect1d(face_1, face_2)


def get_vertices_per_edge(mesh, faces_per_edge=None):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    faces = mesh.f
    suffix = str(zlib.crc32(faces_per_edge.flatten())) if faces_per_edge is not None else ''
    cache_fname = os.path.join(mesh_package_cache_folder, 'verts_per_edge_cache_' + str(zlib.crc32(faces.flatten())) + '_' + suffix + '.pkl')
    try:
        with open(cache_fname, 'rb') as fp:
            return(pickle.load(fp))
    except:
        if faces_per_edge is not None:
            result = np.asarray(np.vstack([row(np.intersect1d(mesh.f[k[0]], mesh.f[k[1]])) for k in faces_per_edge]), np.uint32)
        else:
            vc = sp.coo_matrix(get_vert_connectivity(mesh))
            result = np.hstack((col(vc.row), col(vc.col)))
            result = result[result[:, 0] < result[:, 1]]  # for uniqueness

        with open(cache_fname, 'wb') as fp:
            pickle.dump(result, fp, -1)
        return result

        # s1 = [set([v[0], v[1]]) for v in mesh.v]
        # s2 = [set([v[1], v[2]]) for v in mesh.v]
        # s3 = [set([v[2], v[0]]) for v in mesh.v]
        #
        # return s1+s2+s3


def get_faces_per_edge(mesh):

    faces = mesh.f
    cache_fname = os.path.join(mesh_package_cache_folder, 'edgecache_new_' + str(zlib.crc32(faces.flatten())) + '.pkl')

    try:
        with open(cache_fname, 'rb') as fp:
            return(pickle.load(fp))
    except:
        f = faces
        IS = np.repeat(np.arange(len(f)), 3)
        JS = f.ravel()
        data = np.ones(IS.size)
        f2v = sp.csc_matrix((data, (IS, JS)), shape=(len(f), np.max(f.ravel()) + 1))
        f2f = f2v.dot(f2v.T)
        f2f = f2f.tocoo()
        f2f = np.hstack((col(f2f.row), col(f2f.col), col(f2f.data)))
        which = (f2f[:, 0] < f2f[:, 1]) & (f2f[:, 2] >= 2)
        result = np.asarray(f2f[which, :2], np.uint32)

        with open(cache_fname, 'wb') as fp:
            pickle.dump(result, fp, -1)
        return result


def get_faces_per_edge_old(mesh):
    """Returns an Ex2 array of adjacencies between faces, where
    each element in the array is a face index. Each edge is included
    only once.

    Assumes that the mesh's faces are either all CW or all CCW (but not a mix).
    """

    faces = mesh.f
    cache_fname = os.path.join(mesh_package_cache_folder, 'edgecache_old_' + str(zlib.crc32(faces.flatten())) + '.pkl')
    try:
        with open(cache_fname, 'rb') as fp:
            return(pickle.load(fp))
    except:
        # Raffi: not used
        # num_verts = len(mesh.v)
        # e1 = sp.csc_matrix((num_verts, num_verts))
        # e2 = sp.csc_matrix((num_verts, num_verts))

        IS = np.hstack((faces[:, 0], faces[:, 1], faces[:, 2])).T
        JS = np.hstack((faces[:, 1], faces[:, 2], faces[:, 0])).T
        VS = np.hstack((np.tile(col(np.arange(len(faces))), (3, 1)))).T
        VS = VS + 1  # add "1" so that face "0" won't be ignored in sparse arrays

        adj_mtx_csc = sp.csc_matrix((VS, np.vstack((row(IS), row(JS)))))
        adj_mtx_coo = sp.coo_matrix((VS, np.vstack((row(IS), row(JS)))))

        edges = []
        for i in xrange(len(adj_mtx_coo.row)):
            r = adj_mtx_coo.row[i]
            c = adj_mtx_coo.col[i]
            if r < c:
                edges.append(row(np.array([adj_mtx_csc[c, r], adj_mtx_csc[r, c]])))

        edges = np.concatenate(edges, axis=0)
        edges = edges - 1  # get rid of "1" we added on earlier
        with open(cache_fname, 'wb') as fp:
            pickle.dump(edges, fp, -1)

        return edges
