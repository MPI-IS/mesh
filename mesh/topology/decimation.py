#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.

import numpy as np
import scipy
import math
import heapq
import scipy.sparse as sp

from .linear_mesh_transform import LinearMeshTransform


def remove_redundant_verts(v, f, eps=1e-10):
    """Given verts and faces, this remove colocated vertices and returns
    a new 'f' and 'v' that discludes them"""

    fshape = f.shape

    dist_mtx = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(v))
    redundant = np.asarray(dist_mtx < eps, np.uint32)

    # update faces to not refer to redundant vertices
    f = f.flatten()
    for i in range(redundant.shape[0]):
        which_verts = np.nonzero(redundant[i, :])[0]
        if len(which_verts) < 2:
            continue
        which_facelocs = np.nonzero(np.in1d(f, which_verts))[0]
        f[which_facelocs] = np.min(which_verts)

    # get rid of unused verts, and update faces accordingly
    vertidxs_left = np.unique(f)
    repl = np.arange(np.max(f) + 1)
    repl[vertidxs_left] = np.arange(len(vertidxs_left))
    v = v[vertidxs_left]
    f = repl[f].reshape((-1, fshape[1]))

    return (v, f)


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def qslim_decimator_fast(mesh, factor=None, n_verts_desired=None):
    from experiments.qslim.simplify_ply_nowrite import load_from_ply_qslim
    if factor is None:
        factor = float(n_verts_desired) / len(mesh.v)
    return load_from_ply_qslim(mesh, n_tris=len(mesh.f) * factor, want_optimal=True)


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    from .connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])), shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print('found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost']))
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx

def qslim_decimator(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: An Fx3 array of faces.
    """
    new_faces, mtx = qslim_decimator_transformer(mesh, factor, n_verts_desired)
    return LinearMeshTransform(mtx, new_faces)

def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    #
    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    # for x,y,z coords
    IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
    JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))
    data = np.concatenate((data, data, data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left) * 3, num_original_verts * 3))

    return (new_faces, mtx)
