#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2013-02-20.


import numpy as np

"""
texture.py

"""

__all__ = ['texture_coordinates_by_vertex', ]


def texture_coordinates_by_vertex(self):
    texture_coordinates_by_vertex = [[] for i in range(len(self.v))]
    for i, face in enumerate(self.f):
        for j in [0, 1, 2]:
            texture_coordinates_by_vertex[face[j]].append(self.vt[self.ft[i][j]])
    return texture_coordinates_by_vertex


def reload_texture_image(self):
    import cv2
    # image is loaded as image_height-by-image_width-by-3 array in BGR color order.
    self._texture_image = cv2.imread(self.texture_filepath) if self.texture_filepath else None
    texture_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    if self._texture_image is not None and (self._texture_image.shape[0] != self._texture_image.shape[1] or
       self._texture_image.shape[0] not in texture_sizes or
       self._texture_image.shape[0] not in texture_sizes):
        closest_texture_size_idx = (np.abs(np.array(texture_sizes) - max(self._texture_image.shape))).argmin()
        sz = texture_sizes[closest_texture_size_idx]
        self._texture_image = cv2.resize(self._texture_image, (sz, sz))


def load_texture(self, texture_version):
    '''
    Expect a texture version number as an integer, load the texture version from 'texture_path' (global variable to the
    package).
    Currently there are versions [0,1,2,3] available.
    '''
    import os
    from . import texture_path

    lowres_tex_template = os.path.join(texture_path, 'textured_template_low_v%d.obj' % texture_version)
    highres_tex_template = os.path.join(texture_path, 'textured_template_high_v%d.obj' % texture_version)
    from .mesh import Mesh

    mesh_with_texture = Mesh(filename=lowres_tex_template)
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        mesh_with_texture = Mesh(filename=highres_tex_template)
    self.transfer_texture(mesh_with_texture)


def transfer_texture(self, mesh_with_texture):
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        raise Exception('Mesh topology mismatch')

    self.vt = mesh_with_texture.vt.copy()
    self.ft = mesh_with_texture.ft.copy()

    if not np.all(mesh_with_texture.f == self.f):
        if np.all(mesh_with_texture.f == np.fliplr(self.f)):
            self.ft = np.fliplr(self.ft)
        else:
            # Same shape; let's see if it's face ordering; this could be a bit faster...
            face_mapping = {}
            for f, ii in zip(self.f, range(len(self.f))):
                face_mapping[" ".join([str(x) for x in sorted(f)])] = ii
            self.ft = np.zeros(self.f.shape, dtype=np.uint32)

            for f, ft in zip(mesh_with_texture.f, mesh_with_texture.ft):
                k = " ".join([str(x) for x in sorted(f)])
                if k not in face_mapping:
                    raise Exception('Mesh topology mismatch')
                # the vertex order can be arbitrary...
                ids = []
                for f_id in f:
                    ids.append(np.where(self.f[face_mapping[k]] == f_id)[0][0])
                ids = np.array(ids)
                self.ft[face_mapping[k]] = np.array(ft[ids])

    self.texture_filepath = mesh_with_texture.texture_filepath
    self._texture_image = None


def set_texture_image(self, path_to_texture):
    self.texture_filepath = path_to_texture


def texture_rgb(self, texture_coordinate):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    return np.double(self.texture_image[int(h * (1.0 - texture_coordinate[1]))][int(w * (texture_coordinate[0]))])[::-1]


def texture_rgb_vec(self, texture_coordinates):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    n_ch = self.texture_image.shape[2]
    # XXX texture_coordinates can be lower than 0! clip needed!
    d1 = (h * (1.0 - np.clip(texture_coordinates[:, 1], 0, 1))).astype(np.int)
    d0 = (w * (np.clip(texture_coordinates[:, 0], 0, 1))).astype(np.int)
    flat_texture = self.texture_image.flatten()
    indices = np.hstack([((d1 * (w + 1) * n_ch) + (d0 * n_ch) + (2 - i)).reshape(-1, 1) for i in range(n_ch)])
    return flat_texture[indices]
