#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2012 Max Planck Society. All rights reserved.

import numpy as np
from . import colors


class Lines(object):
    """Collection of 3D lines

    Attributes:
        v: Vx3 array of vertices
        e: Ex2 array of edges
    """

    def __init__(self, v, e, vc=None, ec=None):

        self.v = np.array(v)
        self.e = np.array(e)

        if vc is not None:
            self.set_vertex_colors(vc)

        if ec is not None:
            self.set_edge_colors(ec)

    def colors_like(self, color, arr):
        from .utils import row, col
        if isinstance(color, str):
            color = colors.name_to_rgb[color]
        elif isinstance(color, list):
            color = np.array(color)

        if color.shape == (arr.shape[0],):
            def jet(v):
                fourValue = 4 * v
                red = min(fourValue - 1.5, -fourValue + 4.5)
                green = min(fourValue - 0.5, -fourValue + 3.5)
                blue = min(fourValue + 0.5, -fourValue + 2.5)
                result = np.array([red, green, blue])
                result[result > 1.0] = 1.0
                result[result < 0.0] = 0.0
                return row(result)
            color = col(color)
            color = np.concatenate([jet(color[i]) for i in xrange(color.size)], axis=0)

        return np.ones((arr.shape[0], 3)) * color

    def set_vertex_colors(self, vc):
        self.vc = self.colors_like(vc, self.v)

    def set_edge_colors(self, ec):
        self.ec = self.colors_like(ec, self.e)

    def write_obj(self, filename):
        with open(filename, 'w') as fi:
            for r in self.v:
                fi.write('v %f %f %f\n' % (r[0], r[1], r[2]))
            for e in self.e:
                fi.write('l %d %d\n' % (e[0] + 1, e[1] + 1))
