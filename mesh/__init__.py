#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.

import os
from os.path import abspath, dirname, expanduser, join

from .mesh import Mesh
from .meshviewer import MeshViewer, MeshViewers

texture_path = abspath(join(dirname(__file__), '..', 'data', 'template', 'texture_coordinates'))

if 'PSBODY_MESH_CACHE' in os.environ:
    mesh_package_cache_folder = expanduser(os.environ['PSBODY_MESH_CACHE'])
else:
    mesh_package_cache_folder = expanduser('~/.psbody/mesh_package_cache')

if not os.path.exists(mesh_package_cache_folder):
    os.makedirs(mesh_package_cache_folder)
