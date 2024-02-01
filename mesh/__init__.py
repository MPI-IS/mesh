#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.

import os
from os.path import abspath, dirname, expanduser, join

from .mesh import Mesh
try:
    from .meshviewer import MeshViewer, MeshViewers
except NotImplementedError: 
    print("[WARNING] OpenGL not available. MeshViewer will not be imported.")

texture_path = abspath(join(dirname(__file__), '..', 'data', 'template', 'texture_coordinates'))

if 'PSBODY_MESH_CACHE' in os.environ:
    mesh_package_cache_folder = expanduser(os.environ['PSBODY_MESH_CACHE'])
else:
    mesh_package_cache_folder = expanduser('~/.psbody/mesh_package_cache')

if not os.path.exists(mesh_package_cache_folder):
    os.makedirs(mesh_package_cache_folder)
