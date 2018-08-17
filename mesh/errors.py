# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.

"""
Error heirarchy for Mesh class
"""


class MeshError(Exception):
    """Base error class for Mesh-related errors"""
    pass


class SerializationError(MeshError):
    """Mesh reading or writing errors"""
    pass
