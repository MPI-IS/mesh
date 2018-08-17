#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.

import tempfile
from os.path import abspath, dirname, join

test_data_folder = abspath(join(dirname(__file__), '..', 'data', 'unittest'))

# folder used for creating temporary files
temporary_files_folder = tempfile.gettempdir()
