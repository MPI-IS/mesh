// Copyright (c) 2018 Max Planck Society for non-commercial scientific research
// This file is part of psbody.mesh project which is released under MPI License.
// See file LICENSE.txt for full license details.

#ifndef PLYUTILS_H__
#define PLYUTILS_H__

// needed to avoid the link to debug "_d.lib" libraries
#include "hijack_python_headers.hpp"
#include "rply.h"

static PyObject * plyutils_read(PyObject *self, PyObject *args);
static PyObject * plyutils_write(PyObject *self, PyObject *args);
void error_cb(const char *message);
int vertex_cb(p_ply_argument argument);
int face_cb(p_ply_argument argument);

#endif /* PLYUTILS_H__ */
