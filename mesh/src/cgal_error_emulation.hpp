//!@file
//! Implements a trick to avoid linking with CGAL by defining its own (and
//! unique) error report function.
//!@author Raffi Enficiaud

#ifndef MESH_CGAL_ERROR_EMULATION_HPP__
#define MESH_CGAL_ERROR_EMULATION_HPP__

// exception object
struct mesh_aabb_tree_error {};

#if defined(MESH_CGAL_AVOID_COMPILED_VERSION)

#include <CGAL/assertions.h>
#include <sstream>

// this hack makes it possible to avoid linking with CGAL.
namespace CGAL {
  void assertion_fail(
    const char* expr,
    const char* file,
    int         line,
    const char* msg)
  {
    std::ostringstream o;
    o << "An exception has been caugth during the execution:" << std::endl;
    o << "- file:  " << file << std::endl;
    o << "- line:  " << line << std::endl;
    o << "- error: " << msg << std::endl;

    PyErr_SetString(PyExc_RuntimeError, o.str().c_str());
    throw mesh_aabb_tree_error();
  }
}
#endif

#endif /* MESH_CGAL_ERROR_EMULATION_HPP__ */
