#ifndef MEAH_INCLUDE_PYTHON_HEADER_HPP__
#define MEAH_INCLUDE_PYTHON_HEADER_HPP__

/*!@file
 * This file hijacks the inclusion of the python libraries on Windows to 
 * prevent the linking with the debug version of python.lib (that is named
 * python_d.lib and that is not provided by default).
 */

#undef MESH_HIJACK_AUTO_LINK

#if defined(_WIN32) && defined(_DEBUG)
  #define MESH_HIJACK_AUTO_LINK
  #undef _DEBUG
#endif

#include <Python.h>

#if defined(MESH_HIJACK_AUTO_LINK)
  #define _DEBUG
  #undef MESH_HIJACK_AUTO_LINK
#endif


#endif /* MEAH_INCLUDE_PYTHON_HEADER_HPP__ */