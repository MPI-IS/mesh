# Copyright 2016, Max Planck Society.
# Not licensed
# author Raffi Enficiaud

cmake_minimum_required (VERSION 2.8.12)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

project(Mesh)

add_custom_target(cmake_files SOURCES 
                    cmake/python_helper.cmake
                    cmake/thirdparty.cmake
                    src/hijack_python_headers.hpp)

include(cmake/thirdparty.cmake)
include(cmake/python_helper.cmake)


include_directories(${PYTHON_INCLUDE_PATH})

set(MESHPYTHON_SRC
    __init__.py
    mesh.py
    meshviewer.py
    colors.py
    search.py
  )

# for convenience
add_custom_target(MeshPython SOURCES ${MESHPYTHON_SRC})




# aabb_normals
# CGAL_NDEBUG removes calls to logging, warning and error functions that would need
#   a link to the CGAL libraries
# CGAL_HAS_NO_THREADS removes the thread safety of the AABB tree with the advantage
#   of removing the dependency to boost::thread (and indirectly boost::system compiled library)
# MESH_CGAL_AVOID_COMPILED_VERSION is a define of our own in order to hack a remaining
#   log of error without the need to include CGAL generated libraries
# CGAL_NO_AUTOLINK_CGA prevents autolinking on Visual

set(DEFINES_MESH_EXTENSIONS_WITH_CGAL_WITHOUT_LINK
    -DCGAL_NDEBUG=1
    -DMESH_CGAL_AVOID_COMPILED_VERSION=1
    -DCGAL_HAS_NO_THREADS=1
    -DCGAL_NO_AUTOLINK_CGAL=1)

python_add_library(TARGET aabb_normals SOURCES
  src/cgal_error_emulation.hpp
  src/AABB_n_tree.h
  src/aabb_normals.cpp)
set_property(TARGET aabb_normals PROPERTY FOLDER "GeometryExt/")
target_include_directories(aabb_normals PRIVATE
    ${libcgalroot}/include
    ${NUMPY_INCLUDE_PATH}
    ${Boost_INCLUDE_DIRS})
target_compile_definitions(aabb_normals PRIVATE
    ${DEFINES_MESH_EXTENSIONS_WITH_CGAL_WITHOUT_LINK})


# spatial search
python_add_library(TARGET spatialsearch SOURCES
  src/cgal_error_emulation.hpp
  src/nearest_point_triangle_3.h
  src/nearest_triangle_normals.hpp
  src/nearest_triangle.hpp
    
  src/spatialsearchmodule.cpp
)
set_property(TARGET spatialsearch PROPERTY FOLDER "GeometryExt/")
target_include_directories(spatialsearch PRIVATE
    ${libcgalroot}/include
    ${NUMPY_INCLUDE_PATH}
    ${Boost_INCLUDE_DIRS})
target_compile_definitions(spatialsearch PRIVATE
    ${DEFINES_MESH_EXTENSIONS_WITH_CGAL_WITHOUT_LINK})


    
# visibility
python_add_library(TARGET visibility SOURCES 
    src/py_visibility.cpp
    src/visibility.cpp
    )
target_include_directories(visibility PRIVATE 
  ${libcgalroot}/include
  ${NUMPY_INCLUDE_PATH}
  ${Boost_INCLUDE_DIRS})
set_property(TARGET visibility PROPERTY FOLDER "GeometryExt/")
target_compile_definitions(visibility PRIVATE
    ${DEFINES_MESH_EXTENSIONS_WITH_CGAL_WITHOUT_LINK})


# serialization extensions

# plyutils
python_add_library(TARGET plyutils SOURCES
  src/plyutils.h
  src/plyutils.c
  src/rply.h
  src/rply.c)
set_property(TARGET plyutils PROPERTY FOLDER "SerializationExt/")

# loadobj
python_add_library(TARGET loadobj SOURCES src/py_loadobj.cpp)
target_include_directories(loadobj PRIVATE 
  ${NUMPY_INCLUDE_PATH}
  ${Boost_INCLUDE_DIRS})
set_property(TARGET loadobj PROPERTY FOLDER "SerializationExt/")

