# Copyright 2016, Max Planck Society.
# Not licensed
# author Raffi Enficiaud

# helper file containing commands easing the declaration of python modules

include(CMakeParseArguments)


#.rst:
# .. command:: python_add_library
#
#   Adds a shared library that is meant to be a python extension module.
#   The added library links to python library and has the proper extension.
#
#   ::
#
#     python_add_library(
#         TARGET targetname
#         SOURCES source_list)
#
#   ``targetname`` name of the python extension
#   ``source_list`` list of source files for this target.
#
function(python_add_library)

  if("${PYTHON_MODULES_EXTENSIONS}" STREQUAL "")
    if("${PYTHON_VERSION}" VERSION_GREATER "2.5")
      if(UNIX OR MINGW)
        set(PYTHON_MODULES_EXTENSIONS_TEMP ".so")
      else()
        set(PYTHON_MODULES_EXTENSIONS_TEMP ".pyd")
      endif()
    else()
      if(APPLE)
        set(PYTHON_MODULES_EXTENSIONS_TEMP ".so")
      else()
        set(PYTHON_MODULES_EXTENSIONS_TEMP ${CMAKE_SHARED_LIBRARY_SUFFIX})
      endif()
    endif()
    set(PYTHON_MODULES_EXTENSIONS ${PYTHON_MODULES_EXTENSIONS_TEMP} CACHE STRING "Python modules extension for the current platform")
  endif()

  
  set(options )
  set(oneValueArgs TARGET)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(local_python_add_cmd "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})  

  if("${local_python_add_cmd_TARGET}" STREQUAL "")
    message(FATAL_ERROR "python_add_library: the TARGET should be specified")
  endif()

  if("${local_python_add_cmd_SOURCES}" STREQUAL "")
    message(FATAL_ERROR "python_add_library: at least one source file should be specified")
  endif()
  
  add_library(${local_python_add_cmd_TARGET} SHARED 
      src/hijack_python_headers.hpp  # by default
      "${local_python_add_cmd_SOURCES}")
  target_include_directories(${local_python_add_cmd_TARGET} PRIVATE ${PYTHON_INCLUDE_PATH})
  target_link_libraries(${local_python_add_cmd_TARGET} ${PYTHON_LIBRARY}) # PYTHON_LIBRARIES may contain the debug version of python, which we do not want
  set_target_properties(${local_python_add_cmd_TARGET} 
    PROPERTIES SUFFIX ${PYTHON_MODULES_EXTENSIONS}
    PREFIX "")
  
  if(FALSE)
  set_target_properties(${local_python_add_cmd_TARGET}
      PROPERTIES 
        OUTPUT_NAME_DEBUG ${local_python_add_cmd_TARGET}_d
        )
  endif()

endfunction(python_add_library)