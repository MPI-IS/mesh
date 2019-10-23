// Copyright (c) 2018 Max Planck Society for non-commercial scientific research
// This file is part of psbody.mesh project which is released under MPI License.
// See file LICENSE.txt for full license details.

#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <boost/cstdint.hpp>
#include "nearest_triangle.hpp"
using boost::uint32_t;

#include "cgal_error_emulation.hpp"
#include "visibility.h"


static PyObject *
visibility_compute(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *VisibilityError;

static PyMethodDef visibility_methods[] = {
    {"visibility_compute",
        (PyCFunction)visibility_compute,
        METH_VARARGS | METH_KEYWORDS,
        "visibility_compute."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC PyInit_visibility(void)
{
    PyObject *m;

    /// Static module-definition table
    static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "psbody.mesh.visibility", /* name of module */
        "",          /* module documentation, may be NULL */
        -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        visibility_methods                          /* m_methods */
    };

    /// Actually initialize the module object,
    /// using the new Python 3 module-definition table
    m = PyModule_Create(&moduledef);


    import_array();
    VisibilityError = PyErr_NewException(const_cast<char*>("visibility.VisibilityError"), NULL, NULL);
    Py_INCREF(VisibilityError);
    PyModule_AddObject(m, "VisibilityError", VisibilityError);

    /// Initialize and check module
    if (m == NULL)                        { return NULL; }

    /// Return module object
    return m;
}


template <typename CTYPE, int PYTYPE>
npy_intp parse_pyarray(const PyArrayObject *py_arr, const array<CTYPE,3>* &cpp_arr){
    if (py_arr->descr->type_num != PYTYPE || py_arr->nd != 2) {
        PyErr_SetString(PyExc_ValueError,
                "Array must be of a specific type, and 2 dimensional");
        return NULL;
    }
    npy_intp* dims = PyArray_DIMS(py_arr);
    if (dims[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Array must be Nx3");
        return NULL;
    }
    CTYPE *c_arr = (CTYPE*)PyArray_DATA(py_arr);
    cpp_arr = reinterpret_cast<const array<CTYPE,3>*>(c_arr);
    return dims[0];
}

static PyObject *
visibility_compute(PyObject *self, PyObject *args, PyObject *keywds)
{
    try {
        PyArrayObject *py_v=NULL, *py_f=NULL, *py_n=NULL,
                      *py_cams=NULL, *py_sensors=NULL,
                      *py_extra_v=NULL, *py_extra_f=NULL;
        PyObject *py_tree=NULL;
        double min_dist = 1e-3;

        static char* kwlist[] = {"cams","v","f","tree","n","sensors",
                                 "extra_v", "extra_f", "min_dist", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|O!O!OO!O!O!O!d", kwlist,
                                         &PyArray_Type, &py_cams,
                                         &PyArray_Type, &py_v,
                                         &PyArray_Type, &py_f,
                                         &py_tree,
                                         &PyArray_Type, &py_n,
                                         &PyArray_Type, &py_sensors,
                                         &PyArray_Type, &py_extra_v,
                                         &PyArray_Type, &py_extra_f,
                                         &min_dist))
            return NULL;

        bool use_sensors = (py_sensors != NULL);
        bool use_extramesh = (py_extra_v != NULL && py_extra_f != NULL);

        TreeAndTri* search;
        if(py_tree != NULL){
            search = (TreeAndTri *)PyCapsule_GetPointer(py_tree, NULL);
        }
        else{

            const array<double,3>* verts_arr;
            const array<uint32_t,3>* faces_arr;

            npy_intp nv = parse_pyarray<double, NPY_DOUBLE>(py_v, verts_arr);
            npy_intp nf = parse_pyarray<uint32_t, NPY_UINT32>(py_f, faces_arr);

            search = new TreeAndTri;
            search->points.reserve(nv);
            for(size_t pp=0; pp<nv; ++pp){
                search->points.push_back(K::Point_3(verts_arr[pp][0],
                                                    verts_arr[pp][1],
                                                    verts_arr[pp][2]));
            }

            search->triangles.reserve(nf);
            for(size_t tt=0; tt<nf; ++tt) {
                search->triangles.push_back(K::Triangle_3(search->points[faces_arr[tt][0]],
                                                          search->points[faces_arr[tt][1]],
                                                          search->points[faces_arr[tt][2]]));
            }

            if(use_extramesh){
                const array<double,3>* verts_extra_arr;
                const array<uint32_t,3>* faces_extra_arr;
                npy_intp nv_extra = parse_pyarray<double, NPY_DOUBLE>(py_extra_v, verts_extra_arr);
                npy_intp nf_extra = parse_pyarray<uint32_t, NPY_UINT32>(py_extra_f, faces_extra_arr);
                std::vector<K::Point_3> extrapoints;

                extrapoints.reserve(nv_extra);
                for(size_t pp=0; pp<nv_extra; ++pp){
                    extrapoints.push_back(K::Point_3(verts_extra_arr[pp][0],
                                                     verts_extra_arr[pp][1],
                                                     verts_extra_arr[pp][2]));
                }

                search->triangles.reserve(nf+nf_extra);
                for(size_t tt=0; tt<nf_extra; ++tt) {
                    search->triangles.push_back(K::Triangle_3(extrapoints[faces_extra_arr[tt][0]],
                                                              extrapoints[faces_extra_arr[tt][1]],
                                                              extrapoints[faces_extra_arr[tt][2]]));
                }
            }

            search->tree.rebuild(search->triangles.begin(), search->triangles.end());
            search->tree.accelerate_distance_queries();
        }

        if (py_cams->descr->type_num != NPY_DOUBLE || py_cams->nd != 2) {
            PyErr_SetString(PyExc_ValueError, "Camera positions must be of type double, and 2 dimensional");
            return NULL;
        }

        npy_intp* cam_dims = PyArray_DIMS(py_cams);

        if (cam_dims[1] != 3) {
            PyErr_SetString(PyExc_ValueError, "Cams must be Nx3");
            return NULL;
        }

        double *pN = NULL;
        if (py_n != NULL){
            npy_intp* n_dims = PyArray_DIMS(py_n);
            if (n_dims[1] != 3 || n_dims[0] != search->points.size()) {
                PyErr_SetString(PyExc_ValueError, "Normals should have same number of rows as vertices, and 3 columns");
                return NULL;
            }
            pN = (double*)PyArray_DATA(py_n);
        }

        double *pSensors = NULL;
        if (use_sensors){
            npy_intp* n_dims = PyArray_DIMS(py_sensors);
            if (n_dims[1] != 9 || n_dims[0] != cam_dims[0]) {
                PyErr_SetString(PyExc_ValueError, "Sensors should have same number of rows as cameras, 3x3 columns");
                return NULL;
            }
            pSensors = (double*)PyArray_DATA(py_sensors);
        }

        double *pCams = (double*)PyArray_DATA(py_cams);

        size_t C = cam_dims[0];

        npy_intp result_dims[] = {C,search->points.size()};
        PyObject *py_bin_visibility = PyArray_SimpleNew(2, result_dims, NPY_UINT32);
        PyObject *py_normal_dot_cam = PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);
        uint32_t* visibility = reinterpret_cast<uint32_t*>(PyArray_DATA(py_bin_visibility));
        double* normal_dot_cam = reinterpret_cast<double*>(PyArray_DATA(py_normal_dot_cam));

        _internal_compute(search, pN, pCams, C, use_sensors,
                          pSensors, min_dist, visibility, normal_dot_cam);

        // Cleaning and returning
        delete search;
        return Py_BuildValue("NN",py_bin_visibility, py_normal_dot_cam);

    } catch (VisibilityException& e) {
        PyErr_SetString(VisibilityError, e.what());
        return NULL;
    }
}
