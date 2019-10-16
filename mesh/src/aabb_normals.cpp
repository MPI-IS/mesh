
// needed to avoid the link to debug "_d.lib" libraries
#include "hijack_python_headers.hpp"
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "AABB_n_tree.h"
#include "cgal_error_emulation.hpp"

typedef uint32_t Index;

static PyObject * aabbtree_normals_compute(PyObject *self, PyObject *args);
static PyObject * aabbtree_normals_nearest(PyObject *self, PyObject *args);
static PyObject * aabbtree_normals_selfintersects(PyObject *self, PyObject *args);

static PyMethodDef SpatialsearchMethods[] = {
    { "aabbtree_n_compute",
      aabbtree_normals_compute,
      METH_VARARGS,
      "aabbtree_n_compute."},
    { "aabbtree_n_nearest",
      aabbtree_normals_nearest,
      METH_VARARGS,
      "aabbtree_n_nearest."},
    { "aabbtree_n_selfintersects",
      aabbtree_normals_selfintersects,
      METH_VARARGS,
      "aabbtree_n_selfintersects."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduleDef =
{
    PyModuleDef_HEAD_INIT,
    "aabb_normals", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    SpatialsearchMethods
};

PyMODINIT_FUNC PyInit_aabb_normals(void)
{
    PyObject *module = PyModule_Create(&moduleDef);

    import_array();

    return module;
}

void aabb_tree_destructor(PyObject *ptr)
{
    TreeAndTri* search = (TreeAndTri*) PyCapsule_GetPointer(ptr, NULL);
    delete search;
}


static PyObject *
aabbtree_normals_compute(PyObject *self, PyObject *args)
{
    PyArrayObject *py_v = NULL, *py_f = NULL;

    double eps;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &py_v, &PyArray_Type, &py_f, &eps))
        return NULL;

    if (py_v->descr->type_num != NPY_DOUBLE || py_v->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "Vertices must be of type double, and 2 dimensional");
        return NULL;
    }
    if (py_f->descr->type_num != NPY_UINT32 || py_f->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "Faces must be of type uint32, and 2 dimensional");
        return NULL;
    }

    npy_intp* v_dims = PyArray_DIMS(py_v);
    npy_intp* f_dims = PyArray_DIMS(py_f);

    if (v_dims[1] != 3 || f_dims[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be Nx3");
        return NULL;
    }

    double *pV = (double*)PyArray_DATA(py_v);
    uint32_t *pF = (uint32_t*)PyArray_DATA(py_f);

    size_t P = v_dims[0];
    size_t T = f_dims[0];

    array<uint32_t, 3>* m_mesh_tri=reinterpret_cast<array<uint32_t,3>*>(pF);
    array<double, 3>* m_mesh_points=reinterpret_cast<array<double,3>*>(pV);

    try
    {
      TreeAndTri* search = new TreeAndTri(m_mesh_tri,m_mesh_points,eps,T,P);

      PyObject* result = PyCapsule_New((void*)search, NULL, aabb_tree_destructor);
      return result;
    }
    catch (mesh_aabb_tree_error&)
    {
      return Py_None;
    }

}

static PyObject *
aabbtree_normals_nearest(PyObject *self, PyObject *args)
{
    PyObject *py_tree, *py_v, *py_n;
    if (!PyArg_ParseTuple(args, "OOO", &py_tree, &py_v, &py_n))
        return NULL;

    TreeAndTri *search = (TreeAndTri *) PyCapsule_GetPointer(py_tree, NULL);

    npy_intp* v_dims = PyArray_DIMS(py_v);
    npy_intp* n_dims = PyArray_DIMS(py_n);

    if (v_dims[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be Nx3");
        return NULL;
    }
    if (n_dims[1] != v_dims[1] || n_dims[0] != v_dims[0]){
        PyErr_SetString(PyExc_ValueError, "Normals should have same dimensions as points");
        return NULL;
    }

    size_t S=v_dims[0];

    array<double, 3>* m_sample_points=reinterpret_cast<array<double,3>*>(PyArray_DATA(py_v));
    array<double, 3>* m_sample_n=reinterpret_cast<array<double,3>*>(PyArray_DATA(py_n));

    #ifdef _OPENMP
    omp_set_num_threads(8);
    #endif

    std::vector<Point_Normal> sample_points;
    sample_points.reserve(S);
    for(size_t ss=0; ss<S; ++ss) {
        sample_points.push_back(std::make_pair(K::Point_3(m_sample_points[ss][0],
                                                          m_sample_points[ss][1],
                                                          m_sample_points[ss][2]),
                                               Normal(m_sample_n[ss][0],
                                                      m_sample_n[ss][1],
                                                      m_sample_n[ss][2])));
    }

    npy_intp result1_dims[] = {1, S};

    PyObject *result1 = PyArray_SimpleNew(2, result1_dims, NPY_UINT32);

    uint32_t* closest_triangles=reinterpret_cast<uint32_t*>(PyArray_DATA(result1));
    array<double,3>* closest_point=NULL;
    //if(1) { //nlhs > 1) {
        npy_intp result2_dims[] = {S, 3};
        PyObject *result2 = PyArray_SimpleNew(2, result2_dims, NPY_DOUBLE);
        closest_point=reinterpret_cast<array<double,3>*>(PyArray_DATA(result2));
    //}

    #pragma omp parallel for
    for(size_t ss=0; ss<S; ++ss) {
        Tree::Point_and_primitive_id closest=search->tree.closest_point_and_primitive(sample_points[ss]);
        closest_triangles[ss]=std::distance(search->triangles.begin(), closest.second);
        //if(nlhs > 1) {
            for(size_t cc=0; cc<3; ++cc) {
                closest_point[ss][cc]=closest.first[cc];
            }
        //}
    }
    return Py_BuildValue("NN", result1, result2);
}

static PyObject *
aabbtree_normals_selfintersects(PyObject *self, PyObject *args)
{
    int n_intersections = 0;
    PyObject *py_tree;
    if (!PyArg_ParseTuple(args, "O", &py_tree))
        return NULL;

    TreeAndTri *search = (TreeAndTri *) PyCapsule_GetPointer(py_tree, NULL);

    for(Iterator it=search->triangles.begin();it!=search->triangles.end();++it)
        if(search->tree.do_intersect(*it))
            ++n_intersections;

    return Py_BuildValue("i", n_intersections);
}
