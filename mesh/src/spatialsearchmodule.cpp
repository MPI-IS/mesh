// Copyright (c) 2018 Max Planck Society for non-commercial scientific research
// This file is part of psbody.mesh project which is released under MPI License.
// See file LICENSE.txt for full license details.

// needed to avoid the link to debug "_d.lib" libraries
#include "hijack_python_headers.hpp"
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_TBB
#include "tbb/tbb.h"
using namespace tbb;
#endif

#include "cgal_error_emulation.hpp"
#include "nearest_triangle.hpp"
#include "nearest_point_triangle_3.h"

typedef uint32_t Index;

static PyObject *
spatialsearch_aabbtree_compute(PyObject *self, PyObject *args);

static PyObject *
spatialsearch_aabbtree_nearest(PyObject *self, PyObject *args);

static PyObject *
spatialsearch_aabbtree_nearest_alongnormal(PyObject *self, PyObject *args);

static PyObject *Mesh_IntersectionsError;

static PyMethodDef SpatialsearchMethods[] = {
    {"aabbtree_compute",  spatialsearch_aabbtree_compute, METH_VARARGS,
        "aabbtree_compute."},
    {"aabbtree_nearest",  spatialsearch_aabbtree_nearest, METH_VARARGS,
        "aabbtree_nearest."},
    {"aabbtree_nearest_alongnormal",  spatialsearch_aabbtree_nearest_alongnormal, METH_VARARGS,
        "aabbtree_nearest."},
    {NULL, NULL, 0}        /* Sentinel */
};



PyMODINIT_FUNC
initspatialsearch(void)
{
    PyObject *m = Py_InitModule("spatialsearch", SpatialsearchMethods);
    if (m == NULL) {
        return;
    }

    import_array();

    // Add Exceptions Object
    Mesh_IntersectionsError = PyErr_NewException(const_cast<char*>("spatialsearch.Mesh_IntersectionsError"), NULL, NULL);
    Py_INCREF(Mesh_IntersectionsError);
    PyModule_AddObject(m, "Mesh_IntersectionsError", Mesh_IntersectionsError);
}


void aabb_tree_destructor(void *ptr)
{
    TreeAndTri* search = (TreeAndTri*)ptr;
    delete search;
}

static PyObject *
spatialsearch_aabbtree_compute(PyObject *self, PyObject *args)
{
    PyArrayObject *py_v = NULL, *py_f = NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &py_v,&PyArray_Type, &py_f))
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

    TreeAndTri* search = new TreeAndTri;
    search->points.reserve(P);

    for(size_t pp=0; pp<P; ++pp) {
        search->points.push_back(K::Point_3(m_mesh_points[pp][0], m_mesh_points[pp][1], m_mesh_points[pp][2]));
    }

    search->triangles.reserve(T);

    for(size_t tt=0; tt<T; ++tt) {
        search->triangles.push_back(K::Triangle_3(search->points[m_mesh_tri[tt][0]],
                                                  search->points[m_mesh_tri[tt][1]],
                                                  search->points[m_mesh_tri[tt][2]]));
    }
    search->tree.rebuild(search->triangles.begin(), search->triangles.end());
    search->tree.accelerate_distance_queries();

    PyObject* result = PyCObject_FromVoidPtr((void*)search, aabb_tree_destructor);
    return Py_BuildValue("N", result);
}

void spatialsearch_aabbtree_nearest_one(int ss, TreeAndTri * search, std::vector<K::Point_3> &sample_points,
        uint32_t* closest_triangles, uint32_t* closest_part, array<double,3>* closest_point)
{
    Tree::Point_and_primitive_id closest=search->tree.closest_point_and_primitive(sample_points[ss]);
    closest_triangles[ss]=std::distance(search->triangles.begin(), closest.second);
    for(size_t cc=0; cc<3; ++cc) {
        closest_point[ss][cc]=closest.first[cc];
    }
    K k;
    K::Point_3 result;
    closest_part[ss]=CGAL::iev::nearest_primitive(sample_points[ss], *closest.second, result, k);
}

#ifdef HAVE_TBB

class AaBbTreeNearestTbb {
    TreeAndTri *search;
    std::vector<K::Point_3> *sample_points;
    uint32_t* closest_triangles;
    uint32_t* closest_part;
    array<double,3>* closest_point;
public:
    void operator()( const blocked_range<size_t>& r ) const {
        for( size_t i=r.begin(); i!=r.end(); ++i )
            spatialsearch_aabbtree_nearest_one(i, search, *sample_points, closest_triangles, closest_part, closest_point);
    }
    AaBbTreeNearestTbb( TreeAndTri * search, std::vector<K::Point_3> *sample_points,
            uint32_t* closest_triangles, uint32_t* closest_part, array<double,3>* closest_point) :
        search(search), sample_points(sample_points),
        closest_triangles(closest_triangles),
        closest_part(closest_part),
        closest_point(closest_point) {}
};

#endif

static PyObject *
spatialsearch_aabbtree_nearest(PyObject *self, PyObject *args)
{
    PyObject *py_tree, *py_v;
    if (!PyArg_ParseTuple(args, "OO!", &py_tree, &PyArray_Type, &py_v))
        return NULL;
    TreeAndTri *search = (TreeAndTri *)PyCObject_AsVoidPtr(py_tree);

    npy_intp* v_dims = PyArray_DIMS(py_v);

    if (v_dims[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Input must be Nx3");
        return NULL;
    }

    size_t S=v_dims[0];

    array<double, 3>* m_sample_points=reinterpret_cast<array<double,3>*>(PyArray_DATA(py_v));

    #ifdef _OPENMP
    omp_set_num_threads(8);
    #endif

    std::vector<K::Point_3> sample_points;
    sample_points.reserve(S);
    for(size_t ss=0; ss<S; ++ss) {
        sample_points.push_back(K::Point_3(m_sample_points[ss][0], m_sample_points[ss][1], m_sample_points[ss][2]));
    }

    npy_intp result1_dims[] = {1, S};

    PyObject *result1 = PyArray_SimpleNew(2, result1_dims, NPY_UINT32);
    PyObject *result2 = PyArray_SimpleNew(2, result1_dims, NPY_UINT32);

    uint32_t* closest_triangles=reinterpret_cast<uint32_t*>(PyArray_DATA(result1));
    uint32_t* closest_part=reinterpret_cast<uint32_t*>(PyArray_DATA(result2));

    npy_intp result3_dims[] = {S, 3};
    PyObject *result3 = PyArray_SimpleNew(2, result3_dims, NPY_DOUBLE);
    array<double,3>* closest_point = reinterpret_cast<array<double,3>*>(PyArray_DATA(result3));


#ifdef HAVE_TBB
    parallel_for(blocked_range<size_t>(0,S), AaBbTreeNearestTbb(search, &sample_points, closest_triangles, closest_part, closest_point));
#else
#ifdef HAVE_OPENMP
    #pragma omp parallel for
#endif
    for(size_t ss=0; ss<S; ++ss) {
        spatialsearch_aabbtree_nearest_one(ss, search, sample_points, closest_triangles, closest_part, closest_point);
    }
#endif
    return Py_BuildValue("NNN", result1, result2, result3);
}

static PyObject *
spatialsearch_aabbtree_nearest_alongnormal(PyObject *self, PyObject *args)
{
    PyObject *py_tree, *py_p, *py_n;
    if (!PyArg_ParseTuple(args, "OO!O!", &py_tree, &PyArray_Type, &py_p, &PyArray_Type, &py_n))
        return NULL;
    TreeAndTri *search = (TreeAndTri *)PyCObject_AsVoidPtr(py_tree);

    npy_intp* p_dims = PyArray_DIMS(py_p);
    npy_intp* n_dims = PyArray_DIMS(py_p);

    if (p_dims[1] != 3 || n_dims[1] != 3 || p_dims[0] != n_dims[0]) {
        PyErr_SetString(PyExc_ValueError, "Points and normals must be Nx3");
        return NULL;
    }

    size_t S=p_dims[0];

    array<double, 3>* p_arr=reinterpret_cast<array<double,3>*>(PyArray_DATA(py_p));
    array<double, 3>* n_arr=reinterpret_cast<array<double,3>*>(PyArray_DATA(py_n));

    #ifdef _OPENMP
    omp_set_num_threads(8);
    #endif

    std::vector<K::Point_3> p_v;
    std::vector<K::Vector_3> n_v;
    p_v.reserve(S);
    n_v.reserve(S);
    for(size_t ss=0; ss<S; ++ss) {
        p_v.push_back(K::Point_3(p_arr[ss][0], p_arr[ss][1], p_arr[ss][2]));
        n_v.push_back(K::Vector_3(n_arr[ss][0], n_arr[ss][1], n_arr[ss][2]));
    }

    npy_intp result1_dims[] = {S};

    PyObject *result1 = PyArray_SimpleNew(1, result1_dims, NPY_DOUBLE);

    double* distance=reinterpret_cast<double*>(PyArray_DATA(result1));

    PyObject *result2 = PyArray_SimpleNew(1, result1_dims, NPY_UINT32);
    uint32_t* closest_triangles=reinterpret_cast<uint32_t*>(PyArray_DATA(result2));

    npy_intp result3_dims[] = {S, 3};
    PyObject *result3 = PyArray_SimpleNew(2, result3_dims, NPY_DOUBLE);
    array<double,3>* closest_point = reinterpret_cast<array<double,3>*>(PyArray_DATA(result3));

#ifdef HAVE_OPENMP
    #pragma omp parallel for
#endif
    for(size_t ss=0; ss<S; ++ss) {
        std::list<Tree::Object_and_primitive_id> intersections;
        search->tree.all_intersections(K::Ray_3(p_v[ss],n_v[ss]), std::back_inserter(intersections));
        search->tree.all_intersections(K::Ray_3(p_v[ss],-n_v[ss]), std::back_inserter(intersections));

        std::list<Tree::Object_and_primitive_id>::iterator it_int;
        std::vector<double> ss_dists;
        std::vector<K::Point_3> ss_points;
        std::vector<uint32_t> ss_tris;
        for(it_int=intersections.begin(); it_int!=intersections.end(); ++it_int){
            CGAL::Object object = it_int->first;

            ss_tris.push_back(std::distance(search->triangles.begin(), it_int->second));

            K::Point_3 point;
            K::Segment_3 segment;
            if(CGAL::assign(point,object)){
                ss_dists.push_back(sqrt((point - p_v[ss]).squared_length()));
                ss_points.push_back(point);
            }
            if(CGAL::assign(segment,object)){
                K::Point_3 ray_s = p_v[ss];
                K::Vector_3 ray_dir = n_v[ss];
                K::Point_3 seg_s = segment.source();
                K::Vector_3 seg_dir = segment.to_vector();
                K::Vector_3 source_diff = ray_s - seg_s;
                double ts = ((source_diff.y())*ray_dir.x() - (source_diff.x())*ray_dir.y())/(seg_dir.y()*ray_dir.x() - seg_dir.x()*ray_dir.y());
                point = seg_s + ts*seg_dir;
                if(!segment.has_on(point))
                    std::cerr << "ERROR:Debug segment intersection" << std::endl;
                ss_points.push_back(point);
                ss_dists.push_back(sqrt((point - p_v[ss]).squared_length()));
            }
        }
        if(ss_dists.empty()){
            distance[ss] = 1e100;
        }
        else{
            //distance[ss] = *std::min_element(ss_dists.begin(), ss_dists.end());
            size_t idx = std::distance(ss_dists.begin(), std::min_element(ss_dists.begin(), ss_dists.end()));
            distance[ss] = ss_dists[idx];
            for(size_t cc=0; cc<3; ++cc) {
                closest_point[ss][cc]=ss_points[idx][cc];
            }
            closest_triangles[ss] = ss_tris[idx];
        }
    }
    return Py_BuildValue("NNN", result1, result2, result3);
}
