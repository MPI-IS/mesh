
// needed to avoid the link to debug "_d.lib" libraries
#include "hijack_python_headers.hpp"

#include <numpy/arrayobject.h>
#include <iostream>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
using boost::uint32_t;
using boost::array;

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/array.hpp>
#include <vector>
#include <map>

class LoadObjException: public std::exception {
public:
    LoadObjException(std::string m="loadObjException!"):msg(m) {}
    ~LoadObjException() throw() {}
    const char* what() const throw() { return msg.c_str(); }
private:
    std::string msg;
};

static PyObject *
loadobj(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *LoadObjError;

static PyMethodDef loadobj_methods[] = {
    {"loadobj",  (PyCFunction) loadobj,
        METH_VARARGS | METH_KEYWORDS, "loadobj."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduleDef =
{
    PyModuleDef_HEAD_INIT,
    "serialization.loadobj", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    loadobj_methods
};

PyMODINIT_FUNC PyInit_loadobj(void) {
    PyObject *m = PyModule_Create(&moduleDef);
    if (m == NULL)
        return NULL;

    import_array();
    LoadObjError = PyErr_NewException(const_cast<char*>("loadobj.LoadObjError"), NULL, NULL);
    Py_INCREF(LoadObjError);
    PyModule_AddObject(m, "LoadObjError", LoadObjError);

    return m;
}

static PyObject *
loadobj(PyObject *self, PyObject *args, PyObject *keywds)
{
    try {
        char py_objpatharr[256];
        char *py_objpath = static_cast<char*>(py_objpatharr);

        // a copy of the literal string is done into a (non const) char
        char key1[] = "obj_path";
        static char* kwlist[] = {key1, NULL};

        if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &py_objpath))
            return NULL;

        std::ifstream obj_is(py_objpath,std::ios_base::binary | std::ios_base::in);
        if (!obj_is) {
            PyErr_SetString(PyExc_ValueError, "Could not load file");
            return NULL;
        }

        std::vector<double> v;
        std::vector<double> vt;
        std::vector<double> vn;
        std::vector<uint32_t> f;
        std::vector<uint32_t> ft;
        std::vector<uint32_t> fn;
        v.reserve(30000);
        vt.reserve(30000);
        vn.reserve(30000);
        f.reserve(100000);
        ft.reserve(100000);
        fn.reserve(100000);
        std::map<std::string, std::vector<uint32_t> > segm;

        bool next_v_is_land = false;
        std::string land_name("");
        std::map<std::string, uint32_t> landm;
        
        std::string line;
        std::string curr_segm("");
        std::string mtl_path("");
        unsigned len_vt = 3;
        while (getline(obj_is, line)) {
            if (line.substr(0,6) == "mtllib") {
                mtl_path = line.substr(6);
            }
                
            if (line.substr(0,1) == "g"){
                curr_segm  = line.substr(2);
                if (segm.find(curr_segm) == segm.end())
                    segm[curr_segm] = std::vector<uint32_t>();
            }
            if (line.substr(0,2) == "vt"){
                std::istringstream is(line.substr(2));
                unsigned orig_vt_len = vt.size();
                std::copy(std::istream_iterator<double>(is),
                          std::istream_iterator<double>(),
                          std::back_inserter(vt));
                len_vt = vt.size() - orig_vt_len;
            }
            else if (line.substr(0,2) == "vn"){
                std::istringstream is(line.substr(2));
                std::copy(std::istream_iterator<double>(is),
                          std::istream_iterator<double>(),
                          std::back_inserter(vn));
            }
            else if (line.substr(0,1) == "f"){
                std::istringstream is(line.substr(1));
                std::istream_iterator<std::string> it(is);
                const std::string delims("/");
                std::vector<uint32_t> localf, localfn, localft;
                for(;it!=std::istream_iterator<std::string>();++it){
                    // valid:  v   v/vt   v/vt/vn   v//vn
                    unsigned counter=0;
                    std::istringstream unparsed_face(*it);
                    std::string el;
                    while(std::getline(unparsed_face, el, '/')) {
                        if (el.size() > 0) { // if the element has contents
                            if (counter == 0)
                                localf.push_back(atoi(el.c_str()));
                            if (counter == 1)
                                localft.push_back(atoi(el.c_str()));
                            if (counter == 2)
                                localfn.push_back(atoi(el.c_str()));
                        }
                        counter++;
                    }
                }
                if (localf.size() > 0) {
                    for (int i=1; i<(localf.size()-1); ++i) {
                        f.push_back(localf[0] - 1);
                        f.push_back(localf[i] - 1);
                        f.push_back(localf[i+1] - 1);
                        if (curr_segm != "")
                            segm.find(curr_segm)->second.push_back((f.size()/3)-1);
                    }
                }
                if (localft.size() > 0) {
                    for (int i=1; i<(localft.size()-1); ++i){
                        ft.push_back(localft[0] - 1);
                        ft.push_back(localft[i] - 1);
                        ft.push_back(localft[i+1] - 1);
                    }
                }
                if (localfn.size() > 0) {
                    for (int i=1; i<(localfn.size()-1); ++i){
                        fn.push_back(localfn[0] - 1);
                        fn.push_back(localfn[i] - 1);
                        fn.push_back(localfn[i+1] - 1);
                    }
                }
            }
            else if (line.substr(0,1) == "v"){
                std::istringstream is(line.substr(1));
                std::copy(std::istream_iterator<double>(is),
                          std::istream_iterator<double>(),
                          std::back_inserter(v));
                if (next_v_is_land) {
                    next_v_is_land = false;
                    landm[land_name.c_str()] = v.size()/3-1;
                }
            }
            else if (line.substr(0,9) == "#landmark"){
                next_v_is_land = true;
                land_name = line.substr(10);
            }
        }

        unsigned n_v = v.size()/3;
        unsigned n_vt = vt.size()/len_vt;
        unsigned n_vn = vn.size()/3;
        unsigned n_f = f.size()/3;
        unsigned n_ft = ft.size()/3;
        unsigned n_fn = fn.size()/3;
        npy_intp v_dims[] = {n_v,3};
        npy_intp vn_dims[] = {n_vn,3};
        npy_intp vt_dims[] = {n_vt,len_vt};
        npy_intp f_dims[] = {n_f,3};
        npy_intp ft_dims[] = {n_ft,3};
        npy_intp fn_dims[] = {n_fn,3};
        /*
        // XXX Memory from vectors get deallocated!
        PyObject *py_v = PyArray_SimpleNewFromData(2, v_dims, NPY_DOUBLE, v.data());
        PyObject *py_vt = PyArray_SimpleNewFromData(2, vt_dims, NPY_DOUBLE, vt.data());
        PyObject *py_vn = PyArray_SimpleNewFromData(2, vn_dims, NPY_DOUBLE, vn.data());
        PyObject *py_f = PyArray_SimpleNewFromData(2, f_dims, NPY_UINT32, f.data());
        PyObject *py_ft = PyArray_SimpleNewFromData(2, ft_dims, NPY_UINT32, ft.data());
        PyObject *py_fn = PyArray_SimpleNewFromData(2, fn_dims, NPY_UINT32, fn.data());
        */
        // The following copy would be faster in C++11 with move semantics
        PyObject *py_v = PyArray_SimpleNew(2, v_dims, NPY_DOUBLE);
        std::copy(v.begin(), v.end(), reinterpret_cast<double*>(PyArray_DATA(py_v)));
        PyObject *py_vt = PyArray_SimpleNew(2, vt_dims, NPY_DOUBLE);
        std::copy(vt.begin(), vt.end(), reinterpret_cast<double*>(PyArray_DATA(py_vt)));
        PyObject *py_vn = PyArray_SimpleNew(2, vn_dims, NPY_DOUBLE);
        std::copy(vn.begin(), vn.end(), reinterpret_cast<double*>(PyArray_DATA(py_vn)));
        PyObject *py_f = PyArray_SimpleNew(2, f_dims, NPY_UINT32);
        std::copy(f.begin(), f.end(), reinterpret_cast<uint32_t*>(PyArray_DATA(py_f)));
        PyObject *py_ft = PyArray_SimpleNew(2, ft_dims, NPY_UINT32);
        std::copy(ft.begin(), ft.end(), reinterpret_cast<uint32_t*>(PyArray_DATA(py_ft)));
        PyObject *py_fn = PyArray_SimpleNew(2, fn_dims, NPY_UINT32);
        std::copy(fn.begin(), fn.end(), reinterpret_cast<uint32_t*>(PyArray_DATA(py_fn)));

        PyObject *py_landm = PyDict_New();
        for (std::map<std::string, uint32_t>::iterator it=landm.begin(); it!=landm.end(); ++it)
            PyDict_SetItemString(py_landm, it->first.c_str(), Py_BuildValue("l", it->second));
        
        PyObject *py_segm = PyDict_New();
        for (std::map<std::string, std::vector<uint32_t> >::iterator it=segm.begin(); it!=segm.end(); ++it) {
            unsigned n = it->second.size();
            npy_intp dims[] = {n};
            PyObject *temp = PyArray_SimpleNew(1, dims, NPY_UINT32);
            std::copy(it->second.begin(), it->second.end(), reinterpret_cast<uint32_t*>(PyArray_DATA(temp)));
            PyDict_SetItemString(py_segm, it->first.c_str(), Py_BuildValue("N", temp));
        }

        return Py_BuildValue("NNNNNNsNN",py_v,py_vt,py_vn,py_f,py_ft,py_fn,mtl_path.c_str(),py_landm,py_segm);
    } catch (LoadObjException& e) {
        PyErr_SetString(LoadObjError, e.what());
        return NULL;
    }
}
