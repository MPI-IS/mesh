#include "plyutils.h"

static PyMethodDef PlyutilsMethods[] = {
    {"read",  plyutils_read, METH_VARARGS, "Read a PLY file."},
    {"write",  plyutils_write, METH_VARARGS, "Write a PLY file."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduleDef =
{
    PyModuleDef_HEAD_INIT,
    "serialization.plyutils", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    PlyutilsMethods
};

static PyObject *PlyutilsError;

PyMODINIT_FUNC PyInit_plyutils(void) {
    PyObject *m = PyModule_Create(&moduleDef);
    if (m == NULL)
        return NULL;

    PlyutilsError = PyErr_NewException("plyutils.error", NULL, NULL);
    Py_INCREF(PlyutilsError);
    PyModule_AddObject(m, "error", PlyutilsError);

    return m;
}

int has_color(p_ply ply) {
    p_ply_element el = NULL;
    p_ply_property p = NULL;
    const char *name;
    while ((el = ply_get_next_element(ply, el))) {
        if (ply_get_element_info(el, &name, NULL) && !strcmp(name, "vertex")) {
            while ((p = ply_get_next_property(el, p))) {
                if (ply_get_property_info(p, &name, NULL, NULL, NULL)
                    && (!strcmp(name, "red") || !strcmp(name, "green") || !strcmp(name, "blue")))
                    return 1;
            }
        }
    }
    return 0;
}

int has_normals(p_ply ply) {
    p_ply_element el = NULL;
    p_ply_property p = NULL;
    const char *name;
    while ((el = ply_get_next_element(ply, el))) {
        if (ply_get_element_info(el, &name, NULL) && !strcmp(name, "vertex")) {
            while ((p = ply_get_next_property(el, p))) {
                if (ply_get_property_info(p, &name, NULL, NULL, NULL)
                    && (!strcmp(name, "nx") || !strcmp(name, "ny") || !strcmp(name, "nz")))
                    return 1;
            }
        }
    }
    return 0;
}

static PyObject * plyutils_read(PyObject *self, PyObject *args)
{
    const char *filename;
    p_ply ply = NULL;
    int use_color, use_normals;
    long n_verts, n_faces;
    PyObject *x, *y, *z, *r, *g, *b;
    PyObject *nx, *ny, *nz;
    PyObject *tri;
    
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        PyErr_SetString(PlyutilsError, "plyutils.read doesn't know what to do without a filename.");
        return NULL;
    }
    ply = ply_open(filename, error_cb);
    if (!ply) {
        PyErr_SetString(PlyutilsError, "Failed to open PLY file.");
        return NULL;
    }
    if (!ply_read_header(ply)) {
        PyErr_SetString(PlyutilsError, "plyread_mex: Bad raw header.");
        return NULL;
    }

    use_color = has_color(ply);
    use_normals = has_normals(ply);

    n_verts = ply_set_read_cb(ply, "vertex", "x", vertex_cb, (void*)&x, 0);
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, (void*)&y, 0);
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, (void*)&z, 0);
    if (use_color) {
        ply_set_read_cb(ply, "vertex", "red", vertex_cb, (void*)&r, 0);
        ply_set_read_cb(ply, "vertex", "green", vertex_cb, (void*)&g, 0);
        ply_set_read_cb(ply, "vertex", "blue", vertex_cb, (void*)&b, 0);
    }
    if (use_normals) {
        ply_set_read_cb(ply, "vertex", "nx", vertex_cb, (void*)&nx, 0);
        ply_set_read_cb(ply, "vertex", "ny", vertex_cb, (void*)&ny, 0);
        ply_set_read_cb(ply, "vertex", "nz", vertex_cb, (void*)&nz, 0);
    }
    n_faces = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, (void*)&tri, 0);
    if (n_faces==0)
        n_faces = ply_set_read_cb(ply, "face", "vertex_index", face_cb, (void*)&tri, 0);

    x = PyList_New(n_verts); y = PyList_New(n_verts); z = PyList_New(n_verts);
    if (use_color) {
        r = PyList_New(n_verts); g = PyList_New(n_verts); b = PyList_New(n_verts);
    }
    if (use_normals) {
        nx = PyList_New(n_verts); ny = PyList_New(n_verts); nz = PyList_New(n_verts);
    }
    tri = Py_BuildValue("[N,N,N]", PyList_New(n_faces), PyList_New(n_faces), PyList_New(n_faces));

    if (!ply_read(ply)) {
        char * msg = "Read failed. ";
        char* catString = malloc(strlen(msg)+strlen(filename)+1);
          strcpy(catString, msg);
          strcat(catString, filename);

        PyErr_SetString(PlyutilsError, catString);
        // use the string then delete it when you're done.
         free(catString);
         return NULL;
    }
    ply_close(ply);
    
    if (use_color && !use_normals)
        return Py_BuildValue("{s:[N,N,N],s:N,s:[N,N,N]}", "pts", x, y, z, "tri", tri, "color", r, g, b);
    if (!use_color && use_normals)
        return Py_BuildValue("{s:[N,N,N],s:N,s:[N,N,N]}", "pts", x, y, z, "tri", tri, "normals", nx, ny, nz);
    if (use_color && use_normals)
        return Py_BuildValue("{s:[N,N,N],s:N,s:[N,N,N],s:[N,N,N]}", "pts", x, y, z, "tri", tri, "color", r, g, b, "normals", nx, ny, nz);
    else
        return Py_BuildValue("{s:[N,N,N],s:N}", "pts", x, y, z, "tri", tri);
}

static PyObject * plyutils_write(PyObject *self, PyObject *args)
{
    const char *filename;
    PyObject *pts, *tri, *color, *ascii, *little_endian, *comments;
    PyObject *normals = NULL;
    int use_color, use_normals, res;
    p_ply ply = NULL;
    PyObject *row;
    long ii;
    const char *comment;

    if (!PyArg_ParseTuple(args, "OOOsO|O|OO", &pts, &tri, &color, &filename, &ascii, &little_endian, &comments, &normals))
        return NULL;

    use_color = (PyList_Size(pts) == PyList_Size(color));
    use_normals = 0;
    if (normals!=NULL)
        use_normals = (PyList_Size(pts) == PyList_Size(normals));
    
    if (ascii == Py_True)
        ply = ply_create(filename, PLY_ASCII, error_cb);
    else {
      if (little_endian == Py_True)
        ply = ply_create(filename, PLY_LITTLE_ENDIAN, error_cb);
      else
        ply = ply_create(filename, PLY_BIG_ENDIAN, error_cb);
    }
    
    if (!ply) {
        PyErr_SetString(PlyutilsError, "Failed to create PLY file.");
        return NULL;
    }
    
    res = 1;
    
    for (ii = 0; ii < PyList_Size(comments); ++ii) {
        comment = PyBytes_AsString(PyObject_Str(PyList_GetItem(comments, ii)));
        res &= ply_add_comment(ply, comment);
    }
    
    res &= ply_add_element(ply, "vertex", PyList_Size(pts));
    res &= ply_add_scalar_property(ply, "x", PLY_FLOAT);
    res &= ply_add_scalar_property(ply, "y", PLY_FLOAT);
    res &= ply_add_scalar_property(ply, "z", PLY_FLOAT);

    if(use_normals){
        res &= ply_add_scalar_property(ply, "nx", PLY_FLOAT);
        res &= ply_add_scalar_property(ply, "ny", PLY_FLOAT);
        res &= ply_add_scalar_property(ply, "nz", PLY_FLOAT);
    }

    if(use_color){
        res &= ply_add_scalar_property(ply, "red", PLY_UCHAR);
        res &= ply_add_scalar_property(ply, "green", PLY_UCHAR);
        res &= ply_add_scalar_property(ply, "blue", PLY_UCHAR);
    }

    res &= ply_add_element(ply, "face", PyList_Size(tri));
    res &= ply_add_list_property(ply, "vertex_indices", PLY_UCHAR, PLY_INT);

    res &= ply_write_header(ply);
    if (!res) {
        PyErr_SetString(PlyutilsError, "Failed to write header.");
        return NULL;
    }


    
    for (ii = 0; ii < PyList_Size(pts); ++ii) {
        row = PyList_GetItem(pts, ii);
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 0)));
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 1)));
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 2)));
        if(use_normals){
            row = PyList_GetItem(normals, ii);
            res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 0)));
            res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 1)));
            res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 2)));
        }
        if(use_color){
            row = PyList_GetItem(color, ii);
            res &= ply_write(ply, (unsigned char)PyLong_AsUnsignedLongMask(PyList_GetItem(row, 0)));
            res &= ply_write(ply, (unsigned char)PyLong_AsUnsignedLongMask(PyList_GetItem(row, 1)));
            res &= ply_write(ply, (unsigned char)PyLong_AsUnsignedLongMask(PyList_GetItem(row, 2)));
        }
    }
    if (!res) {
        PyErr_SetString(PlyutilsError, "Error writing points.");
        return NULL;
    }

    for (ii = 0; ii < PyList_Size(tri); ++ii) {
        row = PyList_GetItem(tri, ii);
        res &= ply_write(ply, 3);
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 0)));
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 1)));
        res &= ply_write(ply, PyFloat_AsDouble(PyList_GetItem(row, 2)));
    }
    if (!res) {
        PyErr_SetString(PlyutilsError, "Error writing faces.");
        return NULL;
    }

    ply_close(ply);
    Py_INCREF(Py_None);
    return Py_None;
}

int vertex_cb(p_ply_argument argument) {
    void *p; PyObject* list;
    long ii;
    PyObject *val;

    ply_get_argument_element(argument, NULL, &ii);
    ply_get_argument_user_data(argument, &p, NULL);
    list = (PyObject*)(*(void**)p);

    val = PyFloat_FromDouble(ply_get_argument_value(argument));
    // PyList_Append(list, val);
    // Py_DECREF(val);
    PyList_SET_ITEM(list, ii, val);

    return 1;
}

int face_cb(p_ply_argument argument) {
    void *p; PyObject* tri;
    long ii;
    long length, value_index;
    PyObject *val;

    ply_get_argument_element(argument, NULL, &ii);
    ply_get_argument_user_data(argument, &p, NULL);
    tri = (PyObject*)(*(void**)p);

    ply_get_argument_property(argument, NULL, &length, &value_index);
    if (value_index >= 0 && value_index < PyList_Size(tri)) {
        PyObject* slice = PyList_GetItem(tri, value_index);

        val = PyFloat_FromDouble(ply_get_argument_value(argument));
        // PyList_Append(slice, val);
        // Py_DECREF(val);
        PyList_SET_ITEM(slice, ii, val);
    }

    return 1;
}

void error_cb(const char *message) {
    PyErr_SetString(PlyutilsError, message);
}
