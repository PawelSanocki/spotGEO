#include <Python.h>
#include <arrayobject.h>

int add_a_b(int a, int b){
    return a + b;
}

static PyObject* add(PyObject* self, PyObject* args){
    int sts;
    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b)) return NULL;
    sts = add_a_b(a,b);
    return Py_BuildValue("i",sts);;
}

static PyObject* maxBlur(PyObject* self, PyObject* args){
    PyArrayObject *arrays[2];  /* holds input and output array */
    int size;
    int mask_size;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrays[0])) return NULL;
    sts = add_a_b(a,b);
    return Py_BuildValue("i",sts);;
}

static PyMethodDef filters_methods[] = {
    {"maxBlur", maxBlur, METH_VARARGS, "Performs max blur operation"},
    {"add", add, METH_VARARGS, "Performs adding operation"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef filters = {
 PyModuleDef_HEAD_INIT,
 "filters","",
 -1,
 filters_methods
};

PyMODINIT_FUNC PyInit_filtersC(void){
    import_array();
    return PyModule_Create(&filters);
}
// python setup.py build
// python setup.py install

// https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html#required-subroutine