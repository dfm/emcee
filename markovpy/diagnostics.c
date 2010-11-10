#include <Python.h>
#include <numpy/arrayobject.h>


//static PyObject *SpamError;

PyMODINIT_FUNC initdiagnostics(void);
static PyObject *diagnostics_autocorrelation(PyObject *self, PyObject *args);

static PyMethodDef DiagnosticsMethods[] = {
    {"autocorrelation",  diagnostics_autocorrelation, METH_VARARGS,
     "Calculate the autocorrelation time for a Markov chain."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdiagnostics(void)
{
    PyObject *m;

    m = Py_InitModule("diagnostics", DiagnosticsMethods);
    if (m == NULL)
        return;

    import_array();

    //SpamError = PyErr_NewException("spam2.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}

static PyObject *diagnostics_autocorrelation(PyObject *self, PyObject *args)
{
    PyObject *arg = NULL;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    
    PyObject *arr = NULL;
    arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr == NULL) return NULL;
    
    int i,j,k;

    double *dptr = (double *)PyArray_DATA(arr);
    PyObject *out = PyArray_SimpleNew(PyArray_NDIM(arr)-1,PyArray_DIMS(arr),NPY_OUT_ARRAY);
    PyArray_Mean((PyArrayObject*)arr,PyArray_NDIM(arr)-1,PyArray_NOTYPE,(PyArrayObject*)out);
    double *dptr2 = (double *)PyArray_DATA(out);
    
    printf("mean: %d\n",PyArray_NDIM(out));

    for (i = 0; i < (PyArray_DIMS(arr))[0]; i++) {
        for (j = 0; j < (PyArray_DIMS(arr))[1]; j++) {
            printf("whatever: %d, %d\n",i,j);
        }
    }

    Py_DECREF(arr);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(arr);
    return NULL;
}




