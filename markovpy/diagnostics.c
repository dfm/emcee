#include <Python.h>
#include <numpy/arrayobject.h>


PyMODINIT_FUNC init_C_diagnostics(void);
static PyObject *diagnostics_autocorrelation(PyObject *self, PyObject *args);

static PyMethodDef DiagnosticsMethods[] = {
    {"autocorrelation",  diagnostics_autocorrelation, METH_VARARGS,
     "Calculate the autocorrelation time for a Markov chain."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_C_diagnostics(void)
{
    PyObject *m = Py_InitModule("_C_diagnostics", DiagnosticsMethods);
    if (m == NULL)
        return;
    
    // Import numpy utils
    import_array();
}

static PyObject *diagnostics_autocorrelation(PyObject *self, PyObject *args)
{
    PyObject *arg = NULL, *chain_arr = NULL, *mean_arr = NULL, *time_arr = NULL;
    
    // Parse input... it should be a numpy array
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    chain_arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (chain_arr == NULL) return NULL;
    
    int i,j,k;
    npy_intp dims[1] = {1};
    double *dptr = (double *)PyArray_DATA(chain_arr);
    mean_arr = PyArray_SimpleNew(PyArray_NDIM(chain_arr)-1,PyArray_DIMS(chain_arr),PyArray_DOUBLE);
    PyArray_Mean((PyArrayObject*)chain_arr,PyArray_NDIM(chain_arr)-1,PyArray_NOTYPE,(PyArrayObject*)mean_arr);
    double *dptr2 = (double *)PyArray_DATA(mean_arr);
    
    printf("mean: %f\n",dptr2[0]);

    //for (i = 0; i < (PyArray_DIMS(arr))[0]; i++) {
    //    for (j = 0; j < (PyArray_DIMS(arr))[1]; j++) {
    //        printf("whatever: %d, %d\n",i,j);
    //    }
    //}
    
    printf("blah\n");

    Py_DECREF(chain_arr);
    Py_DECREF(mean_arr);
    Py_INCREF(time_arr);
    return time_arr;

fail:
    Py_XDECREF(chain_arr);
    Py_XDECREF(mean_arr);
    Py_XDECREF(time_arr);
    return NULL;
}




