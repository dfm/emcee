#include <Python.h>
#include <numpy/arrayobject.h>

#include "acor.h"

/* Docstrings */
static char doc[] = "A module to estimate the autocorrelation time of time-series "\
                    "data extremely quickly.\n";
static char acor_doc[] =
"Estimate the autocorrelation time of a time series\n\n"\
"Parameters\n"\
"----------\n"\
"data : numpy.ndarray (N,) or (M, N)\n"\
"    The time series.\n\n"\
"Returns\n"\
"-------\n"\
"tau : numpy.ndarray (1,) or (M,)\n"\
"    An estimate of the autocorrelation times.\n\n";

PyMODINIT_FUNC init_acor(void);
static PyObject *acor_acor(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"acor", acor_acor, METH_VARARGS, acor_doc},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_acor(void)
{
    PyObject *m = Py_InitModule3("_acor", module_methods, doc);
    if (m == NULL)
        return;

    import_array();
}

static PyObject *acor_acor(PyObject *self, PyObject *args)
{
    int i, N, ndim, info;
    double *data;
    double *taus = NULL, mean, sigma;

    /* Return value */
    PyObject *acor_time = NULL;
    npy_intp M[1];

    /* Parse the input tuple */
    PyObject *data_obj;
    if (!PyArg_ParseTuple(args, "O", &data_obj))
        return NULL;

    /* Get the data as a numpy array object */
    PyObject *data_array  = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (data_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "The input data must be a numpy.ndarray.");
        Py_XDECREF(data_array);
        return NULL;
    }

    /* Check the number of dimensions in the input data (must be 1 or 2 only) */
    ndim = (int)PyArray_NDIM(data_array);
    if (ndim > 2 || ndim < 1) {
        PyErr_SetString(PyExc_TypeError, "The input data must be a 1- or 2-D numpy.ndarray.");
        Py_DECREF(data_array);
        return NULL;
    }

    /* Get a pointer to the input data */
    data = (double*)PyArray_DATA(data_array);

    /* N gives the length of the time series */
    N = (int)PyArray_DIM(data_array, ndim-1);

    /* The zeroth (and only) element of M gives the number of series (default to 1) */
    M[0] = 1;
    if (ndim == 2)
        M[0] = (int)PyArray_DIM(data_array, 0);

    /* allocate memory for the output */
    acor_time = PyArray_SimpleNew(1, M, PyArray_DOUBLE);
    if (acor_time == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate memory for output in acor.");
        Py_DECREF(data_array);
        return NULL;
    }

    /* Get a pointer to the output data */
    taus = (double*)PyArray_DATA(acor_time);

    for (i = 0; i < M[0]; i++) {
        info = acor(&mean, &sigma, &(taus[i]), &(data[i*N]), N);
        if (info != 0) {
            PyErr_Format(PyExc_RuntimeError, "The autocorrelation time is too long "\
                                             "relative to the variance in dimension %d.", i+1);
            Py_DECREF(data_array);
            return NULL;
        }
    }

    /* clean up */
    Py_DECREF(data_array);
    return acor_time;
}


