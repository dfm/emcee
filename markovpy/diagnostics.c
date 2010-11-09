#include <Python.h>

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

    //SpamError = PyErr_NewException("spam2.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}

static PyObject *diagnostics_autocorrelation(PyObject *self, PyObject *args)
{
    /*const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }*/

    printf("blah\n");

    return PyLong_FromLong(1.0);
}




