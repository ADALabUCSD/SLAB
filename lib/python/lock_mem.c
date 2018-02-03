#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <Python.h>
#include <sys/mman.h>

static PyObject *_memlock(PyObject *self, PyObject *args)
{
    key_t key;
    size_t size;
    long bytes, key_seed;
    struct shmid_ds *res;
    int segment_id, rc;

    if (!PyArg_ParseTuple(args, "ll", &bytes, &key_seed))
        return NULL;

    size = (size_t) bytes;
    if ((key = ftok("/tmp", (int) key_seed)) == (key_t) -1)
        return PyErr_Format(PyExc_OSError,
            "Could not create key. OS Returned error: %d - %s",
            errno, strerror(errno));

    segment_id = shmget(key, size, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR);
    if (segment_id == -1)
        return PyErr_Format(PyExc_OSError,
            "Could not allocate segment of size: %zu. OS Returned error: %d - %s",
            size, errno, strerror(errno));

    if ((rc = shmctl(segment_id, SHM_LOCK, res)) == -1)
        return PyErr_Format(PyExc_OSError,
            "Could not lock segment. OS Returned error: %d - %s",
            errno, strerror(errno));

    return Py_BuildValue("l", segment_id);
}

static PyMethodDef _memlock_methods[] = {
    {"_memlock", _memlock, METH_VARARGS, "Lock a portion of shared memory"},
    {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC init_memlock(void)
{
    (void) Py_InitModule("_memlock", _memlock_methods);
}
