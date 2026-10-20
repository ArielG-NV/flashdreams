// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "flashdreams.h"

static int fail(char *buffer, size_t capacity, const char *message) {
    if (buffer && capacity) {
        snprintf(buffer, capacity, "%s", message);
    }
    return 1;
}

int fd_run(const char *python_executable, int argc, const char *const *argv,
           const fd_client_window *window, char *error_buffer,
           size_t error_capacity) {
    if (error_buffer && error_capacity) {
        error_buffer[0] = '\0';
    }
    if (argc < 0 || (argc && !argv)) {
        return fail(error_buffer, error_capacity, "Invalid argc/argv.");
    }
    for (int i = 0; i < argc; ++i) {
        if (!argv[i]) {
            return fail(error_buffer, error_capacity, "NULL argv entry.");
        }
    }
    if (window && (window->struct_size != sizeof(*window) ||
                   window->abi_version != FD_ABI_VERSION || !window->open ||
                   !window->get_user_input_events || !window->write || !window->close)) {
        return fail(error_buffer, error_capacity,
                    "Invalid window ABI or missing required window callback.");
    }
    if (!Py_IsInitialized()) {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        config.parse_argv = 0;
        config.install_signal_handlers = 0;
        PyStatus status = PyStatus_Ok();
        if (python_executable) {
            status = PyConfig_SetBytesString(&config, &config.program_name,
                                             python_executable);
            if (!PyStatus_Exception(status)) {
                status = PyConfig_SetBytesString(&config, &config.executable,
                                                 python_executable);
            }
        }
        if (!PyStatus_Exception(status)) {
            status = Py_InitializeFromConfig(&config);
        }
        if (PyStatus_Exception(status)) {
            int result = fail(error_buffer, error_capacity,
                              status.err_msg ? status.err_msg : "Python initialization failed.");
            PyConfig_Clear(&config);
            return result;
        }
        PyConfig_Clear(&config);
        PyEval_SaveThread();
    }

    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *arguments = PyList_New(argc);
    PyObject *module = NULL, *function = NULL, *address = NULL, *result = NULL;
    if (!arguments) {
        goto done;
    }
    for (int i = 0; i < argc; ++i) {
        PyObject *item = PyUnicode_DecodeUTF8(argv[i], (Py_ssize_t)strlen(argv[i]), "strict");
        if (!item) {
            goto done;
        }
        PyList_SET_ITEM(arguments, i, item);
    }
    module = PyImport_ImportModule("flashdreams.runtime_v2.c_client_window");
    if (!module) {
        goto done;
    }
    function = PyObject_GetAttrString(module, "_run");
    address = PyLong_FromVoidPtr((void *)window);
    if (function && address) {
        result = PyObject_CallFunctionObjArgs(function, arguments, address, NULL);
    }

done:;
    int exit_code = result ? 0 : 1;
    if (!result) {
        PyObject *type = NULL, *value = NULL, *traceback = NULL;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        if (type && !PyErr_GivenExceptionMatches(type, PyExc_SystemExit)) {
            PyErr_Display(type, value, traceback);
        }
        PyObject *message = value ? PyObject_Str(value) : NULL;
        const char *utf8 = message ? PyUnicode_AsUTF8(message) : NULL;
        fail(error_buffer, error_capacity, utf8 ? utf8 : "Python launch failed.");
        Py_XDECREF(message);
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        PyErr_Clear();
    }
    Py_XDECREF(result);
    Py_XDECREF(address);
    Py_XDECREF(function);
    Py_XDECREF(module);
    Py_XDECREF(arguments);
    PyGILState_Release(gil);
    return exit_code;
}
