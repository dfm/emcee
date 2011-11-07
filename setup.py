#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

setup(name='PyEST',
        version='0.1',
        description='pyest',
        author='Daniel Foreman-Mackey',
        author_email='dan@danfm.ca',
        packages=['pyest','acor'],
        ext_modules = [Extension('acor._acor',
                        ['acor/acor.cpp','acor/acc.cpp','acor/acc_dfm.cpp',
                        ])],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
        )

