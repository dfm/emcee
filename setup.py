#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension

setup(name='MarkovPy',
        version='0.1',
        description='MarkovPy',
        author='Daniel Foreman-Mackey',
        author_email='dan@danfm.ca',
        packages=['markovpy'],
        ext_modules = [Extension('markovpy.diagnostics', ['markovpy/diagnostics.c'])]
        )

        # ext_modules = [Extension('markovpy/censemble', ['markovpy/censemble.c'])]

