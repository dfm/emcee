# 
#  __init__.py
#  markovpy
#  
#  Created by Dan F-M on 2010-08-10.
#  Copyright 2010 Daniel Foreman-Mackey. All rights reserved.
# 

from mcfit import *
from mcsampler import *
from ensemble import *

VERSION = (1, 0, 0, 'alpha', 0)

def get_version():
    version = '%s.%s' % (VERSION[0], VERSION[1])
    if VERSION[2]:
        version = '%s.%s' % (version, VERSION[2])
    if VERSION[3:] == ('alpha', 0):
        version = '%s pre-alpha' % version
    else:
        if VERSION[3] != 'final':
            version = '%s %s %s' % (version, VERSION[3], VERSION[4])
    return version
