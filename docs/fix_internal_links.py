#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

if len(sys.argv) <= 1:
    sys.exit(0)


def subber(m):
    return m.group(0).replace("``", "`")


prog = re.compile(r":(.+):``(.+)``")

for fn in sys.argv[1:]:
    print("Fixing links in {0}".format(fn))
    with open(fn, "r") as f:
        txt = f.read()
    txt = prog.sub(subber, txt)
    with open(fn, "w") as f:
        f.write(txt)
