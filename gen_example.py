#!/usr/bin/env python
# encoding: utf-8

import os
import pystache

class ExampleView(pystache.View):
    template_path = os.path.join(os.path.join(*(os.path.split(__file__)[:-1])), "views")
    template_name = "example"

    def __init__(self, in_file, **kwargs):
        super(ExampleView, self).__init__(**kwargs)
        self.code = open(in_file).read()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print "You must provide and input and output file"
        sys.exit(1)
    open(sys.argv[2], 'w').write(ExampleView(sys.argv[1]).render())

