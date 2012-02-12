#!/usr/bin/env python
# encoding: utf-8

import os
import re
import inspect

import pystache
import markdown

class DocstringParser(object):
    section_otag = "####"
    section_ctag = ""
    sections = {"Arguments": "args",
        "Keyword Arguments": "kwargs",
               "Exceptions": "errors",
                  "Returns": "returns",
                   "Yields": "yields"}

    def __init__(self):
        tags = {'otag': re.escape(self.section_otag),
                'ctag': re.escape(self.section_ctag)}
        self.section_re = re.compile(r"%(otag)s(.*)%(ctag)s"%tags)
        self.pgroup_re  = re.compile(r"\s+?\*\s", re.M|re.S)
        self.param_re   = re.compile(r"`(.*?)`.*?(\(.*?\))?:(.+)", re.M|re.S)

    def parse(self, obj):
        parsed = {}
        try:
            parsed["name"] = obj.__name__
        except AttributeError:
            pass

        # Inspect the object to get the calling sequence.
        argspec = None
        if inspect.isclass(obj):
            argspec = inspect.getargspec(obj.__init__)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            argspec = inspect.getargspec(obj)

        if argspec is not None:
            # Enumerate the arguments.
            args = []
            delta = len(argspec.args)
            if argspec.defaults is not None:
                delta -= len(argspec.defaults)
            for i, a in enumerate(argspec.args[:delta]):
                if a != "self":
                    args.append(a)

            # Catch the `*args` variable if it exists.
            if argspec.varargs is not None:
                args.append("*%s"%argspec.varargs)

            # Enumerate the keyword arguments.
            for i, a in enumerate(argspec.args[delta:]):
                de = argspec.defaults[i]
                if isinstance(de, str):
                    de = "'" + de + "'"
                args.append(a + "=%s"%(str(de)))

            # Catch `**kwargs`.
            if argspec.keywords is not None:
                args.append("**%s"%argspec.keywords)
            parsed["init_args"] = ", ".join(args)

        # Get the docstring.
        docstring = inspect.getdoc(obj)
        if docstring is None:
            return parsed

        # Parse the docstring
        current_section = "desc"
        parsed["desc"] = []
        for line in docstring.split("\n\n"):
            line = line.strip()
            sect = self.section_re.search(line)
            if sect is not None and len(sect.groups()) > 0:
                try:
                    current_section = self.sections[sect.groups()[0].strip()]
                    parsed[current_section] = []
                except KeyError:
                    current_section = None
            else:
                if current_section == "desc":
                    parsed["desc"] += [line]
                elif current_section in self.sections.values():
                    for p in self.pgroup_re.split(line):
                        param = self.param_re.search(p)
                        if param is not None:
                            param = param.groups()
                            r = {}
                            r["name"]  = param[0].strip()
                            if current_section != "errors":
                                r["dtype"] = param[1].strip("()")
                            r["desc"]  = " ".join([c.strip()
                                            for c in param[-1].split("\n")])
                            parsed[current_section].append(r)

        parsed["desc"] = "\n\n".join(parsed["desc"])

        # Get the methods and properties
        members = inspect.getmembers(obj)
        parser = DocstringParser()
        parsed["properties"] = []
        parsed["_methods"]    = []
        for member in members:
            if member[0][0] != "_":
                m = getattr(obj, member[0])
                if inspect.isdatadescriptor(m):
                    doc = parser.parse(m)
                    doc["name"] = member[0]
                    if "desc" not in doc:
                        doc["desc"] = ""
                    parsed["properties"].append(doc)
                elif inspect.ismethod(m):
                    doc = parser.parse(m)
                    doc["name"] = member[0]
                    if "desc" in doc:
                        parsed["_methods"].append(doc)

        return parsed

class ObjectView(pystache.View):
    template_path = os.path.join("views", "latex")
    template_name = "object"

    _code_re = re.compile("`(.+?)`")
    _quotes_re = re.compile("\"(.+?)\"", re.M|re.S)
    _echars = "\\%#~&_"

    def __init__(self, obj, **kwargs):
        super(ObjectView, self).__init__(**kwargs)
        self._vals = self.escape(obj)

    def escape(self, s):
        if isinstance(s, (list, tuple)):
            return [self.escape(o) for o in s]
        elif isinstance(s, dict):
            return dict([(k, self.escape(s[k])) for k in s])
        for c in self._echars:
            s = s.replace(c, "\\"+c)
        s = s.replace("<", "\\texttt{<}")
        s = s.replace(">", "\\texttt{>}")
        _tex_code = lambda s: "\code{" + s.group(0)[1:-1] + "}"
        s = self._code_re.sub(_tex_code, s)
        _tex_quote = lambda s: "``" + s.group(0)[1:-1] + "''"
        s = self._quotes_re.sub(_tex_quote, s)
        return s

    def has(self, attr):
        return hasattr(self, attr) and len(getattr(self, attr)) > 0

    def __getattr__(self, attr):
        if "has_" in attr:
            pref, suff = attr[:4], attr[4:]
            if pref != "has_":
                raise AttributeError
            return self.has(suff)
        return self._vals[attr]

    @property
    def methods(self):
        return [MethodView(m) for m in self._methods]

class MethodView(ObjectView):
    template_name = "method"

    def escape(self, s):
        return s

class HTMLView(ObjectView):
    template_path = os.path.join("views", "html")

    def escape(self, s):
        if isinstance(s, (list, tuple)):
            return [self.escape(o) for o in s]
        elif isinstance(s, dict):
            return dict([(k, self.escape(s[k])) for k in s])
        _tex_code = lambda s: "<code>" + s.group(0)[1:-1] + "</code>"
        s = self._code_re.sub(_tex_code, s)
        return s

    @property
    def methods(self):
        return [HTMLMethodView(m).set_class(self._vals["name"]) for m in self._methods]

    @property
    def desc(self):
        return markdown.markdown(self._vals["desc"])

class HTMLMethodView(HTMLView):
    template_name = "method"

    def escape(self, s):
        return s

    def set_class(self, cn):
        self.class_name = cn
        return self

if __name__ == '__main__':
    import sys
    import emcee
    parser = DocstringParser()

    if "--html" in sys.argv or len(sys.argv) == 1:
        print HTMLView(parser.parse(emcee.Sampler)).render()
        print HTMLView(parser.parse(emcee.EnsembleSampler)).render()
        print HTMLView(parser.parse(emcee.ensemble.Ensemble)).render()
    if "--tex" in sys.argv or len(sys.argv) == 1:
        print ObjectView(parser.parse(emcee.Sampler)).render()
        print ObjectView(parser.parse(emcee.EnsembleSampler)).render()
        print ObjectView(parser.parse(emcee.ensemble.Ensemble)).render()

