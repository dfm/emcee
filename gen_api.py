#!/usr/bin/env python
# encoding: utf-8
"""
Generate the API Docs for PyEST.

"""

import re
import inspect

import pyest

res = ['(.*) : (.*) (\(.*\)), (optional)', '(.*) : (.*)( \(.*\))',
        '(.*) : (.*), (optional)', '(.*) : (.*)']
res = [re.compile(r) for r in res]

module_name = 'pyest'

cls = pyest.EnsembleSampler

end_marks = ['Parameters', 'Returns', 'Note', 'References']

def iter_lines(pos, lines):
    ret = ''
    flag = False
    end_mark = None
    for pos in range(pos,len(lines)):
        line = lines[pos]
        if line in end_marks:
            end_mark = line
            flag = True
        elif flag:
            break
        elif ret != '' or line != '':
            ret += line + '\n'
    return pos+1, ret, end_mark

def parse_params(params):
    param_list = params.split('\n\n')
    ret = []
    for param in param_list:
        lines = param.splitlines()
        if len(lines) > 1:
            ret.append({'description': '\n'.join(lines[1:])})
            for i, r in enumerate(res):
                m = r.search(lines[0])
                if m is not None:
                    g = m.groups()
                    ret[-1]['param'] = g[0]
                    ret[-1]['type']  = g[1]
                    if i in [0,1]:
                        ret[-1]['shape'] = g[2]
                    else:
                        ret[-1]['shape'] = None
                    if i in [0,2]:
                        ret[-1]['optional'] = True
                    else:
                        ret[-1]['optional'] = False
                    break
    return ret

def param_template(params):
    template = ""
    for p in params:
        template += """
        <tr>
            <td width="100"><code>%(param)s</code></td>
            <td width="100">%(type)s
        """%(p)
        if p['shape'] is not None:
            template += "<br> %(shape)s"%(p)

        template +="""</td>
            <td>
        """

        if p['optional']:
            template += '<span class="label notice">Optional</span> '

        template += """%(description)s</td>
        </tr>
        """%(p)
    return template

def parse_docstring(m):
    ds = m.__doc__

    # argspec
    if m == cls:
        argspec = inspect.getargspec(m.__init__)
    else:
        argspec = inspect.getargspec(m)
    args = []
    try:
        delta = len(argspec.args)-len(argspec.defaults)
    except:
        delta = len(argspec.args)
    for i,a in enumerate(argspec.args):
        if i >= delta:
            de = argspec.defaults[i-delta]
            if isinstance(de, str):
                de = "'" + de + "'"
            args.append(a + "=%s"%(str(de)))
        elif a != 'self':
            args.append(a)
    args = ", ".join(args)

    pos = 2
    lines = ds.splitlines()
    lines = [line.strip() for line in lines]
    title = lines[1]
    pos, preamble, end_mark = iter_lines(pos, lines)

    template = """
<div class="row">
    <div class="span16">
    """
    if m != cls:
        template += "<h3><small>%s.%s.</small>%s<small>(%s)</small></h3>"%(module_name, cls.__name__, m.__name__, args)
    else:
        template += "<h3><small>%s.</small>%s<small>(%s)</small></h3>"%(module_name, cls.__name__, args)
    template += """
    </div>
</div>
<div class="row">
    <div class="span13 offset3">
        <h5>%s</h5>
        <p>
%s
        </p>
    </div>
</div>
        """%(title, preamble)


    if end_mark == 'Parameters':
        pos, params, end_mark = iter_lines(pos, lines)
        params = parse_params(params)

        template += """
<div class="row">
    <div class="span2 offset1">
        <h4>Parameters</h4>
    </div>
    <div class="span13">
        <table class="zebra-striped">
            <tr>
                <th width="100">Parameter</td>
                <th width="100">Type</td>
                <th>Description</td>
            </tr>
        """

        template += param_template(params)

        template += """
        </table>
    </div>
</div>
        """

    if end_mark == 'Returns':
        pos, returns, end_mark= iter_lines(pos, lines)
        returns = parse_params(returns)

        template += """
<div class="row">
    <div class="span2 offset1">
        <h4>Returns</h4>
    </div>
    <div class="span13">
        <table class="zebra-striped">
        """

        template += param_template(returns)

        template += """
        </table>
    </div>
</div>
        """


    return template

meths = [cls, cls.run_mcmc, cls.sample, cls.clear_chain, cls.ensemble_lnposterior]

api = "\n".join([parse_docstring(m) for m in meths])

print api

