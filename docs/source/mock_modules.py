#!/usr/bin/env python

import inspect
import os
import pathlib
import re
import shutil
import sys

import mirheo

mock_name = 'mmirheo'

def simplify_docstring(docstr):
    if docstr is None:
        return None

    strip_self = re.sub('self:\s+libmirheo[^\s\)]+\s+', '', docstr)
    strip_libmir = re.sub('libmirheo\.', '', strip_self)
    return strip_libmir

def class_members(cls):
    return inspect.getmembers(cls, lambda x : inspect.isfunction(x) or
                                              ( hasattr(x, '__name__') and hasattr(cls, x.__name__) and
                                                type(getattr(cls, x.__name__)).__name__ == 'instancemethod') )

def class_properties(cls):
    return inspect.getmembers(cls, lambda x : x.__class__.__name__ == 'property')



def genmodule(name, fname, needfuncs):
    classes = inspect.getmembers(sys.modules[name], inspect.isclass)
    classes = sorted(classes, key = lambda cls: len(cls[1].mro()))

    fout = open(fname, 'w')

    for cname, cls in classes:
        base = ', '.join( [b.__name__ for b in cls.__bases__ if 'pybind11' not in b.__name__] )

        if len(base) > 0:
            signature = '%s(%s)' % (cname, base)
        else:
            signature = cname

        print('class %s:' % signature, file=fout)
        print('    r"""%s\n    """' % simplify_docstring(cls.__doc__), file=fout)

        members = class_members(cls)
        for mname, m in members:
            if mname == '__init__' or mname[0:2] != '__':
                print('    def %s():' % mname, file=fout)
                print('        r"""%s\n        """' % simplify_docstring(m.__doc__), file=fout)
                print('        pass\n', file=fout)

        for pname, p in class_properties(cls):
            print('    @property', file=fout)
            print('    def %s():' % pname, file=fout)
            print('        r"""%s\n        """' % simplify_docstring(p.__doc__), file=fout)
            print('        pass\n', file=fout)


    if needfuncs:
        funcs = inspect.getmembers(sys.modules[name], inspect.isroutine)

        if len(funcs) > 0:
            print('\n# Functions\n', file=fout)

        for fname, f in funcs:
            if fname[0:2] != '__':
                print('def %s():' % fname, file=fout)
                print('    r"""%s\n    """' % simplify_docstring(f.__doc__), file=fout)
                print('    pass\n', file=fout)


    print('', file=fout)
    fout.close()

pathlib.Path(mock_name).mkdir(parents=True, exist_ok=True)
genmodule('mirheo', os.path.join(mock_name, '__init__.py'), False)


submodules = inspect.getmembers(sys.modules['mirheo'],
                                lambda member: inspect.ismodule(member)
                                and 'mirheo' in member.__name__ )


for mname, m in submodules:
    subpath = os.path.join(mock_name, mname)
    dst_init_path = os.path.join(subpath, '__init__.py')
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    genmodule('libmirheo.' + mname, dst_init_path, needfuncs=True)


# copy the python module files

src_init_path = mirheo.__file__
src_module_path = os.path.dirname(src_init_path)

module_files = ['tools.py', 'version.py']

for mf in module_files:
    shutil.copyfile(os.path.join(src_module_path, mf),
                    os.path.join(mock_name, mf))


with open(os.path.join(mock_name, '__init__.py'), 'a') as fout:
    flist = ", ".join([f'"{m.split(".")[0]}"' for m in module_files])
    print("", file=fout)
    print(f"__all__ = [{flist}]", file=fout)
    print("from .version import mir_version", file=fout)
