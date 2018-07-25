#! /usr/bin/env python3

import sphinx.cmd.build
import sphinx.ext.autodoc
import glob
import sys

rst = []
def add_line(self, line, source, *lineno):
    """Append one line of generated reST to the output."""
    rst.append(line)
    self.directive.result.append(self.indent + line, source, *lineno)
sphinx.ext.autodoc.Documenter.add_line = add_line

sys.path.append('../../src')
sys.path.append('../../build')

sphinx.cmd.build.main(sys.argv[1:])


try:
    fname = sys.argv[3]
    with open(fname + '.tmp', 'w') as f:
        for line in rst:
            print(line, file=f)
except:
    pass
