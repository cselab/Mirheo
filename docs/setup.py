# This script will only be run on RTD server
#
# It substitutes the files with autodoc directives
# by the pre-generated docs

import os, glob

src = glob.glob('user/*.rst')
gen = glob.glob('user/*.rst.gen')


for s in src:
    os.rename(s, s+'.bak')
    
for g in gen:
    os.rename(g, g[:-4])
