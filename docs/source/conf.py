#!/usr/bin/env python

import os
import subprocess

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    subprocess.call('cd ..; doxygen', shell=True)

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

extensions = ['breathe']
breathe_projects = { 'udevicex': '../xml' }
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'udevicex'
copyright = 'ETH Zurich'
author = ''

#html_logo = 'quantstack-white.svg'

exclude_patterns = []
highlight_language = 'cuda'
cpp_id_attributes = ['__device__', '__global__', '__inline__', '__host__', 'static']
cpp_paren_attributes = ['__launch_bounds__', '__align__']
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'uDeviceX'
