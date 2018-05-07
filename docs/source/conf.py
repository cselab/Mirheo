#!/usr/bin/env python

import os
import subprocess

html_static_path = ['_static/']

def setup(app):
    app.add_stylesheet('css/custom.css')

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    subprocess.call('cd ..; doxygen', shell=True)

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

extensions = ['breathe', 'sphinx.ext.mathjax']
breathe_projects = { 'uDeviceX': '../xml' }
breathe_default_project = 'uDeviceX'
breathe_domain_by_extension = { "h" : "cpp", "cu" : "cpp" }
        
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'uDeviceX'
copyright = 'ETH Zurich'
author = ''

exclude_patterns = []
highlight_language = 'cuda'
cpp_id_attributes = ['__device__', '__global__', '__host__']
cpp_paren_attributes = ['__launch_bounds__', '__align__']
#pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'uDeviceX'

