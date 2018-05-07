import sys, os, subprocess

extensions = ['breathe', 'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'uDeviceX'
copyright = 'ETH Zurich'
author = 'Dmitry Alexeev'

exclude_patterns = []
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes",]
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "uDevicex"


# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True


breathe_projects = { 'uDeviceX': '../xml' }
breathe_default_project = 'uDeviceX'
breathe_domain_by_extension = { "h" : "cpp", "cu" : "cpp" }

cpp_id_attributes = ['__device__', '__global__', '__host__']
cpp_paren_attributes = ['__launch_bounds__', '__align__']

