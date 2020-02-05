import sys, os, subprocess, glob
import sphinx.ext.autodoc
import subprocess

# compile the xml source
subprocess.run('(cd .. && doxygen)', shell=True)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

extensions = ['breathe', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx_automodapi.automodapi', 'sphinx.ext.napoleon']

add_module_names = False

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Mirheo'
copyright = 'ETH Zurich'
author = 'Dmitry Alexeev, Lucas Amoudruz'

exclude_patterns = []
pygments_style = 'sphinx'
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes",]
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "Mirheo"


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


# Setup breathe
breathe_projects = { 'mirheo': '../xml' }
breathe_default_project = 'mirheo'
breathe_domain_by_extension = { "h" : "cpp", "cu" : "cpp" }

cpp_id_attributes = ['__device__', '__global__', '__host__']
cpp_paren_attributes = ['__launch_bounds__', '__align__']

primary_domain = 'py'


def format_signature(self):
    if self.args is not None:
        # signature given explicitly
        args = "(%s)" % self.args  # type: unicode
    else:
        # try to introspect the signature
        try:
            args = self.format_args()
        except Exception as err:
            #logger.warning(__('error while formatting arguments for %s: %s') %
            #                (self.fullname, err))
            args = None

    retann = self.retann

    result = self.env.app.emit_firstresult(
        'autodoc-process-signature', self.objtype, self.fullname,
        self.object, self.options, args, retann)
    if result:
        args, retann = result

    if args is not None:
        return args + (retann and (' -> %s' % retann) or '')
    else:
        return ''


# AUTOautosummary
# https://stackoverflow.com/questions/20569011/python-sphinx-autosummary-automated-listing-of-member-functions
from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from docutils.parsers.rst import directives
from sphinx.util.inspect import safe_getattr

class AutoAutoSummary(Autosummary):

    option_spec = {
        'methods': directives.unchanged,
        'attributes': directives.unchanged
    }

    required_arguments = 1

    @staticmethod
    def get_members(obj, typ, include_public=None):
        if not include_public:
            include_public = []
        items = []
        for name in dir(obj):
            try:
                #documenter = get_documenter(safe_getattr(obj, name), obj)
                documenter = get_documenter(self.app, safe_getattr(obj, name), obj)
            except AttributeError:
                continue
            if documenter.objtype == typ:
                items.append(name)
        public = [x for x in items if x in include_public or not x.startswith('_')]
        return public, items

    def run(self):
        clazz = str(self.arguments[0])
        try:
            (module_name, class_name) = clazz.rsplit('.', 1)
            m = __import__(module_name, globals(), locals(), [class_name])
            c = getattr(m, class_name)
            if 'methods' in self.options:
                _, methods = self.get_members(c, 'method', ['__init__'])

                self.content = ["~%s.%s" % (clazz, method) for method in methods if method is '__init__' or not method.startswith('_')]
            if 'attributes' in self.options:
                _, attribs = self.get_members(c, 'attribute')
                self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
        finally:
            return super(AutoAutoSummary, self).run()

######################################################################################################################

def setup(app):
    app.add_stylesheet('css/theme.css')
    
    sys.path.insert(0, os.path.abspath('./'))
    sys.path.insert(0, os.path.abspath('./source'))
    sphinx.ext.autodoc.Documenter.format_signature = format_signature

    app.add_directive('autoautosummary', AutoAutoSummary)

