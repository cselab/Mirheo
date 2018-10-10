import sys, os, subprocess, glob
import sphinx.ext.autodoc

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

extensions = ['breathe', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx_automodapi.automodapi', 'sphinx.ext.napoleon']
if on_rtd:
    #extensions += ['exhale']
    pass

add_module_names = False

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
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes",]
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "uDeviceX"


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


# Setup breathe and exhale
breathe_projects = { 'udevicex': '../xml' }
breathe_default_project = 'udevicex'
breathe_domain_by_extension = { "h" : "cpp", "cu" : "cpp" }

cpp_id_attributes = ['__device__', '__global__', '__host__']
cpp_paren_attributes = ['__launch_bounds__', '__align__']

primary_domain = 'py'


exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Library API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    r'''   PROJECT_NAME      = "udevicex"
                                    XML_OUTPUT        = xml

                                    INPUT             = ../../src/core ../../src/plugins

                                    EXCLUDE           = 

                                    ENABLE_PREPROCESSING   = YES
                                    MACRO_EXPANSION        = YES
                                    EXPAND_ONLY_PREDEF     = NO
                                    SKIP_FUNCTION_MACROS   = NO
                                    PREDEFINED             += __align__(x)=            \
                                                            __restrict__=            \
                                                            __launch_bounds__(x,y)=  \
                                                            __H__                    \
                                                            __HD__                   \
                                                            __D__
                                                        
                                    XML_PROGRAMLISTING     = YES
                                    GENERATE_LATEX         = NO
                                    GENERATE_MAN           = NO
                                    GENERATE_RTF           = NO
                                    GENERATE_HTML          = NO
                                    GENERATE_XML           = YES
                                    RECURSIVE              = YES
                                    QUIET                  = YES
                                    WARN_IF_UNDOCUMENTED   = NO
                                    EXTRACT_PRIVATE        = YES
                                    EXTRACT_STATIC         = YES
                                    EXTRACT_ALL            = YES

                                    ALIASES += "rst=\verbatim embed:rst:leading-asterisk"
                                    ALIASES += "endrst=\endverbatim"


                                    FILE_PATTERNS          = *.h \
                                                            *.cpp \
                                                            *.cu

                                    EXTENSION_MAPPING      = cu=C++'''
}


# Override some shit
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

def setup(app):
    app.add_stylesheet('css/theme.css')
    
    sys.path.insert(0, os.path.abspath('./'))
    sys.path.insert(0, os.path.abspath('./source'))
    sphinx.ext.autodoc.Documenter.format_signature = format_signature

