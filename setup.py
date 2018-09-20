#! /usr/bin/env python3

import os, glob
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

import sys
sys.path.insert(0, 'src/udevicex')
import version


class BinaryExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CopyLibrary(build_ext):
    def run(self):
        for ext in self.extensions:
            self.copy_extension(ext)

    def copy_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        library = glob.glob(ext.sourcedir + '/build/libudevicex.cpython*.so')

        if (len(library) == 0):
            raise ValueError('No pre-build library found in folder ' + 
                    ext.sourcedir + '/build/')

        shutil.copy2(library[0], extdir)


setup(
    name='uDeviceX',
    version=version.udx_version,
    author='Dmitry Alexeev',
    author_email='alexeedm@ethz.ch',
    description='The in-silico lab-on-a-chip',
    long_description='',
    packages = ['udevicex'],
    package_dir = {'udevicex' : 'src/udevicex'},
    ext_modules=[BinaryExtension('libudevicex', sourcedir='./')],
    cmdclass=dict(build_ext=CopyLibrary),
    zip_safe=False,
)
