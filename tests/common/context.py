#! /usr/bin/env python3

import sys
import os

__quick_inst_rel_path = "../../build"
__quick_py_rel_path = "../../src/py"
__dir_path = os.path.dirname(__file__)
__pyudx_path = os.path.join(__dir_path, __quick_py_rel_path)
__soudx_path = os.path.join(__dir_path, __quick_inst_rel_path)
sys.path.insert(0, os.path.abspath(__pyudx_path))
sys.path.insert(0, os.path.abspath(__soudx_path))

import udevicex
