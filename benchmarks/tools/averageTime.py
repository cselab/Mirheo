#!/usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='stats file', type=str)
args = parser.parse_args()

data = np.loadtxt(args.file)
ms   = data[1:, 6]

print(np.sum(ms) / len(ms))
