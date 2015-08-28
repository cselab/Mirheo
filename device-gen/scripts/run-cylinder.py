#!/usr/bin/env python
'''
 *  Part of CTC/device-gen/scripts/run-cylinder.py
 *
 *  Created and authored by Kirill Lykov on 2015-08-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
'''

import os

N = 32
radius = 6.0
length = 16.0
os.system("../sdf-cylinder//sdf-unit %d %f unit.dat"%(N, radius))
outFile = "cylinder%d.dat"%(radius)
os.system("../2Dto3D/2Dto3D unit.dat %f %f %d %s"%(length, 0.0, 32, outFile))
