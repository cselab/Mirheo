#!/usr/bin/env python

import argparse
import numpy as np
import udevicex as udx

ranks  = (1, 1, 1)
domain = (16, 16, 8)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

center=(domain[0]*0.5, domain[1]*0.5)
wall = udx.Walls.Cylinder("cylinder", center=center, radius=domain[1]*0.4, axis="z", inside=True)

u.registerWall(wall, 1000)

volume = u.computeVolumeInsideWalls([wall], 100000)

np.savetxt("volume.txt", [volume]);

# nTEST: walls.volume.cylinder
# cd walls/volume
# rm -rf volume*txt
# udx.run --runargs "-n 1" ./cylinder.py > /dev/null
# cp volume.txt volume.out.txt
