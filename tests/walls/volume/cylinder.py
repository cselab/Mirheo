#!/usr/bin/env python

import argparse
import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = (16, 16, 8)

u = mir.Mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

center=(domain[0]*0.5, domain[1]*0.5)
wall = mir.Walls.Cylinder("cylinder", center=center, radius=domain[1]*0.4, axis="z", inside=True)

u.registerWall(wall, 1000)

volume = u.computeVolumeInsideWalls([wall], 100000)

np.savetxt("volume.txt", [volume]);

# nTEST: walls.volume.cylinder
# cd walls/volume
# rm -rf volume*txt
# mir.run --runargs "-n 1" ./cylinder.py
# cp volume.txt volume.out.txt
