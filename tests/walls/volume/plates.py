#!/usr/bin/env python

import argparse
import numpy as np
import ymero as ymr

parser = argparse.ArgumentParser()
parser.add_argument("--D", type = float, required = True)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (8, 16, 8)

u = ymr.ymero(ranks, domain, debug_level=3, log_filename='log')

plate_lo = ymr.Walls.Plane("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              args.D))
plate_hi = ymr.Walls.Plane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - args.D))

u.registerWall(plate_lo, 1000)
u.registerWall(plate_hi, 1000)

volume = u.computeVolumeInsideWalls([plate_lo, plate_hi], 100000)

np.savetxt("volume.txt", [volume]);

# nTEST: walls.volume.plates
# cd walls/volume
# rm -rf volume*txt
# ymr.run --runargs "-n 1" ./plates.py --D 1.0 > /dev/null
# cp volume.txt volume.out.txt
