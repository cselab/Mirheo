#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default = [1,1,1])
args = parser.parse_args()


ranks  = args.ranks
domain = (8, 8, 16)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True,
               checkpoint_every=20, checkpoint_mode="Incremental")

axes = (1.0, 2.0, 3.0)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.33 * domain[2],   1., 0, 0, 0],
         [0.5 * domain[0], 0.5 * domain[1], 0.66 * domain[2],   1., 0, 0, 0]]

ov = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
u.registerParticleVector(ov, ic)

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov, dump_every=10, filename="stats/ellipsoid.csv"))

if args.restart:
    u.restart("restart/")

u.run(62, dt=0.01)

# TEST: restart.plugins.obj_stats
# cd restart
# rm -rf stats rov_stats.out.txt
# mir.run --runargs "-n 2" ./obj_stats.py
# mir.run --runargs "-n 2" ./obj_stats.py --restart
# cp stats/ellipsoid.csv rov_stats.out.txt
