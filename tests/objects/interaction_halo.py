#!/usr/bin/env python

import mirheo as mir
import numpy as np

if __name__ == '__main__':
    dt   = 0.005
    a = 1
    axes = (a, a, a)
    rc = 1

    ranks  = (1, 1, 1)
    domain = (16, 16, 16)

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

    com_q = [[domain[0] - a - rc/4, domain[1]/2, domain[2]/2, 1, 0, 0, 0],
             [0         + a + rc/4, domain[1]/2, domain[2]/2, 1, 0, 0, 0],

             [domain[0]/2 - a - rc/4, 0, domain[2]/2, 1, 0, 0, 0],
             [domain[0]/2 + a + rc/4, 0, domain[2]/2, 1, 0, 0, 0]]

    coords = [[-a, 0, 0],
              [+a, 0, 0],
              [0, -a, 0],
              [0, +a, 0],
              [0, 0, -a],
              [0, 0, +a]]

    pv = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
    ic = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
    vv = mir.Integrators.RigidVelocityVerlet("vv")
    dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=25, gamma=0, power=1, kBT=0)

    u.registerParticleVector(pv, ic)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    u.registerInteraction(dpd)
    u.setInteraction(dpd,  pv, pv)

    u.registerPlugins( mir.Plugins.createDumpObjectStats("objStats", pv, dump_every=100, filename="stats/ellipsoid.csv") )
    u.run(1000, dt=dt)


# nTEST: objects.halo.interactions
# set -eu
# cd objects
# f="pos.txt"
# rm -rf stats $f
# mir.run --runargs "-n 2" ./interaction_halo.py
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv objId time comx comy comz | LC_ALL=en_US.utf8 sort -g -k1 -k2 > pos.out.txt
