#!/usr/bin/env python

import udevicex as udx
from mpi4py import MPI

def run(niter, statsFname, commAddress):
    dt = 0.001

    ranks  = (2, 1, 1)
    domain = (12, 8, 10)
    
    u = udx.udevicex(commAddress, ranks, domain, debug_level=8, log_filename='log')
    
    pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
    ic = udx.InitialConditions.Uniform(density=2)
    u.registerParticleVector(pv=pv, ic=ic)

    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    
    vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    
    stats = udx.Plugins.createStats('stats', statsFname, 1000)
    u.registerPlugins(stats)
    
    u.run(niter)

comm = MPI.COMM_WORLD
run(2002, "stats1.txt", MPI._addressof(comm))
run(2002, "stats2.txt", MPI._addressof(comm))


# nTEST: mpi.rest.consecutive
# cd mpi
# rm -rf stats*.txt
# udx.run --runargs "-n 4" ./rest.py > /dev/null
# cat stats1.txt | awk '{print $1, $2, $3, $4, $5}' >  stats.out.txt
# cat stats2.txt | awk '{print $1, $2, $3, $4, $5}' >> stats.out.txt

