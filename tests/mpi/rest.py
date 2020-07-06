#!/usr/bin/env python

import mirheo as mir
from mpi4py import MPI

def run(niter, statsFname, comm_address):
    dt = 0.001

    ranks  = (2, 1, 1)
    domain = (12, 8, 10)

    u = mir.Mirheo(ranks, domain, dt, debug_level=8, log_filename='log', comm_ptr=comm_address)

    pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
    ic = mir.InitialConditions.Uniform(number_density=2)
    u.registerParticleVector(pv=pv, ic=ic)

    dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    u.registerPlugins(mir.Plugins.createStats('stats', statsFname, 1000))

    u.run(niter)

comm = MPI.COMM_WORLD
run(2002, "stats1", MPI._addressof(comm))
run(2002, "stats2", MPI._addressof(comm))


# nTEST: mpi.rest.consecutive
# cd mpi
# rm -rf stats*.csv
# mir.run --runargs "-n 4" ./rest.py > /dev/null
# mir.post ../tools/dump_csv.py stats1.csv time kBT vx vy vz --header > stats.out.txt
# mir.post ../tools/dump_csv.py stats2.csv time kBT vx vy vz         >> stats.out.txt
