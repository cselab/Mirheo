#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 4

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=50.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)

plate_lo = mir.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))
plate_hi = mir.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)

vv = mir.Integrators.VelocityVerlet("vv")
frozen = u.makeFrozenWallParticles(pvName="plates", walls=[plate_lo, plate_hi], interactions=[dpd], integrator=vv, number_density=density, dt=dt)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)


for p in (pv, frozen):
    u.setInteraction(dpd, p, pv)

vv_dp = mir.Integrators.VelocityVerlet_withConstForce("vv_dp", force)
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv)


sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 0.5)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run(7002, dt=dt)

# nTEST: walls.analytic.plates
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./plates.py
# mir.avgh5 xy velocities h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
