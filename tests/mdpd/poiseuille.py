#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
vtarget = (1.0, 0, 0)

density = 10

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

rc = 1.0
rd = 0.75

plate_lo = mir.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))
plate_hi = mir.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)

den  = mir.Interactions.Pairwise('density', rd, kind="Density", density_kernel="MDPD")
mdpd = mir.Interactions.Pairwise('mdpd', rc, kind="MDPD", rd=rd, a=10.0, b=10.0, gamma=50.0, kbt=0.1, power=0.25)

vv = mir.Integrators.VelocityVerlet("vv")
frozen = u.makeFrozenWallParticles(pvName="frozen", walls=[plate_lo, plate_hi], interactions=[den, mdpd], integrator=vv, density=density, nsteps=1000)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

u.registerInteraction(den)
u.registerInteraction(mdpd)

for p in [pv, frozen]:
    u.setInteraction(den, p, pv)
    u.setInteraction(mdpd, p, pv)
u.setInteraction(den, frozen, frozen)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)


grid_sample_every = 2
grid_dump_every   = 1000
grid_bin_size     = (1., 1., 0.5)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], grid_sample_every, grid_dump_every, grid_bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))


factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

vc_sample_every = 5
vc_tune_every = 5
vc_dump_every = 500

u.registerPlugins(mir.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, vc_sample_every, vc_tune_every, vc_dump_every, vtarget, Kp, Ki, Kd))

u.run(20002)

# nTEST: mdpd.poiseuille
# cd mdpd
# rm -rf h5
# mir.run --runargs "-n 2" ./poiseuille.py
# mir.avgh5 xy velocity h5/solvent-0001[5-9].h5 | awk '{print $1}' > profile.out.txt
