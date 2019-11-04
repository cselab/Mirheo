#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--density', type=float)
parser.add_argument('--axes', type=float, nargs=3)
parser.add_argument('--coords', type=str)
parser.add_argument('--bounce_back', action='store_true', default=False)
parser.add_argument('--xpos', type=float, default=0.5)
args = parser.parse_args()

dt       = 0.001
rc       = 1.0
axes     = tuple(args.axes)
dp_force = 0.5
density  = args.density

ranks  = (1, 1, 1)
domain = (16, 8, 8)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv_sol = mir.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_sol = mir.InitialConditions.Uniform(density)

density_kernel="WendlandC2"

den  = mir.Interactions.Pairwise('density', rc, kind="Density", density_kernel=density_kernel)
sdpd = mir.Interactions.Pairwise('sdpd', rc, kind="SDPD", viscosity=10.0, kBT=0.01, EOS="Linear", density_kernel=density_kernel, sound_speed=10.0, rho_0=0.0)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=dp_force, direction="x")

com_q = [[args.xpos * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

coords = np.loadtxt(args.coords).tolist()

pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = mir.InitialConditions.Rigid(com_q, coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_sol, ic_sol)
u.registerParticleVector(pv_ell, ic_ell)

u.registerIntegrator(vv)
u.registerIntegrator(vv_ell)

u.setIntegrator(vv,     pv_sol)
u.setIntegrator(vv_ell, pv_ell)

u.registerInteraction(den)
u.registerInteraction(sdpd)

u.setInteraction(den, pv_sol, pv_sol)
u.setInteraction(den, pv_ell, pv_sol)
u.setInteraction(den, pv_ell, pv_ell)

u.setInteraction(sdpd, pv_sol, pv_sol)
u.setInteraction(sdpd, pv_ell, pv_sol)

belonging_checker = mir.BelongingCheckers.Ellipsoid("ellipsoid_checker")

u.registerObjectBelongingChecker(belonging_checker, pv_ell)
u.applyObjectBelongingChecker(belonging_checker, pv=pv_sol, correct_every=0, inside="none", outside="")

if args.bounce_back:
    bb = mir.Bouncers.Ellipsoid("bounceEllipsoid", "bounce_back")
    u.registerBouncer(bb)
    u.setBouncer(bb, pv_ell, pv_sol)

dump_every=500
u.registerPlugins(mir.Plugins.createParticleChannelSaver("density_saver", pv_ell, "densities", "den"))
u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv_ell, dump_every, ["den"], 'h5/ell_particles-'))
u.registerPlugins(mir.Plugins.createParticleChannelSaver("density_saver_sol", pv_sol, "densities", "den"))
u.registerPlugins(mir.Plugins.createDumpParticles('partDump_sol', pv_sol, dump_every, ["den"], 'h5/sol_particles-'))

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov=pv_ell, dump_every=dump_every, path="stats"))

u.run(10000)


# nTEST: sdpd.fsi.ellipsoid
# set -eu
# cd sdpd
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./rigid.py --density $rho --axes $ax $ay $az --coords $f
# cat stats/ellipsoid.txt | awk '{print $2, $6, $7, $8, $9}' > rigid.out.txt

# nTEST: sdpd.fsi.ellipsoid.edge
# set -eu
# cd sdpd
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./rigid.py --density $rho --axes $ax $ay $az --coords $f --xpos 0.1
# cat stats/ellipsoid.txt | awk '{print $2, $6, $7, $8, $9}' > rigid.out.txt
