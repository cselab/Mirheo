#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)
L = 8
R = 0.5
num_segments = 4

mass_sol = 0.1
mass_rod = 100.0

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

nparts = 1000
np.random.seed(42)
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 0.5, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pv_solvent = mir.ParticleVectors.ParticleVector('pv', mass_sol)
ic_solvent = mir.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv         = mir.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pv_solvent, ic_solvent)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_solvent)


def center_line(s): return (0, 0, (s-0.5) * L)
def torsion(s): return 0

l0 = L / num_segments

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1.0, 0.0, 0.0, 0.0]]

pv_rod = mir.ParticleVectors.RodVector('rod', mass_rod, num_segments)
ic_rod = mir.InitialConditions.Rod(com_q, center_line, torsion, 2*R)
u.registerParticleVector(pv_rod, ic_rod)

u.setIntegrator(vv, pv_rod)

bb = mir.Bouncers.Rod("bouncer", radius = R)
u.registerBouncer(bb)
u.setBouncer(bb, pv_rod, pv_solvent)

if args.vis:
    dump_every = int(0.1 / dt)
    u.registerPlugins(mir.Plugins.createDumpParticles('solvent_dump', pv_solvent, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpParticles('rod_dump', pv_rod, dump_every, [], 'h5/rod-'))
    u.registerPlugins(mir.Plugins.createStats('stats', "stats.txt", dump_every))
    
tend = int(5.0 / dt)
    
u.run(tend)

if pv_rod is not None:
    rod_pos = pv_rod.getCoordinates()
    np.savetxt("pos.rod.txt", rod_pos)


# nTEST: bounce.rod
# set -eu
# cd bounce/rod
# rm -rf pos.rod.txt pos.rod.out.txt 
# mir.run --runargs "-n 2" ./main.py
# mv pos.rod.txt pos.rod.out.txt 
