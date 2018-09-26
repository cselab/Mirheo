#!/usr/bin/env python

import sys
import numpy as np

import udevicex as udx

import sys, argparse
sys.path.append("../..")
from common.membrane_params import set_lina

parser = argparse.ArgumentParser()
parser.add_argument('--subStep', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

substeps = 10
if args.subStep:
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

nparts = 1000
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pvSolvent = udx.ParticleVectors.ParticleVector('pv', mass = 1)
icSolvent = udx.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv        = udx.Integrators.VelocityVerlet('vv', dt=dt)
u.registerParticleVector(pvSolvent, icSolvent)
u.registerIntegrator(vv)
u.setIntegrator(vv, pvSolvent)


meshRbc = udx.ParticleVectors.MembraneMesh("rbc_mesh.off")
pvRbc   = udx.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=meshRbc)
icRbc   = udx.InitialConditions.Membrane(
    [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0.0, 0.7071, 0.0]]
)

u.registerParticleVector(pvRbc, icRbc)

prmRbc = udx.Interactions.MembraneParameters()

if prmRbc:
    set_lina(1.0, prmRbc)
    prmRbc.dt = dt
    
intRbc = udx.Interactions.MembraneForces("int_rbc", prmRbc, stressFree=True)

if args.subStep:
    integrator = udx.Integrators.SubStepMembrane('substep_membrane', dt, substeps, intRbc)
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pvRbc)
else:
    u.setIntegrator(vv, pvRbc)
    u.registerInteraction(intRbc)
    u.setInteraction(intRbc, pvRbc, pvRbc)


bb = udx.Bouncers.Mesh("bounceRbc", kbt=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pvRbc, pvSolvent)


if args.vis:
    dumpEvery = int(0.1 / dt)
    
    solventDump = udx.Plugins.createDumpParticles('partDump', pvSolvent, dumpEvery, [], 'h5/solvent-')
    u.registerPlugins(solventDump)

    mdump = udx.Plugins.createDumpMesh("mesh_dump", pvRbc, dumpEvery, path="ply/")
    u.registerPlugins(mdump)


tend = int(5.0 / dt)
    
u.run(tend)

if pvRbc is not None:
    rbc_pos = pvRbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: bounce.membrane.mesh
# set -eu
# cd bounce/membrane
# rm -rf pos.rbc.txt pos.rbc.out.txt 
# cp ../../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./mesh.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: bounce.membrane.mesh.substep
# set -eu
# cd bounce/membrane
# rm -rf pos.rbc.txt pos.rbc.out.txt 
# cp ../../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./mesh.py --subStep > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
