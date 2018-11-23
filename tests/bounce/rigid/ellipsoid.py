#!/usr/bin/env python

import numpy as np
import udevicex as ymr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes', type=float, nargs=3)
parser.add_argument('--coords', type=str)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [8., 8., 8.]

dt   = 0.001

u = ymr.udevicex(ranks, tuple(domain), debug_level=3, log_filename='log')

nparts = 100
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pvSolvent = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
icSolvent = ymr.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vvSolvent = ymr.Integrators.VelocityVerlet('vv', dt=dt)
u.registerParticleVector(pvSolvent, icSolvent)
u.registerIntegrator(vvSolvent)
u.setIntegrator(vvSolvent, pvSolvent)


com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.vis:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= args.axes[i]
    mesh = ymr.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes, mesh=mesh)
else:
    pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes)

icEllipsoid = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoid = ymr.Integrators.RigidVelocityVerlet("ellvv", dt)
u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)


bb = ymr.Bouncers.Ellipsoid("bounceEllipsoid")
u.registerBouncer(bb)
u.setBouncer(bb, pvEllipsoid, pvSolvent)


dumpEvery=500

if args.vis:
    solventDump = ymr.Plugins.createDumpParticles('partDump', pvSolvent, dumpEvery, [], 'h5/solvent-')
    u.registerPlugins(solventDump)


    mdump = ymr.Plugins.createDumpMesh("mesh_dump", pvEllipsoid, dumpEvery, path="ply/")
    u.registerPlugins(mdump)


rigStats = ymr.Plugins.createDumpObjectStats("rigStats", ov=pvEllipsoid, dump_every=dumpEvery, path="stats")
u.registerPlugins(rigStats)

u.run(5000)
    

# nTEST: bounce.rigid.ellipsoid
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="pos.txt"
# common_args="--axes 1.0 2.0 1.0"
# rm -rf pos*.txt vel*.txt
# ymr.run --runargs "-n 2"  ../../rigids/createEllipsoid.py $common_args --density 8 --out $f --niter 1000 > /dev/null
# ymr.run --runargs "-n 2" ./ellipsoid.py $common_args --coords $f                       > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $15, $9}' > rigid.out.txt
