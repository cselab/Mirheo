#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes', dest='axes', type=float, nargs=3)
parser.add_argument('--coords', dest='coords', type=str)
parser.add_argument('--withMesh', action='store_true')
parser.add_argument('--omega', type=float, default=0)
parser.add_argument('--phi', type=float, default=0)
parser.set_defaults(withMesh=False)
args = parser.parse_args()

dt   = 0.01
axes = tuple(args.axes)

ranks  = (1, 1, 1)
domain = (16, 8, 8)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.withMesh:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= axes[i]
    mesh = ymr.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes, mesh=mesh)
else:
    pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)

icEllipsoid = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoid = ymr.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)

ovStats = ymr.Plugins.createDumpObjectStats("objStats", ov=pvEllipsoid, dump_every=500, path="stats")
u.registerPlugins(ovStats)

M = (0.1, 0., 0.)

def magneticField(t):
    magn = 0.1
    arg = args.phi + args.omega * t
    return (magn * np.cos(arg), magn * np.sin(arg), 0.)


magneticPlugin = ymr.Plugins.createMagneticOrientation("externalB", pvEllipsoid, moment=M, magneticFunction=magneticField)
u.registerPlugins(magneticPlugin)

if args.withMesh:
    mdump = ymr.Plugins.createDumpMesh("mesh_dump", pvEllipsoid, 1000, path="ply/")
    u.registerPlugins(mdump)

u.run(10000)


# nTEST: rigids.magneticOrientation.Static
# set -eu
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# common_args="--axes 2.0 1.0 1.0"
# ymr.run --runargs "-n 2"  ./createEllipsoid.py $common_args --density 8 --out $f --niter 1000 > /dev/null
# ymr.run --runargs "-n 2" ./magneticOrientation.py $common_args --coords $f --phi 0.7853981634 > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $10, $3}' > rigid.out.txt

# nTEST: rigids.magneticOrientation.Time
# set -eu
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# common_args="--axes 2.0 1.0 1.0"
# ymr.run --runargs "-n 2"  ./createEllipsoid.py $common_args --density 8 --out $f --niter 1000       > /dev/null
# ymr.run --runargs "-n 2" ./magneticOrientation.py $common_args --coords $f --omega 0.005 > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $10, $3}' > rigid.out.txt
