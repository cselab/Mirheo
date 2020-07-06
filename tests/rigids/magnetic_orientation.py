#!/usr/bin/env python

import mirheo as mir
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

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.withMesh:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= axes[i]
    mesh = mir.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes, mesh=mesh)
else:
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)

ic_ell = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_ell, ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov=pv_ell, dump_every=500, path="stats"))

M = (0.1, 0., 0.)

def magneticField(t):
    magn = 0.1
    arg = args.phi + args.omega * t
    return (magn * np.cos(arg), magn * np.sin(arg), 0.)


u.registerPlugins(mir.Plugins.createMagneticOrientation("externalB", pv_ell, moment=M, magneticFunction=magneticField))

if args.withMesh:
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_ell, 1000, path="ply/"))

u.run(10000)

del(u)


# nTEST: rigids.magnetic_orientation.static
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./magnetic_orientation.py --axes $ax $ay $az --coords $f --phi 0.7853981634
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time vx comx > rigid.out.txt

# nTEST: rigids.magnetic_orientation.time
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./magnetic_orientation.py --axes $ax $ay $az --coords $f --omega 0.005
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time vx comx > rigid.out.txt
