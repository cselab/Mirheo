#!/usr/bin/env python

import numpy as np

def createFromMesh(density, vertices, triangles, inertia, niter):

    import udevicex as udx

    def recenter(coords, com):
        coords = [[r[0]-com[0], r[1]-com[1], r[2]-com[2]] for r in coords]
        return coords

    dt = 0.001

    # bounding box
    bb_hi = np.array(vertices).max(axis=0).tolist()
    bb_lo = np.array(vertices).min(axis=0).tolist()

    mesh_size = [hi-lo for lo, hi in zip(bb_lo, bb_hi)]

    fact = 2.0
    domain = (float (int (fact * mesh_size[0] + 1) ),
              float (int (fact * mesh_size[1] + 1) ),
              float (int (fact * mesh_size[2] + 1) ))

    ranks  = (1, 1, 1)
    
    u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')
    
    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.5, dt=dt, power=0.5)
    vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
    
    coords = [bb_lo, bb_hi]
    com_q  = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

    mesh = udx.ParticleVectors.Mesh(vertices, triangles)

    fakeOV = udx.ParticleVectors.RigidObjectVector('OV', mass=1, inertia=inertia, object_size=len(coords), mesh=mesh)
    fakeIc = udx.InitialConditions.Rigid(com_q=com_q, coords=coords)
    belongingChecker = udx.BelongingCheckers.Mesh("meshChecker")
    
    pvMesh = u.makeFrozenRigidParticles(belongingChecker, fakeOV, fakeIc, dpd, vv, density, niter)

    if pvMesh:
        frozenCoords = pvMesh.getCoordinates()
        frozenCoords = recenter(frozenCoords, com_q[0])
    else:
        frozenCoords = [[]]

    return frozenCoords



def createFromMeshFile(density, fname, niter):
    import trimesh
    m = trimesh.load(fname);
    # TODO diagonalize
    inertia = [row[i] for i, row in enumerate(m.moment_inertia)]
    return createFromMesh(density, m.vertices.tolist(), m.faces.tolist(), inertia, niter)

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', dest='density', type=float)
    parser.add_argument('--fname', dest='fname', type=str)
    parser.add_argument('--niter', dest='niter', type=int)
    parser.add_argument('--out', dest='out', type=str)
    args = parser.parse_args()

    coords = createFromMeshFile(args.density, args.fname, args.niter)

    # assume only one rank is working
    np.savetxt(args.out, coords)
    


# nTEST: rigids.createFromMesh
# set -eu
# cd rigids
# cp ../../data/rbc_mesh.off .
# pfile="pos.txt"
# udx.run --runargs "-n 2" ./createFromMesh.py --density 8 --fname rbc_mesh.off --niter 1 --out $pfile > /dev/null
# cat $pfile | sort > pos.out.txt 
