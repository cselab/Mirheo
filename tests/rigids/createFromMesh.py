#!/usr/bin/env python

import numpy as np

def createFromMesh(density, vertices, triangles, inertia, niter):

    import ymero as ymr

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
    
    u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')
    
    dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.5, power=0.5)
    vv = ymr.Integrators.VelocityVerlet('vv')
    
    coords = [bb_lo, bb_hi]
    com_q  = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

    mesh = ymr.ParticleVectors.Mesh(vertices, triangles)

    fakeOV = ymr.ParticleVectors.RigidObjectVector('OV', mass=1, inertia=inertia, object_size=len(coords), mesh=mesh)
    fakeIc = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
    belongingChecker = ymr.BelongingCheckers.Mesh("meshChecker")
    
    pvMesh = u.makeFrozenRigidParticles(belongingChecker, fakeOV, fakeIc, [dpd], vv, density, niter)

    if pvMesh:
        frozenCoords = pvMesh.getCoordinates()
        frozenCoords = recenter(frozenCoords, com_q[0])
    else:
        frozenCoords = [[]]

    if u.isMasterTask():
        return frozenCoords
    else:
        return None


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
    if coords is not None:
        np.savetxt(args.out, coords)
    


# TEST: rigids.createFromMesh
# set -eu
# cd rigids
# cp ../../data/rbc_mesh.off .
# pfile="pos.txt"
# ymr.run --runargs "-n 2" ./createFromMesh.py --density 8 --fname rbc_mesh.off --niter 0 --out $pfile > /dev/null
# cat $pfile | LC_ALL=en_US.utf8 sort > pos.out.txt 
