#!/usr/bin/env python

def createEllipsoid(density, axes, niter):
    import ymero as ymr
    
    def recenter(coords, com):
        coords = [[r[0]-com[0], r[1]-com[1], r[2]-com[2]] for r in coords]
        return coords

    dt = 0.001
    axes = tuple(axes)

    ranks  = (1, 1, 1)
    fact = 3
    domain = (fact*axes[0], fact*axes[1], fact*axes[2])
    
    u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')
    
    dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.5, power=0.5)
    vv = ymr.Integrators.VelocityVerlet('vv')
    
    coords = [[-axes[0], -axes[1], -axes[2]],
              [ axes[0],  axes[1],  axes[2]]]
    com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
    
    fakeOV = ymr.ParticleVectors.RigidEllipsoidVector('OV', mass=1, object_size=len(coords), semi_axes=axes)
    fakeIc = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
    belongingChecker = ymr.BelongingCheckers.Ellipsoid("ellipsoidChecker")
    
    pvEllipsoid = u.makeFrozenRigidParticles(belongingChecker, fakeOV, fakeIc, dpd, vv, density, niter)
    
    if pvEllipsoid:
        frozenCoords = pvEllipsoid.getCoordinates()
        frozenCoords = recenter(frozenCoords, com_q[0])
    else:
        frozenCoords = [[]]

    if u.isMasterTask():
        return frozenCoords
    else:
        return None

if __name__ == '__main__':

    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', dest='density', type=float)
    parser.add_argument('--axes', dest='axes', type=float, nargs=3)
    parser.add_argument('--niter', dest='niter', type=int)
    parser.add_argument('--out', dest='out', type=str)
    args = parser.parse_args()

    coords = createEllipsoid(args.density, args.axes, args.niter)

    if coords is not None:
        np.savetxt(args.out, coords)
    
# nTEST: rigids.createEllipsoid
# set -eu
# cd rigids
# rm -rf pos.txt pos.out.txt
# pfile=pos.txt
# ymr.run --runargs "-n 2"  ./createEllipsoid.py --axes 2.0 3.0 4.0 --density 8 --niter 1 --out $pfile > /dev/null
# cat $pfile | LC_ALL=en_US.utf8 sort > pos.out.txt

