#!/usr/bin/env python

def create_cylinder(density, R, L, niter):
    import mirheo as mir
    
    def recenter(coords, com):
        coords = [[r[0]-com[0], r[1]-com[1], r[2]-com[2]] for r in coords]
        return coords

    dt = 0.001

    ranks  = (1, 1, 1)
    fact = 3
    domain = (fact*R, fact*R, fact*L/2)
    
    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)
    
    dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.5, power=0.5)
    vv = mir.Integrators.VelocityVerlet('vv')
    
    coords = [[-R, -R, -L/2],
              [ R,  R,  L/2]]
    com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
    
    fake_oV = mir.ParticleVectors.RigidCylinderVector('OV', mass=1, object_size=len(coords), radius=R, length=L)
    fake_ic = mir.InitialConditions.Rigid(com_q, coords)
    belonging_checker = mir.BelongingCheckers.Cylinder("checker")
    
    pv_cyl = u.makeFrozenRigidParticles(belonging_checker, fake_oV, fake_ic, [dpd], vv, density, 1.0, dt=dt, nsteps=niter)
    
    if pv_cyl:
        frozen_coords = pv_cyl.getCoordinates()
        frozen_coords = recenter(frozen_coords, com_q[0])
    else:
        frozen_coords = [[]]

    if u.isMasterTask():
        return frozen_coords
    else:
        return None

if __name__ == '__main__':

    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', type=float)
    parser.add_argument('--R', type=float)
    parser.add_argument('--L', type=float)
    parser.add_argument('--niter', type=int)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    coords = create_cylinder(args.density, args.R, args.L, args.niter)

    if coords is not None:
        np.savetxt(args.out, coords)
    
# TEST: rigids.create_cylinder
# set -eu
# cd rigids
# rm -rf pos.txt pos.out.txt
# pfile=pos.txt
# mir.run --runargs "-n 2"  ./create_cylinder.py --R 3.0 --L 5.0 --density 8 --niter 0 --out $pfile
# cat $pfile | LC_ALL=en_US.utf8 sort > pos.out.txt

