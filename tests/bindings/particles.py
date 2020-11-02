#!/usr/bin/env python

import numpy as np
import mirheo as mir

def main():
    ranks  = (1, 1, 1)
    domain = [16, 16, 16]

    dt    = 1e-3
    t_end = 5.0
    mass = 1.0
    k_bound = 1.0

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)


    pv1 = mir.ParticleVectors.ParticleVector("pv1", mass)
    pv2 = mir.ParticleVectors.ParticleVector("pv2", mass)

    n = 5

    pos1 = [[1 + i, 7.9, 8] for i in range(n)]
    pos2 = [[1 + i, 8.1, 8] for i in range(n)]

    vel1 = [[0,0,0] for i in range(n)]
    vel2 = [[0,0,0] for i in range(n)]

    pairs = [[int(i), int(i)] for i in range(n)]

    u.registerParticleVector(pv1, mir.InitialConditions.FromArray(pos1, vel1))
    u.registerParticleVector(pv2, mir.InitialConditions.FromArray(pos2, vel2))

    int_spring = mir.Interactions.ObjBinding("springs", k_bound=k_bound, pairs=pairs);
    u.registerInteraction(int_spring)
    u.setInteraction(int_spring, pv1, pv2)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv1)
    u.setIntegrator(vv, pv2)

    if False:
        t_dump_every = 0.5
        dump_every = int (t_dump_every/dt)
        u.registerPlugins(mir.Plugins.createDumpParticles('pv1_dump', pv1, dump_every, [], 'h5/pv1-'))
        u.registerPlugins(mir.Plugins.createDumpParticles('pv2_dump', pv2, dump_every, [], 'h5/pv2-'))

    u.run(int(t_end / dt), dt=dt)

    if pv1 is not None:
        pos1 = pv1.getCoordinates()
        pos2 = pv1.getCoordinates()
        np.savetxt("pos.txt", np.vstack((pos1, pos2)))

    del u

if __name__ == '__main__':
    main()

# nTEST: bindings.particles
# cd bindings
# rm -rf h5 pos*txt
# mir.run --runargs "-n 2" ./particles.py
# cat pos.txt > pos.out.txt
