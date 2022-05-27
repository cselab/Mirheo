#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

ranks  = (1, 1, 1)
domain = [16, 16, 16]

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

com = [[ 1., 0., 0.],
       [ 5., 0., 0.],
       [-9., 0., 0.], # out of the domain
       [ 0., 7., 0.]]

orientations = [[ 1., 0., 0.],
                [ 0., 1., 0.],
                [-1., 0., 0.], # out of the domain
                [ 0., 0., 1.]]

num_parts_per_chain = 4

cv = mir.ParticleVectors.ChainVector('chains', mass=1, chain_length=num_parts_per_chain)
ic = mir.InitialConditions.StraightChains(positions=com, orientations=orientations, length=1.0)
u.registerParticleVector(cv, ic)

dump_every = 1
u.registerPlugins(mir.Plugins.createDumpParticles('parts_dump', cv, dump_every, [], 'h5/chain_particles-'))

u.run(2, dt=0)

if cv:
    icpos = np.array(cv.getCoordinates())

    num_chains = len(icpos) // num_parts_per_chain

    nv = num_parts_per_chain
    assert nv * num_chains == len(icpos)

    # sort by center of mass x coordinates
    com = np.array([np.mean(icpos[i*nv:(i+1)*nv,:], axis=0) for i in range(num_chains)])

    order = np.argsort(com[:,0])
    idx = np.repeat(order, nv) * nv + np.tile(np.arange(nv), num_chains)
    icpos = icpos[idx,:]
    icvel = icpos[idx,:]

    np.savetxt("pos.ic.txt", icpos)

del u

# nTEST: ic.chain
# cd ic
# rm -rf pos*.txt
# mir.run --runargs "-n 2" ./chain.py
# cat pos.ic.txt | uscale 100 > ic.out.txt
