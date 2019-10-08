#!/usr/bin/env python

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mass', type=float, default=1.0)
parser.add_argument('--EOS', choices=["Linear", "QuasiIncompressible"], required=True)
args = parser.parse_args()


dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', args.mass)
ic = mir.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0

density_kernel="WendlandC2"

EOS=args.EOS

if EOS == "Linear":
    EOS_params = {
        "sound_speed" : 10.0,
        "rho_0"       : 0.0
    }
elif EOS == "QuasiIncompressible":
    EOS_params = {
        "p0"    : 10.0,
        "rho_r" : 10.0
    }

den  = mir.Interactions.Pairwise('den' , rc, kind="Density", density_kernel=density_kernel)
sdpd = mir.Interactions.Pairwise('sdpd', rc, kind="SDPD", viscosity=10.0, kBT=1.0, EOS=EOS, density_kernel=density_kernel, **EOS_params)
u.registerInteraction(den)
u.registerInteraction(sdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(sdpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', "stats.txt", 1000))

u.run(5001)

# nTEST: sdpd.rest
# cd sdpd
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./rest.py --EOS Linear > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

# nTEST: sdpd.rest.QuasiIncompressible
# cd sdpd
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./rest.py --EOS QuasiIncompressible > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

# nTEST: sdpd.rest.mass
# cd sdpd
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./rest.py --EOS Linear --mass 5.0 > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

