#!/usr/bin/env python

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
parser.add_argument('--save-to', type=str, required=True)
parser.add_argument('--load-from', type=str)
args = parser.parse_args()

if not args.load_from:
    u = mir.Mirheo(args.ranks, domain=(4, 6, 8), dt=0.1, debug_level=3, log_filename='log', no_splash=True)

    mesh = mir.ParticleVectors.MembraneMesh('mesh_dummy1.off')
    ov = mir.ParticleVectors.MembraneVector('ov', mesh=mesh, mass=1)
    pv = ov
    ic = mir.InitialConditions.Membrane([])
    u.registerParticleVector(ov, ic)

    wall = mir.Walls.Plane('plane', (0, -1, 0), (1.0, 2.0, 3.0))
    u.registerWall(wall, check_every=123)

    u.registerPlugins(mir.Plugins.createStats('stats', every=10, filename='stats.txt'))
    u.registerPlugins(mir.Plugins.createDumpMesh('rbcs', ov, dump_every=15, path='ply'))
    u.registerPlugins(mir.Plugins.createForceSaver('force_saver', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('dump_particles', pv, 20, ['forces'], 'h5/pv-'))

    u.registerPlugins(mir.Plugins.createBerendsenThermostat('berendsen_thermostat', [pv], kBT=123, tau=10.0))

    # Stores extraForce.dat. We do not check the content of the file, only that it is correctly reloaded.
    forces = [[0.01 * k, 0.02 * k, 0.03 * k] for k in range(6)]
    u.registerPlugins(mir.Plugins.createMembraneExtraForce('extraForce', ov, forces))
    u.registerPlugins(mir.Plugins.createWallRepulsion(
                "wallRepulsion", ov, wall, C=75, h=0.125, max_force=750))

    u.saveSnapshot(args.save_to)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.load_from, debug_level=3, log_filename='log', no_splash=True)
    u.saveSnapshot(args.save_to)

# TEST: snapshot.plugins
# cd snapshot
# rm -rf snapshot1/ snapshot2/
# mir.run --runargs "-n 4" ./plugins.py --ranks 2 1 1 --save-to snapshot1/
# mir.run --runargs "-n 4" ./plugins.py --ranks 2 1 1 --save-to snapshot2/ --load-from snapshot1/
# git --no-pager diff --no-index snapshot1/config.json snapshot2/config.json
# git --no-pager diff --no-index snapshot1/extraForce.dat snapshot2/extraForce.dat
# cp snapshot1/config.json snapshot.out.txt
