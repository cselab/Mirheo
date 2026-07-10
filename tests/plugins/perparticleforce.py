#!/usr/bin/env python

"""Test the AddPerParticleForce plugin and ParticleVector.additiveUpdateChannel.

First invocation: write a checkpoint of random particles, then add a random
'extraforces' dataset to it (this is how the thesis tensile-test scripts
prepared their restart files). Second invocation (--restart): restart from
those files, apply the channel as a force every step with dt=0 (so the saved
forces must equal the channel exactly), ramp the channel with
additiveUpdateChannel and check the forces follow.
"""

import copy
import glob
import re
import xml.etree.ElementTree as ET

import argparse
import h5py
import mirheo as mir
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
args = parser.parse_args()

ranks = (1, 1, 1)
domain = (10.0, 10.0, 10.0)
n = 1000

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True,
               checkpoint_every=(0 if args.restart else 5), checkpoint_folder='ppf_restart/')

pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)

if args.restart:
    ic = mir.InitialConditions.Restart('ppf_restart/')
else:
    np.random.seed(42)
    pos = np.random.rand(n, 3) * domain
    vel = np.zeros((n, 3))
    ic = mir.InitialConditions.FromArray(pos.tolist(), vel.tolist())

u.registerParticleVector(pv, ic)

if args.restart:
    u.registerPlugins(mir.Plugins.createAddPerParticleForce('extra_force', pv, 'extraforces'))

    # a no-op interaction so that the force buffer is cleared every step
    sw2 = mir.Interactions.Pairwise('sw', rc=1.0, kind='SW', epsilon=0.0, sigma=0.0, A=0.0, B=0.0)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    u.registerPlugins(mir.Plugins.createForceSaver('force_saver', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, 1,
                                                      ['forces', 'extraforces'], 'h5/pv-'))

u.run(7 if not args.restart else 5, dt=0.0)

if not args.restart and u.isComputeTask():
    # add the random 'extraforces' channel to the checkpoint just written
    xmf_path = 'ppf_restart/pv.PV.xmf'
    tree = ET.parse(xmf_path)
    grid = tree.getroot().find('.//Grid')
    h5_name = re.match(r'(.*\.h5):', grid.find('Geometry/DataItem').text).group(1)

    np.random.seed(7)
    extraforces = (np.random.rand(n, 3) - 0.5).astype(np.float32)
    with h5py.File('ppf_restart/' + h5_name, 'r+') as f:
        f.create_dataset('extraforces', data=extraforces)

    vel_attr = next(a for a in grid.findall('Attribute') if a.get('Name') == 'velocities')
    xf_attr = copy.deepcopy(vel_attr)
    xf_attr.set('Name', 'extraforces')
    xf_attr.find('DataItem').text = h5_name + ':/extraforces'
    grid.append(xf_attr)
    tree.write(xmf_path, xml_declaration=True)

def sort_rows(a):
    """Lexicographic row order, so arrays can be compared as row multisets.

    The dumped 'forces' channel is not persistent (ForceSaver uses
    PersistenceMode::None), so after a cell-list re-sort its rows may be
    permuted within a cell relative to the particles; the applied forces are
    still correct per particle. The thesis test compared sorted arrays for
    the same reason.
    """
    return a[np.lexsort(a.T[::-1])]

if args.restart:
    if u.isComputeTask():
        pv.additiveUpdateChannel('extraforces', 0.5)
    u.run(2, dt=0.0)

    if u.isComputeTask():
        # with dt=0 the particles never move and the saved forces must equal
        # the 'extraforces' channel, before and after the ramp
        files = sorted(glob.glob('h5/pv-*.h5'))
        with h5py.File(files[4], 'r') as f:  # last dump of the first run
            np.testing.assert_allclose(sort_rows(f['forces'][()]),
                                       sort_rows(f['extraforces'][()]), rtol=1e-6)
            xf0 = f['extraforces'][()]
        with h5py.File(files[-1], 'r') as f:  # after additiveUpdateChannel
            np.testing.assert_allclose(sort_rows(f['forces'][()]),
                                       sort_rows(f['extraforces'][()]), rtol=1e-6)
            xf1 = f['extraforces'][()]

        # the ramp must have moved every nonzero component away from zero by 0.5
        expected = xf0 + 0.5 * np.sign(xf0)
        np.testing.assert_allclose(sort_rows(xf1), sort_rows(expected), rtol=1e-6)

        print("OK")

# TEST: plugins.perparticleforce
# cd plugins
# rm -rf h5 ppf_restart perparticleforce.out.txt
# mir.run --runargs "-n 1" ./perparticleforce.py
# mir.run --runargs "-n 2" ./perparticleforce.py --restart > perparticleforce.out.txt
