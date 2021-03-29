"""
Test ParticleVector channel __cuda_array_interface__ with cupy.
"""

import unittest
import numpy as np
import mirheo as mir

try:
    import cupy as cp
except ImportError:
    cp = None


class TestParticleVector(unittest.TestCase):
    @unittest.skipIf(cp is None, "Skipping cupy tests, cupy not found.")
    def test_cupy_access(self):
        N = 5
        domain = np.array([10.0, 10.0, 10.0])
        mid = 0.5 * domain
        r0 = np.tile(mid, (N, 1))  # Particles start at the middle of the domain.
        v0 = np.zeros((N, 3))

        # Mirheo setup with 1 PV and a VV integrator.
        u = mir.Mirheo(nranks=(1, 1, 1), domain=domain, no_splash=True)
        pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
        u.registerParticleVector(pv, mir.InitialConditions.FromArray(r0, v0))
        vv = mir.Integrators.VelocityVerlet('vv')
        u.registerIntegrator(vv)
        u.setIntegrator(vv, pv)

        # Test shape.
        r = cp.asarray(pv.local['positions'])
        v = cp.asarray(pv.local['velocities'])
        f = cp.asarray(pv.local['__forces'])
        self.assertEqual(r.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(v.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(f.shape, (N, 3))  # The `int` element is stripped away.

        # Test updating velocities and reading positions.
        v = cp.asarray(pv.local['velocities'])
        v[:, :3] = cp.array([
            [10, 20, 30],
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33],
            [14, 24, 34],
        ])
        u.run(1, dt=10.0)
        r = cp.asarray(pv.local['positions'])[:, :3]
        v = cp.asarray(pv.local['velocities'])[:, :3]
        cp.testing.assert_array_equal(r, cp.array(r0 - mid) + 10.0 * v)

        # Test forces.
        r[:] = 0.0  # Reset positions to the middle of the domain.
        v[:] = 0.0  # Reset velocities.
        u.registerPlugins(mir.Plugins.createAddForce('add_force', pv, (1.0, 2.0, 3.0)))
        u.run(1, dt=10.0)
        r = cp.asarray(pv.local['positions'])[:, :3]
        v = cp.asarray(pv.local['velocities'])[:, :3]
        f = cp.asarray(pv.local['__forces'])
        cp.testing.assert_array_equal(r, [[100.0, 200.0, 300.0]] * N)
        cp.testing.assert_array_equal(v, [[10.0, 20.0, 30.0]] * N)
        cp.testing.assert_array_equal(f, [[1.0, 2.0, 3.0]] * N)
