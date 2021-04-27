#!/usr/bin/env python

"""
Test deleting objects by marking all their particles.

The tests below initialize the simulation with several non-interacting objects
along a line centered in y and z and parallel to the x-axis. The tests then
alternate between running and checking current object positions and IDs,
marking some objects in between for deletion."""

from mpi4py import MPI
import numpy as np
import sys
import trimesh
import unittest
import mirheo as mir

try:
    import cupy as cp
except ImportError:
    cp = None

rank = MPI.COMM_WORLD.rank


def to_cupy(a):
    """Circumvent a cupy bug where nullptr buffer is rejected, despite having a
    check for exactly that.
    https://github.com/cupy/cupy/blob/v8.5.0/cupy/core/core.pyx#L2376-L2377
    """
    interface = a.__cuda_array_interface__
    if interface['data'][0] != 0:
        return cp.asarray(a)
    else:
        return cp.ndarray(interface['shape'], dtype=np.dtype(interface['typestr']))


class TestDeleteObjects(unittest.TestCase):
    DOMAIN = [0.0, 20.0, 20.0]
    def _init(self, xs, type_ids, *, nranks_x, domain_x=40.0, comm=MPI.COMM_WORLD):
        """Create Mirheo with given domain size and number of ranks in
        x-direction. Initialized objects (boxes of side length 2) at given COM
        coordinates with given IDs."""
        assert len(xs) == len(type_ids)

        domain = [domain_x, self.DOMAIN[1], self.DOMAIN[2]]
        u = mir.Mirheo(
                nranks=(nranks_x, 1, 1), domain=domain, no_splash=True,
                debug_level=0, comm_ptr=MPI._addressof(comm))

        # Create OV with a dummy box mesh.
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        membrane = mir.ParticleVectors.MembraneMesh(
                vertices=mesh.vertices, faces=mesh.faces)
        ov = mir.ParticleVectors.MembraneVector('ov', mass=1.0, mesh=membrane)

        # Initialize objects with given COM_x. COM_y and COM_z are set to the
        # center of the domain.
        com_q = np.repeat([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], len(xs), axis=0)
        com_q[:, 0] = xs
        com_q[:, 1] = 0.5 * domain[1]
        com_q[:, 2] = 0.5 * domain[2]
        ic = mir.InitialConditions.MembraneWithTypeId(com_q, type_ids=type_ids)
        u.registerParticleVector(ov, ic)

        # Integrator.
        vv = mir.Integrators.VelocityVerlet('vv')
        u.registerIntegrator(vv)
        u.setIntegrator(vv, ov)

        return u, ov, mesh

    def assertObjects(self, u, ov, mesh, xs, type_ids):
        """Check if the objects have the expected coordinates and IDs.

        Note that as soon as redistribution is performed, the objects may be
        reshuffled due to the stochasticity of atomicAdd. Hence, we sort
        objects with one of their vertices as the sort key.
        """
        n = len(xs)
        m = len(mesh.vertices)

        # Read positions and reorder according to the x coordinate of one vertex.
        r = to_cupy(ov.local.per_particle['positions'])[:, :3].get()
        self.assertEqual(r.shape, (n * m, 3), msg=f"rank={rank}")
        order = np.lexsort(r[::m].T)
        r = r.reshape((n, m, 3))[order].reshape((n * m, 3))

        # Check coordinates.
        at = np.repeat([0.5 * np.array(self.DOMAIN)], n, axis=0)
        at[:, 0] = xs
        at = np.repeat(at, m, axis=0)
        expected = np.tile(mesh.vertices, (n, 1)) + at
        shift = np.array(tuple(u.getState().domain_info.local_to_global_shift))
        current = r + shift
        np.testing.assert_almost_equal(current, expected, decimal=5)

        # Check IDs.
        current = to_cupy(ov.local.per_object['membrane_type_id']).get()
        current = current[order]
        expected = type_ids
        np.testing.assert_array_equal(current, expected, f"rank={rank}")

    def markForDeletion(self, ov, mesh, type_id, v_dt=0.0):
        """Mark the object with the given ID for deletion.

        Since we mark the particles too early (before integration), the
        positions have to be shifted by v*dt.
        """
        m = len(mesh.vertices)
        r = to_cupy(ov.local.per_particle['positions'])[:, :3]
        type_ids = to_cupy(ov.local.per_object['membrane_type_id']).get().tolist()
        self.assertIn(type_id, type_ids)
        oid = type_ids.index(type_id)
        obj_r = r[m * oid : m * (oid + 1), :]
        obj_r[:] = ov.MARK_VALUE - v_dt

    @unittest.skipIf(cp is None, "cupy not found")
    def test_one_rank_one_object(self):
        """Test 1 rank with 1 object."""
        comm = MPI.COMM_WORLD.Split(rank < 1)
        if rank >= 1:
            return

        u, ov, mesh = self._init([8.5], [10], nranks_x=1, domain_x=40.0, comm=comm)

        u.run(0, dt=0.0)
        self.assertObjects(u, ov, mesh, [8.5], [10])

        self.markForDeletion(ov, mesh, type_id=10)
        u.run(1, dt=0.0)
        self.assertObjects(u, ov, mesh, [], [])

    @unittest.skipIf(cp is None, "cupy not found")
    def test_one_rank_multiple_objects(self):
        """Test 1 rank with multiple object."""
        comm = MPI.COMM_WORLD.Split(rank < 1)
        if rank >= 1:
            return

        xs = [8.5, 18.5, 23.5, 37.5]
        type_ids = [10, 20, 30, 40]
        u, ov, mesh = self._init(xs, type_ids, nranks_x=1, domain_x=40.0, comm=comm)

        u.run(1, dt=0.0)
        self.assertObjects(u, ov, mesh, [8.5, 18.5, 23.5, 37.5], [10, 20, 30, 40])

        self.markForDeletion(ov, mesh, type_id=30)
        u.run(1, dt=0.0)
        self.assertObjects(u, ov, mesh, [8.5, 18.5, 37.5], [10, 20, 40])

        self.markForDeletion(ov, mesh, type_id=10)
        self.markForDeletion(ov, mesh, type_id=20)
        self.markForDeletion(ov, mesh, type_id=40)
        u.run(1, dt=0.0)
        self.assertObjects(u, ov, mesh, [], [])

    @unittest.skipIf(cp is None, "cupy not found")
    def test_two_ranks_one_object(self):
        """Test 2 ranks with 1 object. The object moves from rank 0 to rank 1."""
        comm = MPI.COMM_WORLD.Split(rank < 2)
        if rank >= 2:
            return

        u, ov, mesh = self._init([19.5], [10], nranks_x=2, domain_x=40.0, comm=comm)

        u.run(0, dt=0.0)
        if rank == 0:
            self.assertObjects(u, ov, mesh, [19.5], [10])
        else:
            self.assertObjects(u, ov, mesh, [], [])

        # Test deleting the object at the same time step it should've migrated.
        to_cupy(ov.local.per_particle['velocities'])[:, 0] = 1.0
        if rank == 0:
            self.markForDeletion(ov, mesh, type_id=10, v_dt=1.0*1.0)
        u.run(1, dt=1.0)
        self.assertObjects(u, ov, mesh, [], [])

    @unittest.skipIf(cp is None, "cupy not found")
    def test_two_ranks_multiple_objects(self):
        """Test 2 ranks with multiple object. The objects move from rank 0 to rank 1."""
        comm = MPI.COMM_WORLD.Split(rank < 2)
        if rank >= 2:
            return

        xs = [17.3, 17.4, 18.5, 18.6, 19.7, 19.8]
        type_ids = [10, 20, 30, 40, 50, 60]
        u, ov, mesh = self._init(xs, type_ids, nranks_x=2, domain_x=40.0, comm=comm)

        u.run(0, dt=0.0)
        if rank == 0:
            self.assertObjects(u, ov, mesh, xs, type_ids)
        else:
            self.assertObjects(u, ov, mesh, [], [])

        # Test if migration itself works.
        to_cupy(ov.local.per_particle['velocities'])[:, 0] = 1.0
        u.run(1, dt=1.0)
        if rank == 0:
            self.assertObjects(u, ov, mesh, [18.3, 18.4, 19.5, 19.6], [10, 20, 30, 40])
        else:
            self.assertObjects(u, ov, mesh, [20.7, 20.8], [50, 60])

        # Test if deletion works.
        if rank == 0:
            self.markForDeletion(ov, mesh, type_id=20, v_dt=1.0*1.0)
            self.markForDeletion(ov, mesh, type_id=30, v_dt=1.0*1.0)
        else:
            self.markForDeletion(ov, mesh, type_id=50, v_dt=1.0*1.0)
        u.run(1, dt=1.0)
        if rank == 0:
            self.assertObjects(u, ov, mesh, [19.3], [10])
        else:
            self.assertObjects(u, ov, mesh, [20.6, 21.8], [40, 60])


if __name__ == '__main__':
    import sys
    out = unittest.main(argv=[sys.argv[0], '-q', '-f'], exit=False)
    if out.result.errors or out.result.failures:
        print("Failed!", flush=True)
        sys.stderr.flush()
        mir.abort()


# TEST: objects.delete_objects
# cd objects
# mir.run -n 2 ./delete_objects.py > delete_objects.out.txt
