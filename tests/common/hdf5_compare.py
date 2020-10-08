#!/usr/bin/env python3
"""
Compare HDF5 files.
"""

import h5py
import numpy as np

def compare_pvs(*paths, rtol, ids_column='ids'):
    """Compare two or more dumps of a ParticleVector.

    The data is imported into numpy, sorted with respect to particle IDs and
    compared.

    Usage:
        ./hdf5_compare.py compare_pvs file1.h5 file2.h5 [...]
    """
    ref_ids = None
    ref_values = None
    for path in paths:
        f = h5py.File(path, 'r')

        values = []
        ids = None
        for k, v in f.items():
            v = np.asarray(v)
            if k == ids_column:
                assert len(v.shape) == 2 and v.shape[1] == 1
                ids = v.flatten()
            else:
                values.append(v)

        if ids is None:
            raise ValueError(f"No column `{ids_column}` in `{file}`.")

        order = np.argsort(ids)
        ids = order[ids]

        values = np.concatenate(values, axis=1)
        values = values[order, :]

        if ref_ids is None:
            ref_ids = ids
            ref_values = values
            continue

        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_allclose(values, ref_values, rtol=rtol)

        f.close()


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    subparser = subparsers.add_parser('compare_pvs', help="Compare ParticleVector dump HDF5 files.")
    subparser.add_argument('--rtol', type=float, default=1e-6, help="Maximum acceptable relative error")
    subparser.add_argument('--files', type=str, nargs='+', help="Files to compare")

    args = parser.parse_args(argv)

    if args.cmd == 'compare_pvs':
        compare_pvs(*args.files, rtol=args.rtol)
    else:
        raise ValueError(args.dest)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
