#!/usr/bin/env python

import h5py as h5
import sys

def h5_open(filename):
    """Open a HDF5 file."""
    try:
        return h5.File(filename, "r")
    except IOError:
        sys.stderr.write("u.avgh5: fails to open <%s>\n" % filename)
        sys.exit(2)


def get_h5_forces(filename):
    """Read the forces channel from a h5 file."""
    f = h5_open(filename)
    forces = f["forces"][()]
    return forces
