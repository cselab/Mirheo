.. _user-bench:

Benchmarks
###########

The following benchmarks represent typical use cases of *uDeviceX*.
They were performed on the `Piz-Daint <https://www.cscs.ch/computers/piz-daint/>`_ supercomputer for both strong and weak scaling.
See in `benchmarks/cases/` for more informations about the run scripts.


Bulk Solvent
============

Periodic Poiseuille flow in a periodic domain in every direction, with solvent only.

.. figure:: ../images/strong_solvent.png
    :figclass: align-center

    strong scaling for multiple domain sizes


.. figure:: ../images/weak_solvent.png
    :figclass: align-center

    weak scaling efficiency for multiple subdomain sizes


Bulk Blood
==========

Periodic Poiseuille flow in a periodic domain in every direction with 45% Hematocrite.

.. figure:: ../images/strong_blood.png
    :figclass: align-center

    strong scaling for multiple domain sizes


.. figure:: ../images/weak_blood.png
    :figclass: align-center

    weak scaling efficiency for multiple subdomain sizes

The weak scaling efficiency is lower than in the solvent only case because of the complexity of the problem:
* Multiple solvents
* FSI interactions
* Many objects
* Bounce back

The above induces a lot more communication.

Poiseuille Flow
===============

TODO

Rigid Objects suspension
========================

TODO
