.. _user-mirheo:

Mirheo coordinator
######################

The coordinator class stitches together data containers, :ref:`user-pv`, and all the handlers,
and provides functions to manipulate the system components.

One and only one instance of this class should be created in the beginning of any simulation setup.

.. note::
    Creating the coordinator will internally call MPI_Init() function, and its destruction
    will call MPI_Finalize().
    Therefore if using a mpi4py Python module, it should be imported in the following way:
    
    .. code-block:: python
        
        import  mpi4py
        mpi4py.rc(initialize=False, finalize=False)
        from mpi4py import MPI

        
.. autoclass:: _mirheo.Mirheo
   :members:
   :undoc-members:
   :special-members: __init__

    .. rubric:: Methods

    .. autoautosummary:: _mirheo.Mirheo
        :methods:


Unit system
===========

Mirheo assumes all values are dimensionless.
However, users may use Mirheo in combination with the pint_ Python package, by defining Mirheo's unit system using :any:`set_unit_registry`:

.. code:: python

    import mirheo as mir
    import pint
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 um')
    ureg.define('mirT = 1 us')
    ureg.define('mirM = 1e-20 kg')
    mir.set_unit_registry(ureg)

    # dt automatically converted to 0.01, matching the value of 1e-8 s in the Mirheo unit system.
    u = mir.Mirheo(..., dt=ureg('1e-8 s'), ...)

.. autofunction:: mirheo.set_unit_registry

.. _pint: https://pint.readthedocs.io/
