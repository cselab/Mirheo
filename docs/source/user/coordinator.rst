.. _user-udevicex:

uDeviceX coordinator
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

        
.. autoclass:: _udevicex.udevicex
   :members:
   :undoc-members:
   :special-members: __init__

    .. rubric:: Methods

    .. autoautosummary:: _udevicex.udevicex
        :methods:

 
