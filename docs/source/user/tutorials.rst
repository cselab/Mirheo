.. _user-tuto:

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash
   
Tutorials
##########

This section introduces more in details the **YMeRo** interface to the user step by step with examples.


Hello World: run YMeRo
**********************

We start with a very minimal script running **YMeRo**.

.. literalinclude:: ../../../tests/doc_scripts/hello.py
   :name: hello.py

The time step of the simulation and the domain size are common to all objects in the simulation,
hence it has to be passed to the coordinator.
We do not add anything more before running the simulation (last line).

.. note::
    We also specified the number of ranks in **each** direction.
    Together with the domain size, this tells **YMeRo** how the simulation domain will be splitted accross MPI ranks.
    The number of simulation tasks must correspond to this variable.

The above script can be run as:

.. code-block:: bash

    mpirun -np 1 python3 hello.py

Running `hello.py` will only print the "hello world" message of **YMeRo**, which consists of the version and git SHA1 of the code.
Furthermore, **YMeRo** will dump log files (one per MPI rank) which name is specified when creating the coordinator.
Depending on the `debug_level` variable, the log files will provide information on the simulation progress.


DPD solvent at rest
*******************

.. literalinclude:: ../../../tests/doc_scripts/rest.py
   :name: rest.py

