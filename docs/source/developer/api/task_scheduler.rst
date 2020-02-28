.. _dev-task-scheduler:

Task Scheduler
==============

Becaus of the high number of tasks to execute and their :ref:`complex dependencies<dev-tasks>`, Mirheo uses
a :any:`mirheo::TaskScheduler` that takes care of executing all these tasks on concurrent streams.
The synchronization is therefore hidden in this class.

API
---

.. doxygenclass:: mirheo::TaskScheduler
   :project: mirheo
   :members:

