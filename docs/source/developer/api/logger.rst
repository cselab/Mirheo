.. _dev-logger:

Logger
======

Example of a log entry::
  
  15:10:35:639  Rank 0000    INFO at /Mirheo/src/mirheo/core/logger.cpp:54 Compiled with maximum debug level 10
  15:10:35:640  Rank 0000    INFO at /Mirheo/src/mirheo/core/logger.cpp:56 Debug level requested 3, set to 3
  15:10:35:640  Rank 0000    INFO at /Mirheo/src/mirheo/core/mirheo.cpp:110 Program started, splitting communicator
  15:10:35:684  Rank 0000    INFO at /Mirheo/src/mirheo/core/mirheo.cpp:58 Detected 1 ranks per node, my intra-node ID will be 0
  15:10:35:717  Rank 0000    INFO at /Mirheo/src/mirheo/core/mirheo.cpp:65 Found 1 GPUs per node, will use GPU 0


API
---

.. doxygenclass:: mirheo::Logger
   :project: mirheo
   :members:
