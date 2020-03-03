.. _dev-domain:

Domain
======

In Mirheo, the simulation domain has a rectangular shape subdivided in equal subdomains.
Each simulation rank is mapped to a single subdomain in a cartesian way.
Furthermore, we distinguish the global coordinates (that are the same for all ranks) from the local coordinates (different from one subdomain to another).
The :any:`mirheo::DomainInfo` utility class provides a description of the domain, subdomain and a mapping between the coordinates of these two entities.


API
---

.. doxygenfunction:: mirheo::createDomainInfo
   :project: mirheo

.. doxygenstruct:: mirheo::DomainInfo
   :project: mirheo
   :members:

