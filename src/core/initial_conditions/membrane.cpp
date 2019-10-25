#include "membrane.h"

#include <core/pvs/membrane_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/quaternion.h>

#include <fstream>
#include <random>

MembraneIC::MembraneIC(const std::vector<ComQ>& com_q, real globalScale) :
    com_q(com_q),
    globalScale(globalScale)
{}

MembraneIC::~MembraneIC() = default;

/**
 * Read mesh topology and initial vertices (vertices are same as particles for RBCs)
 *
 * Then read the center of mass coordinates of the cells and their orientation
 * quaternions from file #icfname.
 *
 * The format of the initial conditions file is such that each line defines
 * one RBC with the following 7 numbers separated by spaces:
 * <tt>COM.x COM.y COM.z  Q.x Q.y Q.z Q.w</tt>
 * \sa quaternion.h
 *
 * To generate an RBC from the IC file, the initial coordinate pattern will be
 * shifted to the COM and rotated according to Q.
 *
 * The RBCs with COM outside of an MPI process's domain will be discarded on
 * that process.
 *
 * Set unique id to all the particles and also write unique cell ids into
 * 'ids' per-object channel
 */
void MembraneIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector*>(pv);
    auto domain = pv->state->domain;
    
    if (ov == nullptr)
        die("RBCs can only be generated out of rbc object vectors");

    // Local number of objects
    int nObjs=0;

    for (auto& entry : com_q)
    {
        real3 com = entry.r;
        real4 q   = entry.q;

        q = normalize(q);

        if (domain.globalStart.x <= com.x && com.x < domain.globalStart.x + domain.localSize.x &&
            domain.globalStart.y <= com.y && com.y < domain.globalStart.y + domain.localSize.y &&
            domain.globalStart.z <= com.z && com.z < domain.globalStart.z + domain.localSize.z)
        {
            com = domain.global2local(com);
            const int oldSize = ov->local()->size();
            ov->local()->resize(oldSize + ov->mesh->getNvertices(), stream);

            auto& pos = ov->local()->positions();
            auto& vel = ov->local()->velocities();
            
            for (int i = 0; i < ov->mesh->getNvertices(); i++)
            {
                const real3 r = Quaternion::rotate(make_real3( ov->mesh->vertexCoordinates[i] * globalScale ), q) + com;
                const Particle p {{r.x, r.y, r.z, 0.f}, make_real4(0.f)};

                pos[oldSize + i] = p.r2Real4();
                vel[oldSize + i] = p.u2Real4();
            }

            nObjs++;
        }
    }

    ov->local()->positions().uploadToDevice(stream);
    ov->local()->velocities().uploadToDevice(stream);
    ov->local()->computeGlobalIds(comm, stream);
    ov->local()->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->copy(ov->local()->positions(), stream);

    info("Initialized %d '%s' membranes", nObjs, ov->name.c_str());
}

