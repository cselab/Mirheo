#include "rigid_ic.h"

#include <random>
#include <fstream>

#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/integrators/rigid_vv.h>

#include <core/rigid_kernels/rigid_motion.h>

void static readXYZ(std::string fname, PyContainer& positions)
{
    enum {X=0, Y=1, Z=2};
    int n;
    float dummy;
    std::string line;

    std::ifstream fin(fname);
    if (!fin.good())
        die("XYZ ellipsoid file %s not found", fname.c_str());
    fin >> n;

    // skip the comment line
    std::getline(fin, line);
    std::getline(fin, line);

    positions.resize(n);
    for (int i=0; i<n; i++)
        fin >> dummy >> positions[i][X] >> positions[i][Y] >> positions[i][Z];
}

RigidIC::RigidIC(ICvector com_q, std::string xyzfname) :
    com_q(com_q)
{
    readXYZ(xyzfname, coords);
}

RigidIC::RigidIC(ICvector com_q, const PyContainer& coords) :
    com_q(com_q), coords(coords)
{}

RigidIC::~RigidIC() = default;


static void copyToPinnedBuffer(const PyContainer& in, PinnedBuffer<float4>& out, cudaStream_t stream)
{
    enum {X=0, Y=1, Z=2};
    out.resize_anew(in.size());

    for (int i = 0; i < in.size(); ++i)
        out[i] = make_float4(in[i][X], in[i][Y], in[i][Z], 0);
        
    out.uploadToDevice(stream);    
}

void RigidIC::exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    if (ov == nullptr)
        die("Can only generate rigid object vector");

    pv->domain = domain;

    copyToPinnedBuffer(coords, ov->initialPositions, stream);

    if (ov->objSize != ov->initialPositions.size())
        die("Object size and XYZ initial conditions don't match in size for '%s': %d vs %d",
                ov->name.c_str(), ov->objSize, ov->initialPositions.size());

    int nObjs=0;
    HostBuffer<RigidMotion> motions;

    for (auto& entry : com_q)
    {
        RigidMotion motion{};
        
        motion.r = {entry[0], entry[1], entry[2]};
        motion.q = make_rigidReal4( make_float4(entry[3], entry[4], entry[5], entry[6]) );
        motion.q = normalize(motion.q);

        if (ov->domain.inSubDomain(motion.r))
        {
            motion.r = make_rigidReal3( ov->domain.global2local(make_float3(motion.r)) );
            motions.resize(nObjs + 1);
            motions[nObjs] = motion;
            nObjs++;
        }
    }

    ov->local()->resize_anew(nObjs * ov->objSize);

    auto ovMotions = ov->local()->extraPerObject.getData<RigidMotion>("motions");
    ovMotions->copy(motions);
    ovMotions->uploadToDevice(stream);

    // Set ids
    // Need to do that, as not all the objects in com_q may be valid
    int totalCount=0; // TODO: int64!
    MPI_Check( MPI_Exscan(&nObjs, &totalCount, 1, MPI_INT, MPI_SUM, comm) );

    auto ids = ov->local()->extraPerObject.getData<int>("ids");
    for (int i=0; i<nObjs; i++)
        (*ids)[i] = totalCount + i;


    for (int i=0; i < ov->local()->size(); i++)
    {
        Particle p(make_float4(0), make_float4(0));
        p.i1 = totalCount*ov->objSize + i;
        ov->local()->coosvels[i] = p;
    }

    ids->uploadToDevice(stream);
    ov->local()->coosvels.uploadToDevice(stream);
    ov->local()->extraPerParticle.getData<Particle>("old_particles")->copy(ov->local()->coosvels, stream);

    info("Read %d %s objects", nObjs, ov->name.c_str());

    // Do the initial rotation
    ov->requireDataPerObject<RigidMotion>("old_motions", false);
    ov->local()->forces.clear(stream);
    IntegratorVVRigid integrator("dummy", 0.0f);
    integrator.stage2(pv, 0, stream);
}

