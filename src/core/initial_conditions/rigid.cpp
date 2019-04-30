#include "rigid.h"

#include <core/integrators/rigid_vv.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/rigid_kernels/rigid_motion.h>

#include <fstream>
#include <random>

void static readXYZ(std::string fname, PyTypes::VectorOfFloat3& positions)
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

RigidIC::RigidIC(PyTypes::VectorOfFloat7 com_q, std::string xyzfname) :
    com_q(com_q)
{
    readXYZ(xyzfname, coords);
}

RigidIC::RigidIC(PyTypes::VectorOfFloat7 com_q, const PyTypes::VectorOfFloat3& coords) :
    com_q(com_q), coords(coords)
{}

RigidIC::RigidIC(PyTypes::VectorOfFloat7 com_q, const PyTypes::VectorOfFloat3& coords, const PyTypes::VectorOfFloat3& comVelocities) :
    com_q(com_q), coords(coords), comVelocities(comVelocities)
{
    if (com_q.size() != comVelocities.size())
        die("Incompatible sizes of initial positions and rotations");
}

RigidIC::~RigidIC() = default;


static void copyToPinnedBuffer(const PyTypes::VectorOfFloat3& in, PinnedBuffer<float4>& out, cudaStream_t stream)
{
    enum {X=0, Y=1, Z=2};
    out.resize_anew(in.size());

    for (int i = 0; i < in.size(); ++i)
        out[i] = make_float4(in[i][X], in[i][Y], in[i][Z], 0);
        
    out.uploadToDevice(stream);    
}

void RigidIC::exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    if (ov == nullptr)
        die("Can only generate rigid object vector");

    copyToPinnedBuffer(coords, ov->initialPositions, stream);
    if (ov->objSize != ov->initialPositions.size())
        die("Object size and XYZ initial conditions don't match in size for '%s': %d vs %d",
            ov->name.c_str(), ov->objSize, ov->initialPositions.size());

    int nObjs = 0;
    HostBuffer<RigidMotion> motions;

    for (int i = 0; i < com_q.size(); i++)
    {
        auto& entry = com_q[i];
        
        // Zero everything at first
        RigidMotion motion{};
        
        motion.r = {entry[0], entry[1], entry[2]};
        motion.q = make_rigidReal4( make_float4(entry[3], entry[4], entry[5], entry[6]) );
        motion.q = normalize(motion.q);
        
        if (i < comVelocities.size()) motion.vel = {comVelocities[i][0], comVelocities[i][1], comVelocities[i][2]};

        if (ov->state->domain.inSubDomain(motion.r))
        {
            motion.r = make_rigidReal3( ov->state->domain.global2local(make_float3(motion.r)) );
            motions.resize(nObjs + 1);
            motions[nObjs] = motion;
            nObjs++;
        }
    }

    ov->local()->resize_anew(nObjs * ov->objSize);

    auto ovMotions = ov->local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    ovMotions->copy(motions);
    ovMotions->uploadToDevice(stream);

    ov->local()->positions().uploadToDevice(stream);
    ov->local()->velocities().uploadToDevice(stream);
    ov->local()->computeGlobalIds(comm, stream);
    ov->local()->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->copy(ov->local()->positions(), stream);

    info("Read %d %s objects", nObjs, ov->name.c_str());

    // Do the initial rotation
    ov->local()->forces().clear(stream);
    YmrState dummyState(ov->state->domain, /* dt */ 0.f);
    IntegratorVVRigid integrator(&dummyState, "__dummy__");
    integrator.stage2(pv, stream);
}

