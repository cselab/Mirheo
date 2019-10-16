#include "rigid.h"

#include <core/integrators/rigid_vv.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/rigid/utils.h>

#include <fstream>
#include <random>

static std::vector<float3> readXYZ(const std::string& fname)
{
    std::vector<float3> positions;
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
    for (int i = 0; i < n; ++i)
        fin >> dummy >> positions[i].x >> positions[i].y >> positions[i].z;
    return positions;
}



RigidIC::RigidIC(const std::vector<ComQ>& com_q, const std::string& xyzfname) :
    RigidIC(com_q, readXYZ(xyzfname))
{}

RigidIC::RigidIC(const std::vector<ComQ>& com_q, const std::vector<float3>& coords) :
    com_q(com_q),
    coords(coords)
{}

RigidIC::RigidIC(const std::vector<ComQ>& com_q,
                 const std::vector<float3>& coords,
                 const std::vector<float3>& comVelocities) :
    com_q(com_q),
    coords(coords),
    comVelocities(comVelocities)
{
    if (com_q.size() != comVelocities.size())
        die("Incompatible sizes of initial positions and rotations");
}

RigidIC::~RigidIC() = default;


static PinnedBuffer<float4> getInitialPositions(const std::vector<float3>& in,
                                                cudaStream_t stream)
{
    PinnedBuffer<float4> out(in.size());
    
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = make_float4(in[i].x, in[i].y, in[i].z, 0);
        
    out.uploadToDevice(stream);
    return out;
}

static void checkInitialPositions(const DomainInfo& domain,
                                  const PinnedBuffer<float4>& positions)
{
    if (positions.size() < 1)
        die("Expect at least one particle per rigid object");

    const float3 r0 = make_float3(positions[0]);
    
    float3 low {r0}, hig {r0};
    for (auto r4 : positions)
    {
        const float3 r = make_float3(r4);
        low = fminf(low, r);
        hig = fmaxf(hig, r);
    }

    const auto L = domain.localSize;
    const auto l = hig - low;

    const auto Lmax = std::max(L.x, std::max(L.y, L.z));
    const auto lmax = std::max(l.x, std::max(l.y, l.z));

    if (lmax >= Lmax)
        warn("Object dimensions are larger than the domain size");
}

static std::vector<RigidMotion> createMotions(const DomainInfo& domain,
                                              const std::vector<ComQ>& com_q,
                                              const std::vector<float3>& comVelocities)
{
    std::vector<RigidMotion> motions;

    for (size_t i = 0; i < com_q.size(); ++i)
    {
        const auto& entry = com_q[i];
        
        // Zero everything at first
        RigidMotion motion{};
        
        motion.r = make_rigidReal3( entry.r );
        motion.q = make_rigidReal4( entry.q );
        motion.q = normalize(motion.q);
        
        if (i < comVelocities.size())
            motion.vel = {comVelocities[i].x, comVelocities[i].y, comVelocities[i].z};

        if (domain.inSubDomain(motion.r))
        {
            motion.r = make_rigidReal3( domain.global2local(make_float3(motion.r)) );
            motions.push_back(motion);
        }
    }
    return motions;
}

static void setParticlesFromMotions(RigidObjectVector *rov, cudaStream_t stream)
{
    // use rigid object integrator to set up the particles positions, velocities and old positions
    rov->local()->forces().clear(stream);
    const float dummyDt = 0.f;
    const MirState dummyState(rov->state->domain, dummyDt);
    IntegratorVVRigid integrator(&dummyState, "__dummy__");
    integrator.stage2(rov, stream);
}

void RigidIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto rov = dynamic_cast<RigidObjectVector*>(pv);
    if (rov == nullptr)
        die("Can only generate rigid object vector");

    const auto domain = rov->state->domain;

    rov->initialPositions = getInitialPositions(coords, stream);
    checkInitialPositions(domain, rov->initialPositions);

    auto lrov = rov->local();
    
    if (rov->objSize != static_cast<int>(rov->initialPositions.size()))
        die("Object size and XYZ initial conditions don't match in size for '%s': %d vs %d",
            rov->name.c_str(), rov->objSize, rov->initialPositions.size());

    const auto motions = createMotions(domain, com_q, comVelocities);
    const auto nObjs = motions.size();
    
    lrov->resize_anew(nObjs * rov->objSize);

    auto& rovMotions = *lrov->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    std::copy(motions.begin(), motions.end(), rovMotions.begin());
    rovMotions.uploadToDevice(stream);

    setParticlesFromMotions(rov, stream);
    lrov->computeGlobalIds(comm, stream);

    info("Read %d %s objects", nObjs, rov->name.c_str());
}

