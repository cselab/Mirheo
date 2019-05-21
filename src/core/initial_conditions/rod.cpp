#include "rod.h"

#include <core/pvs/rod_vector.h>
#include <core/utils/quaternion.h>

#include <fstream>
#include <random>

RodIC::RodIC(PyTypes::VectorOfFloat7 com_q, MappingFunc3D centerLine, MappingFunc1D torsion, float a) :
    com_q(com_q),
    centerLine(centerLine),
    torsion(torsion),
    a(a)
{}

RodIC::~RodIC() = default;

static float3 getFirstBishop(float3 r0, float3 r1, float3 r2)
{
    float3 t0 = normalize(r1 - r0);
    float3 t1 = normalize(r2 - r1);
    float3 b = cross(t0, t1);
    float3 u;
    
    if (length(b) > 1e-6)
    {
        u = b - dot(b, t0) * t0;
    }
    else
    {
        u = anyOrthogonal(t0);
    }
    return normalize(u);
}

std::vector<float3> createRodTemplate(int nSegments, float a,
                                      const RodIC::MappingFunc3D& centerLine,
                                      const RodIC::MappingFunc1D& torsion)
{
    assert(nSegments > 1);
    
    std::vector<float3> positions (5*nSegments + 1);
    float h = 1.f / nSegments;

    float3 u; // bishop frame
    
    for (int i = 0; i <= nSegments; ++i)
        positions[i*5] = make_float3(centerLine(i*h));

    u = getFirstBishop(positions[0], positions[5], positions[10]);

    double theta = 0; // angle w.r.t. bishop frame
    
    for (int i = 0; i < nSegments; ++i)
    {
        auto r0 = positions[5*(i + 0)];
        auto r1 = positions[5*(i + 1)];

        auto r = 0.5f * (r0 + r1);
        float cost = cos(theta);
        float sint = sin(theta);

        auto t0 = normalize(r1-r0);

        u = normalize(u - dot(t0, u)*t0);
        auto v = cross(t0, u);

        // material frame
        float3 mu =  cost * u + sint * v;
        float3 mv = -sint * u + cost * v;
            
        positions[5*i + 1] = r - 0.5 * a * mu;
        positions[5*i + 2] = r + 0.5 * a * mu;
        positions[5*i + 3] = r - 0.5 * a * mv;
        positions[5*i + 4] = r + 0.5 * a * mv;

        if (i < nSegments - 1)
        {
            auto r2 = positions[5*(i + 2)];
            auto t1 = normalize(r2-r1);

            auto q = getQfrom(t0, t1);
            u = normalize(rotate(u, q));

            auto l = 0.5 * (length(r1-r0) + length(r2-r1));
            // use trapezoidal rule to integrate the angle
            theta += l * 0.5 * (torsion((i+0.5f)*h) + torsion((i+1.5f)*h));
        }
    }
    
    return positions;
}

void RodIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto rv = dynamic_cast<RodVector*>(pv);
    auto domain = pv->state->domain;
    
    if (rv == nullptr)
        die("rods can only be generated out of rod vectors; provided '%s'", pv->name.c_str());

    int objSize = rv->objSize;
    int nObjs = 0;
    int nSegments = (objSize - 1) / 5;

    auto positions = createRodTemplate(nSegments, a, centerLine, torsion);

    assert(objSize == positions.size());
    
    for (auto& entry : com_q)
    {
        float3 com {entry[0], entry[1], entry[2]};
        float4 q   {entry[3], entry[4], entry[5], entry[6]};

        q = normalize(q);        

        if (domain.globalStart.x <= com.x && com.x < domain.globalStart.x + domain.localSize.x &&
            domain.globalStart.y <= com.y && com.y < domain.globalStart.y + domain.localSize.y &&
            domain.globalStart.z <= com.z && com.z < domain.globalStart.z + domain.localSize.z)
        {
            com = domain.global2local(com);
            int oldSize = rv->local()->size();
            rv->local()->resize(oldSize + objSize, stream);

            float4 *pos = rv->local()->positions() .data();
            float4 *vel = rv->local()->velocities().data();
            
            for (int i = 0; i < objSize; i++)
            {
                float3 r = rotate(positions[i], q) + com;
                Particle p;
                p.r = r;
                p.u = make_float3(0);

                pos[oldSize + i] = p.r2Float4();
                vel[oldSize + i] = p.u2Float4();
            }

            nObjs++;
        }
    }

    rv->local()->positions() .uploadToDevice(stream);
    rv->local()->velocities().uploadToDevice(stream);
    rv->local()->computeGlobalIds(comm, stream);
    rv->local()->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->copy(rv->local()->positions(), stream);

    info("Initialized %d '%s' rods", nObjs, rv->name.c_str());
}

