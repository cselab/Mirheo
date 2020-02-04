#include "rod.h"

#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/utils/quaternion.h>

#include <fstream>
#include <limits>
#include <random>

namespace mirheo
{

const real RodIC::Default = std::numeric_limits<real>::infinity();
const real3 RodIC::DefaultFrame = {Default, Default, Default};

RodIC::RodIC(const std::vector<ComQ>& comQ, MappingFunc3D centerLine, MappingFunc1D torsion,
             real a, real3 initialMaterialFrame) :
    comQ_(comQ),
    centerLine_(centerLine),
    torsion_(torsion),
    initialMaterialFrame_(initialMaterialFrame),
    a_(a)
{}

RodIC::~RodIC() = default;

static bool isDefaultFrame(real3 v)
{
    const real defVal = RodIC::Default;
    return v.x == defVal && v.y == defVal && v.z == defVal;
}

static real3 getFirstBishop(real3 r0, real3 r1, real3 r2, real3 initialMaterialFrame)
{
    const real3 t0 = normalize(r1 - r0);
    real3 u;
    
    if (isDefaultFrame(initialMaterialFrame))
    {
        const real3 t1 = normalize(r2 - r1);
        const real3 b = cross(t0, t1);
        
        if (length(b) > 1e-6_r)
        {
            u = b - dot(b, t0) * t0;
        }
        else
        {
            u = anyOrthogonal(t0);
        }
    }
    else
    {
        u = initialMaterialFrame - dot(initialMaterialFrame, t0);

        if (length(u) < 1e-4)
            die("provided initial frame must not be aligned with the centerline");
    }
    return normalize(u);
}

static std::vector<real3> createRodTemplate(int nSegments, real a, real3 initialMaterialFrame,
                                            const RodIC::MappingFunc3D& centerLine,
                                            const RodIC::MappingFunc1D& torsion)
{
    assert(nSegments > 1);
    
    std::vector<real3> positions (5*nSegments + 1);
    real h = 1._r / static_cast<real>(nSegments);

    real3 u; // bishop frame
    
    for (int i = 0; i <= nSegments; ++i)
        positions[i*5] = make_real3(centerLine(static_cast<real>(i)*h));

    u = getFirstBishop(positions[0], positions[5], positions[10], initialMaterialFrame);

    double theta = 0; // angle w.r.t. bishop frame
    
    for (int i = 0; i < nSegments; ++i)
    {
        auto r0 = positions[5*(i + 0)];
        auto r1 = positions[5*(i + 1)];

        auto r = 0.5_r * (r0 + r1);
        real cost = static_cast<real>(math::cos(theta));
        real sint = static_cast<real>(math::sin(theta));

        auto t0 = normalize(r1-r0);

        u = normalize(u - dot(t0, u)*t0);
        auto v = cross(t0, u);

        // material frame
        real3 mu =  cost * u + sint * v;
        real3 mv = -sint * u + cost * v;
            
        positions[5*i + 1] = r - 0.5_r * a * mu;
        positions[5*i + 2] = r + 0.5_r * a * mu;
        positions[5*i + 3] = r - 0.5_r * a * mv;
        positions[5*i + 4] = r + 0.5_r * a * mv;

        if (i < nSegments - 1)
        {
            auto r2 = positions[5*(i + 2)];
            auto t1 = normalize(r2-r1);

            const auto q = Quaternion<real>::createFromVectors(t0, t1);
            u = normalize(q.rotate(u));

            auto l = 0.5_r * (length(r1-r0) + length(r2-r1));
            // use trapezoidal rule to integrate the angle
            theta += l * 0.5_r * (torsion((static_cast<real>(i)+0.5_r)*h) + torsion((static_cast<real>(i)+1.5_r)*h));
        }
    }
    
    return positions;
}

void RodIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto rv = dynamic_cast<RodVector*>(pv);
    auto domain = pv->getState()->domain;
    
    if (rv == nullptr)
        die("rods can only be generated out of rod vectors; provided '%s'", pv->getCName());

    const int objSize = rv->objSize;
    const int nSegments = (objSize - 1) / 5;
    int nObjs = 0;

    const auto positions = createRodTemplate(nSegments, a_, initialMaterialFrame_, centerLine_, torsion_);

    assert(objSize == static_cast<int>(positions.size()));
    
    for (auto& entry : comQ_)
    {
        real3 com = entry.r;
        const auto q = Quaternion<real>::createFromComponents(entry.q).normalized();

        if (domain.globalStart.x <= com.x && com.x < domain.globalStart.x + domain.localSize.x &&
            domain.globalStart.y <= com.y && com.y < domain.globalStart.y + domain.localSize.y &&
            domain.globalStart.z <= com.z && com.z < domain.globalStart.z + domain.localSize.z)
        {
            com = domain.global2local(com);
            int oldSize = rv->local()->size();
            rv->local()->resize(oldSize + objSize, stream);

            real4 *pos = rv->local()->positions() .data();
            real4 *vel = rv->local()->velocities().data();
            
            for (int i = 0; i < objSize; i++)
            {
                const real3 r = com + q.rotate(positions[i]);
                const Particle p {{r.x, r.y, r.z, 0._r}, make_real4(0._r)};

                pos[oldSize + i] = p.r2Real4();
                vel[oldSize + i] = p.u2Real4();
            }

            nObjs++;
        }
    }

    rv->local()->positions() .uploadToDevice(stream);
    rv->local()->velocities().uploadToDevice(stream);
    rv->local()->computeGlobalIds(comm, stream);
    rv->local()->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->copy(rv->local()->positions(), stream);

    info("Initialized %d '%s' rods", nObjs, rv->getCName());
}


} // namespace mirheo
