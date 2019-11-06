#include <mirheo/core/datatypes.h>
#include <mirheo/core/integrators/rigid_vv.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/rigid/utils.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace mirheo;

namespace mirheo { Logger logger; }

static inline std::vector<float3> makeTemplate(real x = 1.0_r, real y = 1.0_r, real z = 1.0_r)
{
    return {{+x, 0.0_r, 0.0_r},
            {-x, 0.0_r, 0.0_r},
            {0.0_r, +y, 0.0_r},
            {0.0_r, -y, 0.0_r},
            {0.0_r, 0.0_r, +z},
            {0.0_r, 0.0_r, -z}};
}

static PinnedBuffer<real4> getInitialPositions(const std::vector<real3>& in, cudaStream_t stream)
{
    PinnedBuffer<real4> out(in.size());
    
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = make_real4(in[i].x, in[i].y, in[i].z, 0);
        
    out.uploadToDevice(stream);
    return out;
}

static void setParticlesFromMotions(RigidObjectVector *rov, cudaStream_t stream)
{
    // use rigid object integrator to set up the particles positions, velocities and old positions
    rov->local()->forces().clear(stream);
    const real dummyDt = 0._r;
    const MirState dummyState(rov->state->domain, dummyDt);
    IntegratorVVRigid integrator(&dummyState, "__dummy__");
    integrator.stage2(rov, stream);
}

static inline std::unique_ptr<RigidObjectVector> makeRigidVector(const MirState *state, real mass, real3 J, real3 omega, cudaStream_t stream)
{
    const auto posTemplate = makeTemplate();
    const int objSize = posTemplate.size();
    const int nObjects = 1;
    const real partMass = mass / objSize;

    auto rov = std::make_unique<RigidObjectVector>(state, "rigid_body", partMass, J, objSize, std::make_shared<Mesh>(), nObjects);

    auto lrov = rov->local();
    auto& motions = *lrov->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    
    for (auto& m : motions)
    {
        constexpr RigidReal3 zero3 {0.0_r, 0.0_r, 0.0_r};
        constexpr RigidReal4 qIdentity {1.0_r, 0.0_r, 0.0_r, 0.0_r};
        
        m.r = m.vel = m.force = m.torque = zero3;
        m.q = qIdentity;
        m.omega = make_rigidReal3(omega);
    }
    motions.uploadToDevice(stream);

    rov->initialPositions = getInitialPositions(posTemplate, stream);
    setParticlesFromMotions(rov.get(), stream);
    
    return rov;
}

TEST (RIGID_MOTION, GPU)
{
    constexpr real dt = 1e-3_r;
    constexpr real tend = 10.0_r;
    constexpr real L = 32.0_r;
    constexpr real mass = 1.0_r;

    const real3 J     {1.0_r, 2.0_r, 3.0_r};
    const real3 omega {10.0_r, 5.0_r, 2.0_r};
    

    DomainInfo domain {{L, L, L}, {0._r, 0._r, 0._r}, {L, L, L}};
    MirState state(domain, dt);

    IntegratorVVRigid gpuIntegrator(&state, "rigid_vv");
    auto rov = makeRigidVector(&state, mass, J, omega, defaultStream);

    gpuIntegrator.setPrerequisites(rov.get());

    for (; state.currentTime < tend; state.currentTime += state.dt)
    {
        gpuIntegrator.stage1(rov.get(), defaultStream);
        gpuIntegrator.stage2(rov.get(), defaultStream);
    }

    // for (int i = 0; i < N; ++i)
    //     ASSERT_EQ(hData[i], dData[i]);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
