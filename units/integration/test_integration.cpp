#include <core/integrators/factory.h>
#include <core/logger.h>

#include <gtest/gtest.h>
#include <vector>

Logger logger;
cudaStream_t defaultStream = 0;

static void run_gpu(Integrator *integrator, ParticleVector *pv, int nsteps, YmrState *state)
{
    integrator->setPrerequisites(pv);
    
    for (int i = 0; i < nsteps; ++i) {
        state->currentStep = i;
        state->currentTime = i * state->dt;
        
        integrator->stage1(pv, defaultStream);
        integrator->stage2(pv, defaultStream);
    }

    pv->local()->positions ().downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    pv->local()->velocities().downloadFromDevice(defaultStream, ContainersSynch::Synch);
}

static void run_cpu(std::vector<float4>& pos, std::vector<float4>& vel,
                    std::vector<Force>& forces, int nsteps, float dt, float mass)
{
    float dt_m = dt / mass;
    
    for (int step = 0; step < nsteps; ++step) {
        for (int i = 0; i < pos.size(); ++i) {
            float4& r = pos[i];
            float4& v = vel[i];
            Force   f = forces[i];

            v.x += dt_m * f.f.x;
            v.y += dt_m * f.f.y;
            v.z += dt_m * f.f.z;

            r.x += dt * v.x;
            r.y += dt * v.y;
            r.z += dt * v.z;
        }
    }
}

static void initializeParticles(ParticleVector *pv, std::vector<float4>& hostPositions, std::vector<float4>& hostVelocities)
{
    auto& pos = pv->local()->positions();
    auto& vel = pv->local()->velocities();
    
    for (int i = 0; i < pos.size(); ++i) {
        pos[i].x = drand48();
        pos[i].y = drand48();
        pos[i].z = drand48();

        vel[i].x = drand48();
        vel[i].y = drand48();
        vel[i].z = drand48();
    }
    pos.uploadToDevice(defaultStream);
    vel.uploadToDevice(defaultStream);

    hostPositions .resize(pos.size());
    hostVelocities.resize(vel.size());
    std::copy(pos.begin(), pos.end(), hostPositions .begin());
    std::copy(vel.begin(), vel.end(), hostVelocities.begin());
}

static void initializeForces(ParticleVector *pv, std::vector<Force>& hostForces)
{
    auto &forces = pv->local()->forces();
    hostForces.resize(forces.size());

    for (auto& f : hostForces) {
        f.f.x = drand48();
        f.f.y = drand48();
        f.f.z = drand48();
    }    

    CUDA_Check( cudaMemcpyAsync(forces.devPtr(), hostForces.data(),
                                forces.size() * sizeof(Force),
                                cudaMemcpyHostToDevice, defaultStream) );
}

static void computeError(int n,
                         const float4 *pos1, const float4 *vel1,
                         const float4 *pos2, const float4 *vel2,
                         double& l2, double& linf)
{
    l2 = 0;
    linf = -1;
    double dx, dy, dz, du, dv, dw;
    
    for (int i = 0; i < n; ++i) {
        auto r1 = pos1[i], r2 = pos2[i];
        auto v1 = vel1[i], v2 = vel2[i];

        dx = fabs(r1.x - r2.x);
        dy = fabs(r1.y - r2.y);
        dz = fabs(r1.z - r2.z);

        du = fabs(v1.x - v2.x);
        dv = fabs(v1.y - v2.y);
        dw = fabs(v1.z - v2.z);

        l2 += dx*dx + dy*dy + dz*dz;
        l2 += du*du + dv*dv + dw*dw;

        linf = std::max(linf, dx);
        linf = std::max(linf, dy);
        linf = std::max(linf, dz);

        linf = std::max(linf, du);
        linf = std::max(linf, dv);
        linf = std::max(linf, dw);
    }
    l2 = std::sqrt(l2);
}

static void testVelocityVerlet(float dt, float mass, int nparticles, int nsteps, double tolerance)
{
    double l2, linf;
    DomainInfo domain; // dummy domain
    YmrState state(domain, dt);
    
    auto vv = IntegratorFactory::createVV(&state, "vv");
    ParticleVector pv(&state, "pv", mass, nparticles);

    std::vector<float4> hostPositions, hostVelocities;
    std::vector<Force> hostForces;
    
    initializeParticles(&pv, hostPositions, hostVelocities);
    initializeForces(&pv, hostForces);
    
    run_gpu(vv.get(), &pv, nsteps, &state);
    run_cpu(hostPositions, hostVelocities, hostForces, nsteps, dt, mass);

    computeError(pv.local()->size(),
                 pv.local()->positions ().data(),
                 pv.local()->velocities().data(),
                 hostPositions.data(), hostVelocities.data(),
                 l2, linf);
    
    ASSERT_LE(l2, tolerance);
    ASSERT_LE(linf, tolerance);
}

TEST(Integration, velocityVerlet1)
{
    testVelocityVerlet(0.1, 1.0, 1000, 100, 5e-4);
}

TEST(Integration, velocityVerletSmallMass)
{
    testVelocityVerlet(0.1, 0.1, 1000, 100, 5e-3);
}

TEST(Integration, velocityVerletLargeMass)
{
    testVelocityVerlet(0.1, 10000.0, 1, 10000, 5e-5);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    logger.init(MPI_COMM_WORLD, "integration.log", 9);

    testing::InitGoogleTest(&argc, argv);

    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
