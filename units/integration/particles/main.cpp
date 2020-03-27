#include <mirheo/core/integrators/factory.h>
#include <mirheo/core/logger.h>

#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using namespace mirheo;

static void run_gpu(Integrator *integrator, ParticleVector *pv, int nsteps, MirState *state)
{
    integrator->setPrerequisites(pv);
    
    for (int i = 0; i < nsteps; ++i) {
        state->currentStep = i;
        state->currentTime = i * state->dt;
        
        integrator->execute(pv, defaultStream);
    }

    pv->local()->positions ().downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    pv->local()->velocities().downloadFromDevice(defaultStream, ContainersSynch::Synch);
}

static void run_cpu(std::vector<real4>& pos, std::vector<real4>& vel,
                    const std::vector<Force>& forces, int nsteps, real dt, real mass)
{
    real dt_m = dt / mass;
    
    for (int step = 0; step < nsteps; ++step) {
        for (size_t i = 0; i < pos.size(); ++i) {
            real4& r = pos[i];
            real4& v = vel[i];
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

static std::tuple<std::vector<real4>, std::vector<real4>>
initializeParticles(ParticleVector *pv)
{
    std::vector<real4> hostPositions, hostVelocities;
    
    auto& pos = pv->local()->positions();
    auto& vel = pv->local()->velocities();
    
    for (size_t i = 0; i < pos.size(); ++i)
    {
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

    return {std::move(hostPositions), std::move(hostVelocities)};
}

static std::vector<Force> initializeForces(ParticleVector *pv)
{
    auto &forces = pv->local()->forces();

    for (auto& f : forces)
    {
        f.f.x = drand48();
        f.f.y = drand48();
        f.f.z = drand48();
    }    

    forces.uploadToDevice(defaultStream);
    
    return {forces.begin(), forces.end()};
}

static std::tuple<double, double>
computeError(int n,
             const real4 *pos1, const real4 *vel1,
             const real4 *pos2, const real4 *vel2)
{
    double l2 {0.};
    double linf {-1.};
    double dx, dy, dz, du, dv, dw;
    
    for (int i = 0; i < n; ++i) {
        auto r1 = pos1[i], r2 = pos2[i];
        auto v1 = vel1[i], v2 = vel2[i];

        dx = math::abs(r1.x - r2.x);
        dy = math::abs(r1.y - r2.y);
        dz = math::abs(r1.z - r2.z);

        du = math::abs(v1.x - v2.x);
        dv = math::abs(v1.y - v2.y);
        dw = math::abs(v1.z - v2.z);

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
    return {l2, linf};
}

static void testVelocityVerlet(real dt, real mass, int nparticles, int nsteps, double tolerance)
{
    double l2, linf;
    DomainInfo domain; // dummy domain
    MirState state(domain, dt, UnitConversion{});
    
    auto vv = integrator_factory::createVV(&state, "vv");
    ParticleVector pv(&state, "pv", mass, nparticles);

    std::vector<real4> hostPositions, hostVelocities;
    
    std::tie(hostPositions, hostVelocities) = initializeParticles(&pv);
    const auto hostForces = initializeForces(&pv);
    
    run_gpu(vv.get(), &pv, nsteps, &state);
    run_cpu(hostPositions, hostVelocities, hostForces, nsteps, dt, mass);

    std::tie(l2, linf) = computeError(pv.local()->size(),
                                      pv.local()->positions ().data(),
                                      pv.local()->velocities().data(),
                                      hostPositions.data(), hostVelocities.data());
    
    ASSERT_LE(l2, tolerance);
    ASSERT_LE(linf, tolerance);
}

TEST(Integration_particles,velocityVerlet)
{
    testVelocityVerlet(0.1, 1.0, 1000, 100, 5e-4);
}

TEST(Integration_particles, velocityVerletSmallMass)
{
    testVelocityVerlet(0.1, 0.1, 1000, 100, 5e-3);
}

TEST(Integration_particles, velocityVerletLargeMass)
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
