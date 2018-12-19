#include <core/integrators/factory.h>
#include <core/logger.h>

#include <gtest/gtest.h>
#include <vector>

Logger logger;
cudaStream_t defaultStream = 0;

static void run_gpu(Integrator *integrator, ParticleVector *pv, int nsteps, float dt)
{
    integrator->setPrerequisites(pv);
    
    for (int i = 0; i < nsteps; ++i) {        
        integrator->stage1(pv, i * dt, defaultStream);
        integrator->stage2(pv, i * dt, defaultStream);
    }

    pv->local()->coosvels.downloadFromDevice(defaultStream, ContainersSynch::Synch);
}

static void run_cpu(std::vector<Particle>& particles, std::vector<Force>& forces, int nsteps, float dt, float mass)
{
    float dt_m = dt / mass;
    
    for (int step = 0; step < nsteps; ++step) {
        for (int i = 0; i < particles.size(); ++i) {
            Particle& p = particles[i];
            const Force f = forces[i];

            p.u.x += dt_m * f.f.x;
            p.u.y += dt_m * f.f.y;
            p.u.z += dt_m * f.f.z;

            p.r.x += dt * p.u.x;
            p.r.y += dt * p.u.y;
            p.r.z += dt * p.u.z;
        }
    }
}

static void initializeParticles(ParticleVector *pv, std::vector<Particle>& hostParticles)
{
    auto& coosvels = pv->local()->coosvels;
    
    for (auto& p : coosvels) {
        p.r.x = drand48();
        p.r.y = drand48();
        p.r.z = drand48();

        p.u.x = drand48();
        p.u.y = drand48();
        p.u.z = drand48();
    }
    coosvels.uploadToDevice(defaultStream);

    hostParticles.resize(coosvels.size());
    std::copy(coosvels.begin(), coosvels.end(), hostParticles.begin());
}

static void initializeForces(ParticleVector *pv, std::vector<Force>& hostForces)
{
    auto &forces = pv->local()->forces;
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

static void computeError(int n, const Particle *parts1, const Particle *parts2,
                         double& l2, double& linf)
{
    l2 = 0;
    linf = -1;
    double dx, dy, dz, du, dv, dw;
    
    for (int i = 0; i < n; ++i) {
        auto p1 = parts1[i], p2 = parts2[i];

        dx = fabs(p1.r.x - p2.r.x);
        dy = fabs(p1.r.y - p2.r.y);
        dz = fabs(p1.r.z - p2.r.z);

        du = fabs(p1.u.x - p2.u.x);
        dv = fabs(p1.u.y - p2.u.y);
        dw = fabs(p1.u.z - p2.u.z);

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
    
    Integrator *vv = IntegratorFactory::createVV("vv", dt);
    ParticleVector pv("pv", mass, nparticles);

    std::vector<Particle> hostParticles;
    std::vector<Force> hostForces;
    
    initializeParticles(&pv, hostParticles);
    initializeForces(&pv, hostForces);
    
    run_gpu(vv, &pv, nsteps, dt);
    run_cpu(hostParticles, hostForces, nsteps, dt, mass);

    computeError(pv.local()->size(), pv.local()->coosvels.data(),
                 hostParticles.data(), l2, linf);
    
    ASSERT_LE(l2, tolerance);
    ASSERT_LE(linf, tolerance);
    
    delete vv;
}

TEST(Integration, velocityVerlet1)
{
    testVelocityVerlet(0.1, 1.0, 1000, 100, 5e-4);
}

TEST(Integration, velocityVerletMass)
{
    testVelocityVerlet(0.1, 0.1, 1000, 100, 5e-3);
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
