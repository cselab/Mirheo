#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/initial_conditions/rigid.h>
#include <mirheo/core/initial_conditions/uniform.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/utils/cuda_common.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using namespace mirheo;

const std::string restartPath = "./"; // no need to create folder

constexpr int cartMaxdims = 3;
const int cartDims[] = {2, 2, 1}; // assume 4 nodes for this test
constexpr real mass = 1;

static std::unique_ptr<ParticleVector> initializeRandomPV(const MPI_Comm& comm,
                                                          const std::string& pvName,
                                                          const MirState *state,
                                                          real density)
{
    auto pv = std::make_unique<ParticleVector> (state, pvName, mass);
    UniformIC ic(density);
    ic.exec(comm, pv.get(), defaultStream);
    return pv;
}

inline MPI_Comm createCart(MPI_Comm comm = MPI_COMM_WORLD)
{
    const int periods[] = {1, 1, 1};
    constexpr int reorder = 0;

    MPI_Comm cart;
    MPI_Check( MPI_Cart_create(comm, cartMaxdims, cartDims, periods, reorder, &cart) );
    return cart;
}

inline void destroyCart(MPI_Comm& cart)
{
    MPI_Check( MPI_Comm_free(&cart) );
}


template<typename T>
inline void compare(const std::string& name, const PinnedBuffer<T>& a, const PinnedBuffer<T>& b)
{
    ASSERT_EQ(a.size(), b.size()) << "channel " << name << " have different sizes";
}

inline void compare(const std::string& name,
                    const DataManager::ChannelDescription& a,
                    const DataManager::ChannelDescription& b)
{
    ASSERT_EQ(a.persistence, b.persistence);
    ASSERT_EQ(a.shift, b.shift);

    mpark::visit([&](auto aPtr)
    {
        using T = typename std::remove_pointer<decltype(aPtr)>::type::value_type;
        ASSERT_TRUE(mpark::holds_alternative<PinnedBuffer<T>*>(b.varDataPtr)) << "channel " << name << ": containers have different types";
        compare(name, *aPtr, *mpark::get<PinnedBuffer<T>*>(b.varDataPtr));
    }, a.varDataPtr);
}

inline void compare(const DataManager& a, const DataManager& b)
{
    const auto& sca = a.getSortedChannels();
    const auto& scb = b.getSortedChannels();

    ASSERT_EQ(sca.size(), scb.size()) << "different number of channels";

    for (size_t i = 0; i < sca.size(); ++i)
    {
        const auto& cha = sca[i];
        compare(cha.first, *cha.second, b.getChannelDescOrDie(cha.first));
    }
}

TEST (RESTART, pv)
{
    const std::string pvName = "pv";
    auto comm = createCart();
    real dt = 0.f;
    real L = 64.f;
    real density = 4.f;
    DomainInfo domain = createDomainInfo(comm, {L, L, L});
    MirState state(domain, dt, UnitConversion{});
    auto pv0 = initializeRandomPV(comm, pvName, &state, density);
    auto pv1 = std::make_unique<ParticleVector> (&state, pvName, mass);

    auto backupData = pv0->local()->dataPerParticle;

    constexpr int checkPointId = 0;
    pv0->checkpoint(comm, restartPath, checkPointId);
    pv1->restart   (comm, restartPath);

    compare(backupData, pv1->local()->dataPerParticle);

    destroyCart(comm);
}

// rejection sampling for particles inside ellipsoid
static auto generateUniformEllipsoid(int n, real3 axes, long seed = 424242)
{
    std::vector<real3> pos;
    pos.reserve(n);

    Ellipsoid ell(axes);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(-axes.x, axes.x);
    std::uniform_real_distribution<real> dy(-axes.y, axes.y);
    std::uniform_real_distribution<real> dz(-axes.z, axes.z);

    while (static_cast<int>(pos.size()) < n)
    {
        const real3 r {dx(gen), dy(gen), dz(gen)};
        if (ell.inOutFunction(r) < 0.f)
            pos.push_back(r);
    }
    return pos;
}

static auto generateObjectComQ(int n, real3 L, long seed=12345)
{
    std::vector<ComQ> com_q;
    com_q.reserve(n);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(0.f, L.x);
    std::uniform_real_distribution<real> dy(0.f, L.y);
    std::uniform_real_distribution<real> dz(0.f, L.z);

    real3 com {0.f, 0.f, 0.f};
    for (int i = 0; i < n; ++i)
    {
        const real3 r {dx(gen), dy(gen), dz(gen)};
        const real4 q {1.f, 0.f, 0.f, 0.f};
        com += r;
        com_q.push_back({r, q});
    }
    com *= 1.0 / n;

    for (auto& rq : com_q)
        rq.r -= com;

    return com_q;
}

static std::unique_ptr<RigidShapedObjectVector<Ellipsoid>>
initializeRandomREV(const MPI_Comm& comm, const std::string& ovName, const MirState *state, int nObjs, int objSize)
{
    real3 axes {1.f, 1.f, 1.f};
    Ellipsoid ellipsoid(axes);

    auto rev = std::make_unique<RigidShapedObjectVector<Ellipsoid>>
        (state, ovName, mass, objSize, ellipsoid);

    auto com_q  = generateObjectComQ(nObjs, state->domain.globalSize);
    auto coords = generateUniformEllipsoid(objSize, axes);

    RigidIC ic(com_q, coords);
    ic.exec(comm, rev.get(), defaultStream);

    return rev;
}

TEST (RESTART, rov)
{
    const std::string rovName = "rov";
    auto comm = createCart();
    const real dt = 0.f;
    const real L = 64.f;
    const int nObjs = 512;
    const int objSize = 666;
    DomainInfo domain = createDomainInfo(comm, {L, L, L});
    MirState state(domain, dt, UnitConversion{});

    auto rov0 = initializeRandomREV(comm, rovName, &state, nObjs, objSize);
    auto rov1 = std::make_unique<RigidShapedObjectVector<Ellipsoid>> (&state, rovName, mass, objSize, Ellipsoid{{1.f, 1.f, 1.f}});

    auto backupDataParticles = rov0->local()->dataPerParticle;
    auto backupDataObjects   = rov0->local()->dataPerObject;

    constexpr int checkPointId = 0;
    rov0->checkpoint(comm, restartPath, checkPointId);
    rov1->restart   (comm, restartPath);

    compare(backupDataParticles, rov1->local()->dataPerParticle);
    compare(backupDataObjects,   rov1->local()->dataPerObject  );

    destroyCart(comm);
}

inline int getRank(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "restart.log", 3);

    testing::InitGoogleTest(&argc, argv);

    // auto& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // only root listens to gtest
    // if (getRank(MPI_COMM_WORLD) != 0)
    //     delete listeners.Release(listeners.default_result_printer());

    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
