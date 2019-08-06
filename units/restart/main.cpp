#include <core/initial_conditions/uniform.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

Logger logger;
const std::string restartPath = "restart/";

constexpr int cartMaxdims = 3;
const int cartDims[] = {2, 2, 1}; 


static std::unique_ptr<ParticleVector> initializeRandomPV(const MPI_Comm& comm, const MirState *state, float density)
{
    float mass = 1;
    auto pv = std::make_unique<ParticleVector> (state, "pv", mass);
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
    auto comm = createCart();
    float dt = 0.f;
    float L = 64.f;
    float density = 4.f;
    DomainInfo domain = createDomainInfo(comm, {L, L, L});
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();

    auto backupData = lpv->dataPerParticle;

    constexpr int checkPointId = 0;
    pv->checkpoint(comm, restartPath, checkPointId);
    pv->restart   (comm, restartPath);

    compare(backupData, lpv->dataPerParticle);

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
    // TODO cart topology

    logger.init(MPI_COMM_WORLD, "restart.log", 3);

    testing::InitGoogleTest(&argc, argv);

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // only root listens to gtest
    if (getRank(MPI_COMM_WORLD) != 0)
        delete listeners.Release(listeners.default_result_printer());
    
    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
