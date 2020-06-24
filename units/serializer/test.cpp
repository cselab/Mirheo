#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>
#include <mirheo/plugins/utils/simple_serializer.h>

#include <vector>
#include <string>
#include <cstdio>
#include <typeinfo>

#include <gtest/gtest.h>

using namespace mirheo;

template <class Cont, class Vec, typename Cmp>
void test(Vec vals, Cmp cmp)
{
    std::vector<char> buf;
    Cont dst;

    {
        Cont src(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
            src[i] = vals[i];

        SimpleSerializer::serialize(buf, src);
    }

    SimpleSerializer::deserialize(buf, dst);

    for (size_t i = 0; i < vals.size(); ++i)
        ASSERT_TRUE(cmp(dst[i], vals[i])) << "mismatch on " + std::to_string(i);
}

TEST(Serializer, VectorOfString)
{
    test< std::vector<std::string> >( std::vector<std::string>{"density", "velocity"},
                                   [] (std::string a, std::string b) { return a == b; } );
}

TEST(Serializer, VectorOfParticle)
{
    test< std::vector<Particle> >( std::vector<Particle>{ Particle({0,0,0,0}, {1,1,1,1}), Particle({3,3,3,3}, {5,5,5,5}) },
                                   [] (Particle& a, Particle& b) { return a.r.x == b.r.x; } );
}

TEST(Serializer, HostBufferOfParticle)
{
    test< HostBuffer<Particle> >( std::vector<Particle>{ Particle({0,0,0,0}, {1,1,1,1}), Particle({3,3,3,3}, {5,5,5,5}) },
                                  [] (Particle& a, Particle& b) { return a.r.x == b.r.x; } );
}

TEST(Serializer, PinnedBufferOfDouble)
{
    test< PinnedBuffer<double> >( std::vector<double>{1,2,3,3.14,-2.7128, 1e42},
                                  [] (double& a, double& b) { return a == b; } );
}

TEST(Serializer, Mixed)
{
    float s1 = 1, s2 = 2, d1, d2;
    double s3 = 42.0, d3;
    std::vector<int> s4{2,3,4,5}, d4;
    std::vector<std::string> s5{"density", "velocity"},  d5;
    std::vector<char> buf;

    SimpleSerializer::serialize  (buf, s1,s2,s3,s4,s5);
    SimpleSerializer::deserialize(buf, d1,d2,d3,d4,d5);

    ASSERT_EQ(s1, d1) << "mismatch on 1";
    ASSERT_EQ(s2, d2) << "mismatch on 2";
    ASSERT_EQ(s3, d3) << "mismatch on 3";

    for (size_t i = 0; i < s4.size(); i++)
        ASSERT_EQ(s4[i], d4[i]) << "mismatch on 4[" + std::to_string(i) + "]";

    for (size_t i = 0; i < s5.size(); i++)
        ASSERT_EQ(s5[i], d5[i]) << "mismatch on 5[" + std::to_string(i) + "]";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    logger.init(MPI_COMM_WORLD, "serializer.log", 9);

    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
