#include <plugins/simple_serializer.h>
#include <core/containers.h>
#include <core/logger.h>
#include <core/datatypes.h>

#include <vector>
#include <string>
#include <cstdio>
#include <typeinfo>

Logger logger;

void myassert(bool condition, const std::string& message)
{
    if (!condition)
    {
        printf("%s\n", message.c_str());
        fflush(stdout);
    }
        
    assert(condition);
}


template <class Cont, class Vec, typename Cmp>
void test(Vec vals, Cmp cmp)
{
    std::vector<char> buf;
    Cont dst;
    
    {
        Cont src(vals.size());
        for (int i=0; i<vals.size(); i++)
            src[i] = vals[i];
        
        SimpleSerializer::serialize(buf, src);
    }
    
    SimpleSerializer::deserialize(buf, dst);
    
    for (int i=0; i<vals.size(); i++)
        myassert(cmp(dst[i], vals[i]), "mismatch on " + std::to_string(i));
    
    printf("passed: container %s\n", typeid(dst).name());
}

int main()
{
    MPI_Init(nullptr, nullptr);
    logger.init(MPI_COMM_WORLD, stdout);
    
    test< std::vector<std::string> >( std::vector<std::string>{"density", "velocity"},
                                   [] (std::string a, std::string b) { return a == b; } );
    
    test< std::vector<Particle> >( std::vector<Particle>{ Particle({0,0,0,0}, {1,1,1,1}), Particle({3,3,3,3}, {5,5,5,5}) },
                                   [] (Particle& a, Particle& b) { return a.r.x == b.r.x; } );
    
    test< HostBuffer<Particle> >( std::vector<Particle>{ Particle({0,0,0,0}, {1,1,1,1}), Particle({3,3,3,3}, {5,5,5,5}) },
                                  [] (Particle& a, Particle& b) { return a.r.x == b.r.x; } );
    
    test< PinnedBuffer<double> >( std::vector<double>{1,2,3,3.14,-2.7128, 1e42},
                                   [] (double& a, double& b) { return a == b; } );
    
    float s1 = 1, s2 = 2, d1, d2;
    double s3 = 42.0, d3;
    std::vector<int> s4{2,3,4,5}, d4;
    std::vector<std::string> s5{"density", "velocity"},  d5;
    std::vector<char> buf;
    
    SimpleSerializer::serialize  (buf, s1,s2,s3,s4,s5);
    SimpleSerializer::deserialize(buf, d1,d2,d3,d4,d5);
    
    myassert(s1==d1, "mismatch on 1");
    myassert(s2==d2, "mismatch on 2");
    myassert(s3==d3, "mismatch on 3");
    
    for (int i=0; i<s4.size(); i++)
        myassert(s4[i]==d4[i], "mismatch on 4[" + std::to_string(i) + "]"); 
    
    for (int i=0; i<s5.size(); i++)
        myassert(s5[i]==d5[i], "mismatch on 5[" + std::to_string(i) + "]");
    
    printf("Combined passed\n");
    
    MPI_Finalize();
    
    return 0;
}
