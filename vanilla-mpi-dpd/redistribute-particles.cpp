#include <cassert>
#include <vector>
#include <algorithm>

#include "redistribute-particles.h"

using namespace std;

RedistributeParticles::RedistributeParticles(MPI_Comm cartcomm, int L): cartcomm(cartcomm), L(L)
{
    assert(L % 2 == 0);
	    
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
	    
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
	    
    for(int c = 0; c < 3; ++c)
	domain_extent[c] = L * dims[c];

    rankneighbors[0] = myrank;
    for(int i = 1; i < 27; ++i)
    {
	int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
	
	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];
		
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, rankneighbors + i) );
    }
}
    
int RedistributeParticles::stage1(Particle * p, int n)
{
    //naive way of performing reordering of particles
    //this will be converted into a CUDA KERNEL in the non-vanilla version
    {
	vector<int> myentries[27];

	for(int i = 0; i < n; ++i)
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p[i].x[c] >= -L/2) + (p[i].x[c] >= L/2)) % 3;
		
	    int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    myentries[code].push_back(i);

	    for(int c = 0; c < 3; ++c)
		assert(p[i].x[c] >= -L/2 - L && p[i].x[c] < L/2 + L);
	}
	    
	notleaving = myentries[0].size();

	tmp.resize(n);
	    
	for(int i = 0, c = 0; i < 27; ++i)
	{
	    leaving_start[i] = c;
	
	    for(int j = 0; j < myentries[i].size(); ++j, ++c)
		tmp[c] = p[myentries[i][j]];
	}
	    
	leaving_start[27] = n;
    }
	  
    MPI_Status statuses[26];
    if (pending_send)	    
	MPI_CHECK( MPI_Waitall(26, sendreq + 1, statuses) );

    //in the non-vanilla version will use GPUDirect RDMA here
    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Isend(&tmp.front() + leaving_start[i], leaving_start[i + 1] - leaving_start[i],
			     Particle::datatype(), rankneighbors[i], tagbase_redistribute_particles + i, cartcomm, sendreq + i) );
    
    pending_send = true;

    //prepare offsets for the new particles landing in my subdomain
    arriving = 0;
    arriving_start[0] = notleaving;
    for(int i = 1; i < 27; ++i)
    {
	MPI_Status status;
	MPI_CHECK( MPI_Probe(MPI_ANY_SOURCE, tagbase_redistribute_particles + i, cartcomm, &status) );
		
	int local_count;
	MPI_CHECK( MPI_Get_count(&status, Particle::datatype(), &local_count) );

	arriving_start[i] = notleaving + arriving;
	arriving += local_count;
    }
	    
    arriving_start[27] = notleaving + arriving;

    return notleaving + arriving;
}

void RedistributeParticles::stage2(Particle * p, int n)
{
    assert(n == notleaving + arriving);

    copy(tmp.begin(), tmp.begin() + notleaving, p);

    //in the non-vanilla version will use GPUDirect RDMA here
    //and this recv requests will be moved into stage1 but the receiving buffer would be another tmp vector (tmp2)
    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Irecv(p + arriving_start[i], arriving_start[i + 1] - arriving_start[i], Particle::datatype(),
			     MPI_ANY_SOURCE, tagbase_redistribute_particles + i, cartcomm, recvreq + i) );

    MPI_Status statuses[26];	    
    MPI_CHECK( MPI_Waitall(26, recvreq + 1, statuses) );
    
    //change the system of reference of the particles
    //in the non-vanilla version this will be a CUDA KERNEL
    {
	for(int i = 1; i < 27; ++i)
	{
	    int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
	    
	    for(int j = arriving_start[i]; j < arriving_start[i + 1]; ++j)
		for(int c = 0; c < 3; ++c)
		    p[j].x[c] -= d[c] * L;
	}
	    
	for(int i = 0; i < n; ++i)
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p[i].x[c] >= -L/2) + (p[i].x[c] >= L/2)) % 3;
		
	    int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    assert(code == 0);
	}
    }
}

