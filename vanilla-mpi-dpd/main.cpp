#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <mpi.h>

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS) 
    {
	char error_string[2048];
	int length_of_error_string = sizeof(error_string);
	MPI_Error_string(code, error_string, &length_of_error_string);
	 
	printf("mpiAssert: %s %s %d\n", error_string, file, line);
	 	 
	MPI_Abort(MPI_COMM_WORLD, code);
    }
}

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

const float dt = 0.02;
const float tend = 100;
const float kBT = 0.1;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 2.5;
static const int tagbase_dpd_remote_interactions = 0;
static const int tagbase_redistribute_particles = 255;

//for now saru is at the CORE of our DPD interaction kernel.
//i use and misuse seed1-3 as i see fit, they are NOT always associated to the same things.
float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;
    
    int state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    int wstate = (state + seed2) ^ (((signed int)state)>>8);
    state  = state + (wstate*(wstate^0xdddf97f5));
    wstate = 0xABCB96F7 + (wstate>>1);
    
    state  = 0x4beb5d59*state + 0x2600e1f7; // LCG
    wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
    
    unsigned int v = (state ^ (state>>26))+wstate;
    unsigned int r = (v^(v>>20))*0x6957f5a7;
    
    float res = r / (4294967295.0f);
    
    return res;
}

//AoS is the dpd currency because of the spatial locality.
//AoS - SoA conversion might be performed within the hpc kernels.
struct Particle
{
    float x, y, z, u = 0, v = 0, w = 0;

    static bool initialized;
    static MPI_Datatype mytype;

    static MPI_Datatype datatype()
	{
	    if (!initialized)
	    {
		MPI_CHECK( MPI_Type_contiguous(6, MPI_FLOAT, &mytype));

		MPI_CHECK(MPI_Type_commit(&mytype));

		initialized = true;
	    }

	    return mytype;
	}
};

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

//why do i need this?
struct Acceleration
{
    float ax = 0, ay = 0, az = 0;
};

//local interactions deserve a kernel on their own, since they are expected to take most of the computational time.
//saru tag is there to prevent the realization of the same random force twice between different timesteps
void dpd_local_interactions(Particle * p, int n, int saru_tag, Acceleration * a)
{
    for(int i = 0; i < n; ++i)
    {
	float xf = 0, yf = 0, zf = 0;

	for(int j = 0; j < n; ++j)
	{
	    if (j == i)
		continue;
	    
	    float _xr = p[i].x - p[j].x;
	    float _yr = p[i].y - p[j].y;
	    float _zr = p[i].z - p[j].z;
       		    
	    float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	    float invrij = 1.f / sqrtf(rij2);

	    if (rij2 == 0)
		invrij = 100000;
	    
	    float rij = rij2 * invrij;
	    float wr = max((float)0, 1 - rij);
		    
	    float xr = _xr * invrij;
	    float yr = _yr * invrij;
	    float zr = _zr * invrij;
		
	    float rdotv = 
		xr * (p[i].u - p[j].u) +
		yr * (p[i].v - p[j].v) +
		zr * (p[i].w - p[j].w);
	    
	    float mysaru = saru(min(i, j), max(i, j), saru_tag);
	    
	    float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
	    float strength = (aij - gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

	    xf += strength * xr;
	    yf += strength * yr;
	    zf += strength * zr;
	}

	a[i].ax += xf;
	a[i].ay += yf;
	a[i].az += zf;
    }
}

void dpd_bipartite_kernel(Particle * pdst, int ndst, Particle * psrc, int nsrc, int saru_tag1, int saru_tag2, int saru_mask, Acceleration * a)
{    
    for(int i = 0; i < ndst; ++i)
    {
	float xf = 0, yf = 0, zf = 0;

	for(int j = 0; j < nsrc; ++j)
	{
	    float _xr = pdst[i].x - psrc[j].x;
	    float _yr = pdst[i].y - psrc[j].y;
	    float _zr = pdst[i].z - psrc[j].z;
		    
	    float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	    float invrij = 1.f / sqrtf(rij2);

	    if (rij2 == 0)
		invrij = 100000;
	    
	    float rij = rij2 * invrij;
	    float wr = max((float)0, 1 - rij);
		    
	    float xr = _xr * invrij;
	    float yr = _yr * invrij;
	    float zr = _zr * invrij;
		
	    float rdotv = 
		xr * (pdst[i].u - psrc[j].u) +
		yr * (pdst[i].v - psrc[j].v) +
		zr * (pdst[i].w - psrc[j].w);
	    
	    float mysaru = saru(saru_tag1, saru_tag2, saru_mask ? i + ndst * j : j + nsrc * i);
	    
	    float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
	    float strength = (aij - gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

	    xf += strength * xr;
	    yf += strength * yr;
	    zf += strength * zr;
	}

	a[i].ax += xf;
	a[i].ay += yf;
	a[i].az += zf;
    }
}

void dpd_remote_interactions(MPI_Comm cartcomm, Particle * p, int n, int L, int saru_tag1, Acceleration * a)
{
    assert(L % 2 == 0);
    
    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
				    
    vector<Particle> mypacks[26];
    vector<int> myentries[26];
    MPI_Request sendreq[26];
    
    //collect my halo particles into packs. send them to the surrounding ranks.
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int halo_start[3];
	for(int c = 0; c < 3; ++c)
	    halo_start[c] = max(d[c] * L - L/2 - 1, -L/2);
	
	int halo_end[3];
	for(int c = 0; c < 3; ++c)
	    halo_end[c] = min(d[c] * L + L/2 + 1, L/2);
	
	for(int j = 0; j < n; ++j)
	    if (p[j].x >= halo_start[0] && p[j].x < halo_end[0] &&
		p[j].y >= halo_start[1] && p[j].y < halo_end[1] &&
		p[j].z >= halo_start[2] && p[j].z < halo_end[2] )
	    {
		mypacks[i].push_back(p[j]);
		myentries[i].push_back(j);
	    }
	
	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	int dstrank;
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, &dstrank) );
	
	MPI_CHECK( MPI_Issend(&mypacks[i].front(), mypacks[i].size(), Particle::datatype(), dstrank,
			      tagbase_dpd_remote_interactions + i, cartcomm, sendreq + i) );
    }

    //get remote particle packs from surrounding ranks.
    vector<Particle> srcpacks[26];
    MPI_Request recvreq[26];
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int tag = tagbase_dpd_remote_interactions + (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	
	MPI_Status status;
	MPI_CHECK( MPI_Probe(MPI_ANY_SOURCE, tag, cartcomm, &status) );

	int count;
	MPI_CHECK( MPI_Get_count(&status, Particle::datatype(), &count) );

	srcpacks[i].resize(count);
	
	MPI_CHECK( MPI_Irecv(&srcpacks[i].front(), count, Particle::datatype(), MPI_ANY_SOURCE, tag, cartcomm, recvreq + i) );
    }

    //we want to keep it simple. that's why.
    MPI_Status statuses[26];
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
    MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );

    int myrank, nranks;
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    //compute saru tags
    int saru_tag2[26], saru_mask[26];
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

	int indx[3];
	for(int c = 0; c < 3; ++c)
	    indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] + max(coords[c], coordsneighbor[c]);

	saru_tag2[i] = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

	int dstrank;
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, &dstrank) );

	saru_mask[i] = min(dstrank, myrank) == myrank;
    }

    //compute interactions with the remote particle packs,
    //after properly shifting them to my system of reference
    vector<Acceleration> apacks[26];
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	for(int j = 0; j < srcpacks[i].size(); ++j)
	{
	    srcpacks[i][j].x += d[0] * L;
	    srcpacks[i][j].y += d[1] * L;
	    srcpacks[i][j].z += d[2] * L;
	}

	int npack = mypacks[i].size();
	
	apacks[i].resize(npack);
	for(auto& e: apacks[i])
	    e.ax = e.ay = e.az = 0;
	
	dpd_bipartite_kernel(&mypacks[i].front(), npack, &srcpacks[i].front(), srcpacks[i].size(),
			     saru_tag1, saru_tag2[i], saru_mask[i], &apacks[i].front());
    }

    //blend the freshly computed partial results to my local acceleration vector.
    for(int i = 0; i < 26; ++i)
    {
	Particle * ppack = &mypacks[i].front();
	
	for(int j = 0; j < mypacks[i].size() ; ++j)
	{
	    int entry = myentries[i][j];
	    
	    a[entry].ax += apacks[i][j].ax;
	    a[entry].ay += apacks[i][j].ay;
	    a[entry].az += apacks[i][j].az;
	}
    }
}

//velocity verlet stages
void update_stage1(Particle * p, Acceleration * a, int n, float dt)
{
    for(int i = 0; i < n; ++i)
    {
	p[i].u += a[i].ax * dt * 0.5;
	p[i].v += a[i].ay * dt * 0.5;
	p[i].w += a[i].az * dt * 0.5;

	p[i].x += p[i].u * dt;
	p[i].y += p[i].v * dt;
	p[i].z += p[i].w * dt;
    }
}

void update_stage2(Particle * p, Acceleration * a, int n, float dt)
{
    for(int i = 0; i < n; ++i)
    {
	p[i].u += a[i].ax * dt * 0.5;
	p[i].v += a[i].ay * dt * 0.5;
	p[i].w += a[i].az * dt * 0.5;
    }
}

void diagnostics(MPI_Comm comm, Particle * particles, int n, float dt, int idstep)
{
    int nlocal = n;
    
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
    {
	p[0] += particles[i].u;
	p[1] += particles[i].v;
	p[2] += particles[i].w;
    }

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, &p, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    double ke = 0;
    for(int i = 0; i < n; ++i)
	ke += pow(particles[i].u, 2) + pow(particles[i].v, 2) + pow(particles[i].w, 2);

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    double kbt = 0.5 * ke / (n * 3. / 2);
    
    if (rank == 0)
    {
	FILE * f = fopen("diag.txt", idstep ? "a" : "w");

	if (idstep == 0)
	    fprintf(f, "TSTEP\tKBT\tPX\tPY\tPZ\n");
	
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
	
	fclose(f);
    }
    
    //output VMD file
    {
	std::stringstream ss;

	if (rank == 0)
	{
	    ss << n << "\n";
	    ss << "dpdparticle\n";
	}
    
	for(int i = 0; i < nlocal; ++i)
	    ss << "1 " << particles[i].x << " " << particles[i].y << " " << particles[i].z << "\n";

	string content = ss.str();

	int len = content.size();
	int offset;
	MPI_CHECK( MPI_Exscan(&len, &offset, 1, MPI_INTEGER, MPI_SUM, comm)); 
	
	MPI_File f;
	char fn[] = "trajectories.xyz";
	MPI_CHECK( MPI_File_open(comm, fn, MPI_MODE_WRONLY | (idstep == 0 ? MPI_MODE_CREATE : MPI_MODE_APPEND), MPI_INFO_NULL, &f) );

	if (idstep == 0)
	    MPI_CHECK( MPI_File_set_size (f, 0));

	MPI_Offset base;
	MPI_CHECK( MPI_File_get_position(f, &base));
	
	MPI_Status status;
	MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.data()), len, MPI_CHAR, &status));
	
	MPI_CHECK( MPI_File_close(&f));
    }
}

//particles may fall outside my subdomain. i might loose old particles and receive new ones.
//redistribution is performed in 2 stages and unfortunately the first stage is stateful (tmp vector here below)
//that's why i need a class.
class RedistributeParticles
{
    MPI_Comm cartcomm;
    
    int L, myrank, dims[3], periods[3], coords[3], rankneighbors[27], domain_extent[3];
    int leaving_start[28], arriving_start[28], notleaving, arriving;

    vector<Particle> tmp;

    MPI_Request sendreq[27], recvreq[27];
    
public:
    
    RedistributeParticles(MPI_Comm cartcomm, int L): cartcomm(cartcomm), L(L)
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
    
    int stage1(Particle * p, int n)
	{
	    vector<int> myentries[27];
	    
	    for(int i = 0; i < n; ++i)
	    {
		int xcode = (2 + (p[i].x >= -L/2) + (p[i].x >= L/2)) % 3;
		int ycode = (2 + (p[i].y >= -L/2) + (p[i].y >= L/2)) % 3;
		int zcode = (2 + (p[i].z >= -L/2) + (p[i].z >= L/2)) % 3;
		int code = xcode + 3 * (ycode + 3 * zcode);

		myentries[code].push_back(i);

		assert(p[i].x >= -L/2 - L && p[i].x < L/2 + L);
		assert(p[i].y >= -L/2 - L && p[i].y < L/2 + L);
		assert(p[i].z >= -L/2 - L && p[i].z < L/2 + L);
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
	  
	    for(int i = 1; i < 27; ++i)
		MPI_CHECK( MPI_Issend(&tmp.front() + leaving_start[i], leaving_start[i + 1] - leaving_start[i],
				     Particle::datatype(), rankneighbors[i], tagbase_redistribute_particles + i, cartcomm, sendreq + i) );
    
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

    int stage2(Particle * p, int n)
	{
	    assert(n == notleaving + arriving);

	    copy(tmp.begin(), tmp.begin() + notleaving, p);

	    for(int i = 1; i < 27; ++i)
		MPI_CHECK( MPI_Irecv(p + arriving_start[i], arriving_start[i + 1] - arriving_start[i], Particle::datatype(),
				     MPI_ANY_SOURCE, tagbase_redistribute_particles + i, cartcomm, recvreq + i) );

	    MPI_Status statuses[26];	    
	    MPI_CHECK( MPI_Waitall(26, recvreq + 1, statuses) );
	    MPI_CHECK( MPI_Waitall(26, sendreq + 1, statuses) );

	    for(int i = 1; i < 27; ++i)
	    {
		int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
	    
		for(int j = arriving_start[i]; j < arriving_start[i + 1]; ++j)
		{
		    p[j].x -= d[0] * L;
		    p[j].y -= d[1] * L;
		    p[j].z -= d[2] * L;
		}
	    }
	    
	    for(int i = 0; i < n; ++i)
	    {	
		int xcode = (2 + (p[i].x >= -L/2) + (p[i].x >= L/2)) % 3;
		int ycode = (2 + (p[i].y >= -L/2) + (p[i].y >= L/2)) % 3;
		int zcode = (2 + (p[i].z >= -L/2) + (p[i].z >= L/2)) % 3;
		int code = xcode + 3 * (ycode + 3 * zcode);

		assert(code == 0);
	    }
	}
};

int main(int argc, char ** argv)
{
    int ranks[3];
    
    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    {
	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);
    }
    
    MPI_CHECK( MPI_Init(&argc, &argv) );

    int nranks, rank;
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

    MPI_Comm cartcomm;
    
    int periods[] = {1,1,1};
    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );

    const int L = 4;

    vector<Particle> particles(L * L * L * 3);

    for(auto& p : particles)
    {
	p.x = -L * 0.5 + drand48() * L;
	p.y = -L * 0.5 + drand48() * L;
	p.z = -L * 0.5 + drand48() * L;
    }

    RedistributeParticles redistribute(cartcomm, L);
    
    const size_t nsteps = (int)(tend / dt);

    int domainsize[3];
    for(int i = 0; i < 3; ++i)
	domainsize[i] = L * ranks[i];

    vector<Acceleration> accel(particles.size());

    int saru_tag = rank;
    
    dpd_local_interactions(&particles.front(), particles.size(), saru_tag, &accel.front());
    saru_tag += nranks - rank;
    
    dpd_remote_interactions(cartcomm, &particles.front(), particles.size(), L, saru_tag, &accel.front());
    saru_tag += 1 + rank;
    
    for(int it = 0; it < nsteps; ++it)
    {
	if (rank == 0)
	    printf("beginning of time step %d for rank %d\n", it, rank);
	
	update_stage1(&particles.front(), &accel.front(), particles.size(), dt);
	
	int newnp = redistribute.stage1(&particles.front(), particles.size());

	particles.resize(newnp);
	accel.resize(newnp);

	redistribute.stage2(&particles.front(), particles.size());
	
	for(auto& a : accel)
	    a.ax = a.ay = a.az = 0;
	
	dpd_local_interactions(&particles.front(), particles.size(), saru_tag, &accel.front());
	 saru_tag += nranks - rank;
	
	dpd_remote_interactions(cartcomm, &particles.front(), particles.size(), L, saru_tag, &accel.front());
	saru_tag += 1 + rank;

	update_stage2(&particles.front(), &accel.front(), particles.size(), dt);

	diagnostics(cartcomm, &particles.front(), particles.size(), dt, it);
    }
    
    MPI_CHECK( MPI_Finalize() );

    if (rank == 0)
	printf("simulation is done\n");
    
    return 0;
}
	
