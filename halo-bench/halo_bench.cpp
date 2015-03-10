#include <cstring>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <sys/time.h>
#include <time.h>
#include <sched.h>
using namespace std;

//#define USE_MPI_ISEND
//#define USE_MPI_SEND
//#define USE_MPI_ISSEND
//#define USE_MPI_SSEND
//#define USE_MPI_PERS_SEND
//#define USE_MPI_PERS_SSEND
//#define USE_MPI_RSEND

//#define USE_MPI_IRECV
//#define USE_MPI_PERS_RECV

#if 1
#include "hpm.cpp"
HPM hpm;
#endif

void *myalloc(size_t nbytes)
{
#if 1
	enum { alignment_bytes = 4096 } ;
	void * tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, nbytes);
	return tmp;
#else
	return malloc(nbytes);
#endif
}

double my_Wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec*1.0E-6;
}


#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS)
    {
     	char error_string[2048];
        int length_of_error_string = sizeof(error_string);
        MPI_Error_string(code, error_string, &length_of_error_string);

        printf("mpiAssert: %s %d %s\n", file, line, error_string);

        MPI_Abort(MPI_COMM_WORLD, code);
    }
}

class HaloExchanger
{
    MPI_Comm cartcomm;
    MPI_Request sendcountreq[26], recvcountreq[26];
    
    int recv_tags[26], recv_counts[26], nlocal;
    int send_counts[26];

    int Nint;
    int *send_buf[26], *recv_buf[26];
    MPI_Request sendreq[26], recvreq[26];

    bool firstpost;

    int use_prof;
    double local_duration, remote_duration, imbalance; 

protected:
    
    int L, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26];
    
    //plain copy of the offsets for the cpu (i speculate that reading multiple times the zero-copy entries is slower)
    int nsendreq;

    void post_expected_recv();
    
    //cuda-sync after to wait for packing of the halo, mpi non-blocking
    void pack_and_post(); //const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount);

    //mpi-sync for the surrounding halos, shift particle coord to the sysref of this rank
    void wait_for_messages();

    void spawn_local_work();
    void spawn_remote_work();

    const int basetag;

public:
    
    HaloExchanger(MPI_Comm cartcomm, int L, const int basetag);
    
    ~HaloExchanger();

    void exchange(); //Particle * const plocal, int nlocal, SimpleDeviceBuffer<Particle>& result);

    void set_prof(int flag);
    void set_local_duration(double t);
    void set_remote_duration(double t);
    void set_imbalance(double value);

};

HaloExchanger::HaloExchanger(MPI_Comm _cartcomm, int L, const int basetag):  L(L), basetag(basetag), firstpost(true)
{
    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i) {
	sendcountreq[i] = MPI_REQUEST_NULL;
	recvcountreq[i] = MPI_REQUEST_NULL;
	sendreq[i] = MPI_REQUEST_NULL;
	recvreq[i] = MPI_REQUEST_NULL;
    }

    Nint = L;
    for(int i = 0; i < 26; ++i) {
	send_buf[i] = (int *)myalloc(Nint*sizeof(int));
	recv_buf[i] = (int *)myalloc(Nint*sizeof(int));
    }

    for(int i = 0; i < 26; ++i) {
	for (int j = 0; j < Nint; j++) {
		send_buf[i][j] = myrank;
		recv_buf[i][j] = -1;
	}
    }

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
    }

    use_prof = 0;
    local_duration = 2300;
    remote_duration = 2300;
    imbalance = 0.04;
}

void HaloExchanger::pack_and_post() //const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount)
{
    if (firstpost) {
#if defined(USE_MPI_PERS_SEND)
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Send_init(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendcountreq + i) );
	}
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Send_init(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendreq + i) );
	}
#endif

#if defined(USE_MPI_PERS_SSEND)
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Ssend_init(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendcountreq + i) );
	}
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Ssend_init(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendreq + i) );
	}
#endif

#if defined(USE_MPI_PERS_RECV)
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Recv_init(recv_counts+i, 1, MPI_INT, dstranks[i], basetag +  recv_tags[i] + 150, cartcomm, recvcountreq + i) );
	}
	for(int i = 0; i < 26; ++i) {
		MPI_CHECK( MPI_Recv_init(recv_buf[i], Nint, MPI_INT, dstranks[i], basetag +  recv_tags[i] + 150, cartcomm, recvreq + i) );
	}
#endif
	post_expected_recv();


    }
    else
    {
	MPI_Status statuses[26 * 2];

	if (use_prof) hpm.HPM_Start("wait1");
	MPI_CHECK( MPI_Waitall(26, sendcountreq, statuses) );
	MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );
	if (use_prof) hpm.HPM_Stop("wait1");
    }
      
    for(int i = 0; i < 26; ++i)
	send_counts[i] = 1;


     spawn_local_work();
  
//   MPI_CHECK( MPI_Startall(26, sendcountreq + 0) );
	 
    for(int i = 0; i < 26; ++i) {
#if defined(USE_MPI_ISEND)
	MPI_CHECK( MPI_Isend(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendreq + i) );	
#elif defined(USE_MPI_SEND)
	MPI_CHECK( MPI_Send(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#elif defined(USE_MPI_ISSEND)
	MPI_CHECK( MPI_Issend(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendreq + i) );
#elif defined(USE_MPI_SSEND)
	MPI_CHECK( MPI_Ssend(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#elif defined(USE_MPI_PERS_SEND)||defined(USE_MPI_PERS_SSEND)
	MPI_CHECK( MPI_Start(sendreq + i) );
#elif defined(USE_MPI_RSEND)
	MPI_CHECK( MPI_Rsend(send_buf[i], Nint, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#else
#error	"MPI send method not defined!"
#endif

    }

    for(int i = 0; i < 26; ++i) {
#if defined(USE_MPI_ISEND)
	MPI_CHECK( MPI_Isend(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendcountreq + i) );	
#elif defined(USE_MPI_SEND)
	MPI_CHECK( MPI_Send(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#elif defined(USE_MPI_ISSEND)
	MPI_CHECK( MPI_Issend(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm, sendcountreq + i) );
#elif defined(USE_MPI_SSEND)
	MPI_CHECK( MPI_Ssend(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#elif defined(USE_MPI_PERS_SEND)||defined(USE_MPI_PERS_SSEND)
	MPI_CHECK( MPI_Start(sendcountreq + i) );
#elif defined(USE_MPI_RSEND)
	MPI_CHECK( MPI_Rsend(send_counts+i, 1, MPI_INT, dstranks[i], basetag +  i + 150, cartcomm) );
#else
#error	"MPI send method not defined!"
#endif

    }

    nsendreq = 0;
    
    firstpost = false;
}

void HaloExchanger::post_expected_recv()
{
//    MPI_Startall(26, recvcountreq);

    for(int i = 0; i < 26; ++i) {
#if defined(USE_MPI_IRECV)
	MPI_CHECK( MPI_Irecv(recv_buf[i], Nint, MPI_INT, dstranks[i], basetag + recv_tags[i] + 150, cartcomm, recvreq + i) );
#elif defined(USE_MPI_PERS_RECV)
	MPI_Start(recvreq+i);
#else
#error "MPI recv method not defined!" 
#endif
    }

    for(int i = 0; i < 26; ++i) {
#if defined(USE_MPI_IRECV)
	MPI_CHECK( MPI_Irecv(recv_counts+i, 1, MPI_INT, dstranks[i], basetag + recv_tags[i] + 150, cartcomm, recvcountreq + i) );
#elif defined(USE_MPI_PERS_RECV)
	MPI_Start(recvcountreq+i);
#else
#error "MPI recv method not defined!" 
#endif
    }


}

void HaloExchanger::wait_for_messages()
{
    {
	MPI_Status statuses[26];

	if (use_prof) hpm.HPM_Start("wait2");
	MPI_CHECK( MPI_Waitall(26, recvcountreq, statuses) );
	MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
	if (use_prof) hpm.HPM_Stop("wait2");

    }

#if DBG
    int chksum = 0;
#endif
    for(int i = 0; i < 26; ++i)
    {
//	const int count = recv_counts[i];
	const int count = recv_buf[i][0];
#if DBG
	chksum += count;
#endif
    }
#if DBG
    printf("myrank = %d chksum = %d\n", myrank, chksum); 
#endif
    post_expected_recv();
}

void HaloExchanger::exchange() //Particle * const plocal, int nlocal, SimpleDeviceBuffer<Particle>& retval)
{
    pack_and_post(); //plocal, nlocal, cells.start, cells.count);
    wait_for_messages();
    spawn_remote_work();
}

void HaloExchanger::spawn_local_work()
{
    double t0 = my_Wtime()*1e6; // current time in us
    double duration = local_duration;

    if (drand48() < 0.5)
	duration = duration * ( 1 + imbalance * drand48());
    else        
	duration = duration * ( 1 - imbalance * drand48());

    double t_end = t0 + duration;
    while (t0 < t_end) {
		sched_yield();
		t0 = my_Wtime()*1e6;
    }
}

void HaloExchanger::spawn_remote_work()
{
    double t0 = my_Wtime()*1e6; // current time in us
    double duration = remote_duration; //2300;

    if (drand48() < 0.5)
	duration = duration * ( 1 + imbalance * drand48());
    else        
	duration = duration * ( 1 - imbalance * drand48());

    double t_end = t0 + duration;
    while (t0 < t_end) {
		sched_yield();
		t0 = my_Wtime()*1e6;
    }
}

void HaloExchanger::set_prof(int flag)
{
	use_prof = 1;
}

void HaloExchanger::set_local_duration(double t)
{
	local_duration = t;
}

void HaloExchanger::set_remote_duration(double t)
{
	remote_duration = t;
}

void HaloExchanger::set_imbalance(double value)
{
	imbalance = value;
}

HaloExchanger::~HaloExchanger()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    if (!firstpost)
    {
	for(int i = 0; i < 26; ++i) {
		if (recvcountreq[i] != MPI_REQUEST_NULL) MPI_CHECK( MPI_Cancel(recvcountreq + i) );
		if (recvreq[i] != MPI_REQUEST_NULL) MPI_CHECK( MPI_Cancel(recvreq + i) );
	}
    }

    for(int i = 0; i < 26; ++i) {
	free(send_buf[i]);
	free(recv_buf[i]);
    }
}

int main(int argc, char *argv[])
{
    int ranks[3];
    int N = 1;
    double t0, t1;
    double local_wt = 2300, remote_wt = 2300, imbalance = 0;

    if ((argc != 4) && (argc != 5) && (argc != 6) && (argc != 7) && (argc != 8))
    {
     	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks> [<N>] [<local_time>] [<remote_time>] [<imbalance>]\n");
        exit(-1);
    }
    else {
        for(int i = 0; i < 3; ++i)
            ranks[i] = atoi(argv[1 + i]);

        if (argc >= 5) N = atoi(argv[4]); 
        if (argc >= 6) local_wt = atof(argv[5]); 
        if (argc >= 7) remote_wt = atof(argv[6]); 
        if (argc == 8) imbalance = atof(argv[7]); 
    }
	
	t0 = my_Wtime();
//	MPI_CHECK( MPI_Init(&argc, &argv) );
	int provided;
	MPI_CHECK( MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) );
	t1 = my_Wtime();

	int rank, nranks;

	MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

	if (rank == 0) {
		//printf("MPI initalization in %f secs\n", t1-t0); fflush(0);
	}


        srand48(rank);

	MPI_Comm cartcomm;
	int periods[] = {1, 1, 1};
	int reorder = 1;
	t0 = MPI_Wtime();
	MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, reorder, &cartcomm) );
	t1 = MPI_Wtime();

	if (rank == 0) {
		//printf("after cart creation in %f secs\n", t1-t0); fflush(0);
	}


	{
#if DBG
	int warmup_iter = 0, n_iter = 1e1;
#else
	int warmup_iter = 1e2, n_iter = 1e4;	// osu: 1e3, 1e5 for small and 1e1, 1e2 for large (>8192 bytes)
#endif
	double tt[n_iter]; 

	if (N > 10000) {
		warmup_iter = 1e1;
		n_iter = 1e3;
	}

	HaloExchanger haloex(cartcomm, N, 0);
	haloex.set_local_duration(local_wt);
	haloex.set_remote_duration(remote_wt);
	haloex.set_imbalance(imbalance);

	for (int i = 0; i < warmup_iter; i++) {
		haloex.exchange();
	}

	if (rank == 0) {
		printf("after warmup round\n"); fflush(0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	haloex.set_prof(1);
	double t0 = my_Wtime();
	hpm.HPM_Start("main");
	for (int i = 0; i < n_iter; i++) {
		double tt0 = my_Wtime();
		haloex.exchange();
		tt[i] = my_Wtime() - tt0;
	}
	hpm.HPM_Stop("main");
	double t1 = my_Wtime();
	double t_elapsed = t1-t0;

	std::vector<double> t_vec;
	for (int i=0; i<n_iter; i++)
                t_vec.push_back(1e6*tt[i]);

	sort(t_vec.begin(), t_vec.end());

        // percentiles to be selected
        const int I1 = n_iter*.1;
        const int I2 = n_iter*.5;
        const int I3 = n_iter*.9;

        // output 10th, 50th, 90th percentiles
	if (rank == 0)
		printf("%d elapsed = %lf ms (%lf us per iter) (%lf - %lf - %lf - %lf - %lf)\n",
			N, 1e3*t_elapsed, 1e6*t_elapsed/n_iter, t_vec[0], t_vec[I1], t_vec[I2], t_vec[I3], t_vec[t_vec.size()-1]);

	if (rank == 0)
		hpm.HPM_Stats();
	}

	MPI_Finalize();

	return 0;
}
