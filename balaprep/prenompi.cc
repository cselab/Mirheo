/*
 *  prenompi.cc
 *  Part of uDeviceX/balaprep/
 *
 *  Created and authored by Massimo Bernaschi on 2015-02-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <csignal>
#include <errno.h>
#include <vector>
#include <map>

static float sysL[3];

using namespace std;

template<int k>
struct Bspline
{
    template<int i>
    static float eval(float x)
        {
            return
                (x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
                (i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
        }
};

template<>
struct Bspline<1>
{
    template <int i>
    static float eval(float x)
        {
            return  (float)(i) <= x && x < (float)(i + 1);
        }
};

typedef struct workxrank { int rank; int pad; double work; } workxrank_t;
int compwork(const void *a, const void *b) {
    double worka=((workxrank_t *)a)->work;
    double workb=((workxrank_t *)b)->work;
    if(worka<workb) {
                    return 1;
    } else if(workb<worka) {
                    return -1;
    } else {
                    return 0;
    }

}
#define INPUTFILE "sdf.dat"
workxrank_t *reorder(int *ranks) {
  int lrank, ldims[3], lperiods[3], lcoords[3];
  int locL[3];
  static int N[3];
  static float *data=NULL;
  int retval;
  int tranks=ranks[0]*ranks[1]*ranks[2];

  if(data==NULL) {
    FILE * f = fopen(INPUTFILE, "rb");
    if(f==NULL) {
      fprintf(stderr,"Could not open file %s\n",INPUTFILE);
      exit(1);
    }
    retval = fscanf(f, "%f %f %f\n", sysL + 0, sysL + 1, sysL + 2);
    assert(retval == 3);
    retval = fscanf(f, "%d %d %d\n", N + 0, N + 1, N + 2);
    assert(retval == 3);
    const int nvoxels = N[0] * N[1] * N[2];
    data = new float[nvoxels];

    retval = fread(data, sizeof(float), nvoxels, f);
    assert(retval == nvoxels);

    fclose(f);
    return NULL;
  }

#if 1 /* peh */
  printf("sysL = %f %f %f\n",  sysL[0],  sysL[1],  sysL[2]);
  printf("r(sysL)= %f %f %f\n",  roundf(sysL[0]),  roundf(sysL[1]),  roundf(sysL[2]));
  printf("rank = %d %d %d\n", ranks[0], ranks[1], ranks[2]);
#endif

  locL[0]=roundf(sysL[0])/ranks[0]; locL[1]=roundf(sysL[1])/ranks[1]; locL[2]=roundf(sysL[2])/ranks[2];

  printf("locL = %f %f %f\n",  locL[0],  locL[1],  locL[2]);
  for(int c=0; c<3; c++) {
    if(locL[c]%2 || locL[c]*ranks[c]<sysL[c]) {
#if 0 /* peh */
      printf("Invalid conf %d %d %d\n",ranks[0],ranks[1],ranks[2]);
      return NULL;
#endif
    }
  }
  double *rbuftval=new double[tranks];
  for(int r=0; r<tranks; r++) {
    lcoords[0]=(r/ranks[2]/ranks[1])%ranks[0];
    lcoords[1]=(r/ranks[2])%ranks[1];
    lcoords[2]=r%ranks[2];
    float start[3], spacing[3];
    for(int c = 0; c < 3; ++c)
      {
        ldims[c]=ranks[c];
        start[c] = lcoords[c] * locL[c] / (float)(ldims[c] * locL[c]) * N[c];
        spacing[c] = N[c] / (float)(ldims[c] * locL[c]) ;
      }

    int nsize[3] = {locL[0], locL[1], locL[2]};

    Bspline<4> bsp;
    double totval=0.;
    for(int iz = 0; iz < nsize[2]; ++iz)
      for(int iy = 0; iy < nsize[1]; ++iy)
        for(int ix = 0; ix < nsize[0]; ++ix)
          {
            const float x[3] = {
              start[0] + (ix  + 0.5f) * spacing[0] - 0.5f,
              start[1] + (iy  + 0.5f) * spacing[1] - 0.5f,
              start[2] + (iz  + 0.5f) * spacing[2] - 0.5f
            };

            int anchor[3];
            for(int c = 0; c < 3; ++c)
              anchor[c] = (int)floor(x[c]);

            float w[3][4];
            for(int c = 0; c < 3; ++c)
              for(int i = 0; i < 4; ++i)
                w[c][i] = bsp.eval<0>(x[c] - (anchor[c] - 1 + i) + 2);

            float tmp[4][4];
            for(int sz = 0; sz < 4; ++sz)
              for(int sy = 0; sy < 4; ++sy)
                {
                  float s = 0;

                  for(int sx = 0; sx < 4; ++sx)
                    {
                      const int l[3] = {sx, sy, sz};

                      int g[3];
                      for(int c = 0; c < 3; ++c)
                        g[c] = max(0, min(N[c] - 1, l[c] - 1 + anchor[c]));

                      s += w[0][sx] * data[g[0] + N[0] * (g[1] + N[1] * g[2])];
                    }

                  tmp[sz][sy] = s;
                }

            float partial[4];
            for(int sz = 0; sz < 4; ++sz)
              {
                float s = 0;

                for(int sy = 0; sy < 4; ++sy)
                  s += w[1][sy] * tmp[sz][sy];

                partial[sz] = s;
              }

            float val = 0;
            for(int sz = 0; sz < 4; ++sz)
              val += w[2][sz] * partial[sz];

            totval+=(val< 0.? 1.0:0.0);
          }
    rbuftval[r]=totval;
#if 1 /* peh */
    printf("r:%d -> totval:%f\n", r, totval);
#endif
  }
  int *neworder=new int[tranks];
  workxrank_t *wxra=new workxrank_t[tranks];
  for(int i=0; i<tranks; i++) {
      wxra[i].rank=i;
      wxra[i].work=ceil(rbuftval[i]);
  }
   qsort((void *)wxra,tranks,sizeof(workxrank_t),compwork);

  for(int i=0; i<tranks; i++) {
      printf("%d %f %d\n",i,wxra[i].work,wxra[i].rank);
  }

  delete [] rbuftval;

  return wxra;

}

double *findbalance(int nodes, int nranks, workxrank_t *allwxra, workxrank_t **p2pwxra, int maxrxn) {
  int *counter=new int[nodes];
  double *work=new double[nodes];
  double totwork=0.;
  //  int maxrxn=nranks/nodes;
  for(int i=0; i<nodes; i++) {
    counter[i]=0;
    work[i]=0.;
  }
  for(int i=0; i<nranks; i++) {
    totwork+=allwxra[i].work;
  }
  int j=0;
  for(int i=0; i<nranks; i++) {
    p2pwxra[j][counter[j]]=allwxra[i];
    work[j]+=allwxra[i].work;
    counter[j]++;
    double min=totwork;
    for(int k=0; k<nodes; k++) {
      if(work[k]<min && counter[k]<maxrxn) {
        min=work[k];
        j=k;
      }
    }
  }
  delete [] counter;
  return work;
}


#define MINARG 4
int main(int argc, char ** argv) {
  int ranks[3];
  int granks[3];
  int iranks[3];
  int nodes, branks;
  int tranks, num1, minbsize=0;
  int oversubf=1;

  if (argc < MINARG)
    {
      fprintf(stderr,
              "Usage: %s <nodes> <oversubscribefactor> <min boxsize> (0=any value) [rankX rankY rankZ]\n",argv[0]);
      exit(-1);
    }
  else {
    nodes=atoi(argv[1]);
    oversubf=atoi(argv[2]);
    minbsize=atoi(argv[3]);
    tranks=nodes*oversubf;
  }
  if(argc>MINARG) {
    int checkranks=1;
    for(int i = 0; i < 3; ++i) {
       iranks[i] = atoi(argv[4 + i]);
       checkranks*=iranks[i];
    }
    if(0 && checkranks!=tranks) {
      fprintf(stderr,"Invalid rank combination: product must be equal to %d\n",tranks);
      exit(1);
    }
    printf("Forcing ranks %d %d %d\n",iranks[0],iranks[1],iranks[2]);
  } else {
    for(int i = 0; i < 3; ++i)
       iranks[i] = -1;
  }

  int nranks, rank;
  double totwork=0.;

  int periods[] = {1, 1, 1};
  double gmax=0;
  workxrank_t **p2pwxra=new workxrank_t*[nodes];
  workxrank_t **gp2pwxra=new workxrank_t*[nodes];
  for(int i=0; i<nodes; i++) {
       p2pwxra[i]=new workxrank_t[tranks/nodes];
       gp2pwxra[i]=new workxrank_t[tranks/nodes];
  }
  branks=1;
  { /* just read data and set sysL */
    for(int i = 0; i < 3; ++i) {
      ranks[i]=0;
    }
    reorder(ranks);
  }
  for(int l = 1; l<=oversubf; l++) {
    tranks=nodes*l;
    for(int i = 1; i <= tranks; i++) {
      if((tranks % i) == 0) {
        num1=tranks/i;
        for(int j = 1; j <= num1; j++) {
          if((num1 % j) == 0) {
            ranks[0]=i;
            ranks[1]=j;
            ranks[2]=num1/j;
            if( (sysL[0]/ranks[0]<minbsize) ||
                (iranks[0]>0 && iranks[0]!=ranks[0])) continue;
            if((sysL[1]/ranks[1]<minbsize) ||
               (iranks[1]>0 && iranks[1]!=ranks[1])) continue;
            if((sysL[2]/ranks[2]<minbsize) ||
               (iranks[2]>0 && iranks[2]!=ranks[2])) continue;
            workxrank_t *allwxra=reorder(ranks);
            if(allwxra==NULL) continue;
            double *work=findbalance(nodes,tranks,allwxra,p2pwxra,oversubf);
            double max=0;
            double ltotwork=0.;
            for(int i=0; i<nodes; i++) {
              //              printf("work node %d=%f;",i,work[i]);
              // for(int j=0; j<tranks/nodes; j++) {
              //  if(p2pwxra[i][j].work>0.) {
              //    printf(" %d",p2pwxra[i][j].rank);
              //  }
              // }
              // printf("\n");
              ltotwork+=work[i];
              if(work[i]>max) {
                max=work[i];
              }
            }
            if(totwork==0.) {
              totwork=ltotwork;
            }
            if(ltotwork!=totwork) {
              fprintf(stderr,"Unexpected total work %f, expected %f\n",ltotwork,totwork);
              //exit(1);
            }
            printf("max work for config %d %d %d=%f\n",ranks[0],ranks[1],ranks[2],max);
            if(gmax==0. || max<gmax) {
              for(int l=0; l<nodes; l++) {
                for(int k=0; k<tranks/nodes; k++) {
                  gp2pwxra[l][k].work=p2pwxra[l][k].work;
                  gp2pwxra[l][k].rank=p2pwxra[l][k].rank;
                }
              }
              granks[0]=ranks[0]; granks[1]=ranks[1]; granks[2]=ranks[2];
              gmax=max;
              branks=tranks;
            }
            if(allwxra) {
              delete allwxra;
            }
          }
        }
      }
    }
  }

  double minwork=gmax;
  for(int i=0; i<nodes; i++) {
    double work=0.;
    for(int j=0; j<branks/nodes; j++) {
      work+=gp2pwxra[i][j].work;
    }
    minwork=(minwork>work)?work:minwork;
  }

  double unbalance=((gmax/(totwork/nodes))-1.0)*100.;
  printf("best config=%d %d %d, unbalance=%f%%, oversubscribe factor=%d\n",granks[0],granks[1],granks[2],
         unbalance,branks/nodes);
  printf("Lx=%f Ly=%f Lz=%f\n",roundf(sysL[0])/granks[0],roundf(sysL[1])/granks[1],roundf(sysL[2])/granks[2]);
  for(int i=0; i<nodes; i++) {
    printf("node %d ranks: ",i);
    for(int j=0; j<branks/nodes; j++) {
      if(gp2pwxra[i][j].work>0.) {
        printf(" %d",gp2pwxra[i][j].rank);
      }
    }
    printf("\n");
  }
  //     if (order)
  //        delete order;
  // MPI_CHECK(MPI_Comm_free(&AROcomm));
#if 0
  int mylorank;
  Rank2Node(&mylorank);
#define RANK2NODE "rank2node.dat"
  int *aranks=new int[nranks];
  FILE *fp=fopen(RANK2NODE,"r");
  if(fp==NULL) {
    fprintf(stderr,"Error opening %s file\n",RANK2NODE);
    MPI_Abort(MPI_COMM_WORLD,2);
    exit(1);
  }
  int countr=0;
  {
#define MAXINPUTLINE 1024
#define DELIMITER " \t"
    char line[MAXINPUTLINE];
    char *token;

    bool survive=false;
    while (fgets(line, MAXINPUTLINE, fp)) {
      token=strtok(line,DELIMITER);
      while(token!=NULL) {
        aranks[countr]=atoi(token);
        if(aranks[countr]==rank) {
          survive=true;
        }
        countr++;
        token=strtok(NULL,DELIMITER);
      }
    }
    fclose(fp);
    if(rank==0) {
      for(int i=0; i<countr; i++) {
        printf("rank %d survive!\n",aranks[i]);
      }
    }
    if(!survive) {
      printf("rank %d bye bye\n",rank);
    }
  }
  MPI_Group orig_group, new_group;
  MPI_CHECK( MPI_Comm_group(MPI_COMM_WORLD, &orig_group) );
  MPI_CHECK( MPI_Group_incl(orig_group, countr, aranks, &new_group) );
  MPI_CHECK( MPI_Comm_create(MPI_COMM_WORLD, new_group, &AROcomm) );

  MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &AROcomm) );

  MPI_CHECK(MPI_Comm_dup(AROcomm,&cartcomm));
  MPI_CHECK(MPI_Comm_free(&cartcomm));
#endif

  return 0;
}

