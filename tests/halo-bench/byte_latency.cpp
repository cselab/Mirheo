/******************************************************************************
* FILE: mpi_latency.c
* DESCRIPTION:  
*   MPI Latency Timing Program - C Version
*   In this example code, a MPI communication timing test is performed.
*   MPI task 0 will send "reps" number of 1 byte messages to MPI task 1,
*   waiting for a reply between each rep. Before and after timings are made 
*   for each rep and an average calculated when completed.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#define	NUMBER_REPS	10000

int main (int argc, char *argv[])
{
int reps,                   /* number of samples per test */
    tag,                    /* MPI message tag parameter */
    numtasks,               /* number of MPI tasks */
    rank,                   /* my MPI task number */
    dest, source,           /* send/receive task designators */
    avgT,                   /* average time per rep in microseconds */
    rc,                     /* return code */
    n;
double T1, T2,              /* start/end times per rep */
    sumT,                   /* sum of all reps times */
    deltaT,
    avgT2;                 /* time for one rep */
char msg;                   /* buffer containing 1 byte message */
MPI_Status status;          /* MPI receive routine parameter */


    int rank0 = 0;
    int rank1 = 1;

    if (argc == 3)
    {
       rank0 = atoi(argv[1]);
       rank1 = atoi(argv[2]);
    }

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
if (rank == rank0 && numtasks != 2) {
   printf("Number of tasks = %d\n",numtasks);
   printf("Only need 2 tasks - extra will be ignored...\n");
   }
MPI_Barrier(MPI_COMM_WORLD);


 

sumT = 0;
msg = 'x';
tag = 1;
reps = NUMBER_REPS;

if (rank == rank0) {
   /* round-trip latency timing test */
   printf("task %d has started...\n", rank);
   printf("Beginning latency timing test. Number of reps = %d.\n", reps);
   printf("***************************************************\n");
   printf("Rep#       T1               T2            deltaT\n");
   dest = rank1;
   source = rank1;
   for (n = 1; n <= reps+100; n++) {
      T1 = MPI_Wtime();     /* start time */
      /* send message to worker - message tag set to 1.  */
      /* If return code indicates error quit */
      rc = MPI_Send(&msg, 1, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
         printf("Send error in task 0!\n");
         MPI_Abort(MPI_COMM_WORLD, rc);
         exit(1);
         }
      /* Now wait to receive the echo reply from the worker  */
      /* If return code indicates error quit */
      rc = MPI_Recv(&msg, 1, MPI_BYTE, source, tag, MPI_COMM_WORLD, 
                    &status);
      if (rc != MPI_SUCCESS) {
         printf("Receive error in task 0!\n");
         MPI_Abort(MPI_COMM_WORLD, rc);
         exit(1);
         }
      T2 = MPI_Wtime();     /* end time */

      /* calculate round trip time and print */
      deltaT = T2 - T1;
      //printf("%4d  %8.8f  %8.8f  %2.8f\n", n, T1, T2, deltaT);
      if (n > 100)   sumT += deltaT;
      }
   avgT = (sumT*1000000)/reps;
   avgT2 = (sumT*1000000)/reps;
   printf("***************************************************\n");
   printf("\n*** Avg round trip time = %.2f microseconds\n", avgT2);
   printf("*** Avg one way latency = %.2f microseconds\n", avgT2/2);
   printf("\n*** Avg round trip time = %d microseconds\n", avgT);
   printf("*** Avg one way latency = %d microseconds\n", avgT/2);
   } 

else if (rank == rank1) {
   printf("task %d has started...\n", rank);
   dest = rank0;
   source = rank0;
   for (n = 1; n <= reps+100; n++) {
      rc = MPI_Recv(&msg, 1, MPI_BYTE, source, tag, MPI_COMM_WORLD, 
                    &status);
      if (rc != MPI_SUCCESS) {
         printf("Receive error in task 1!\n");
         MPI_Abort(MPI_COMM_WORLD, rc);
         exit(1);
         }
      rc = MPI_Send(&msg, 1, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
         printf("Send error in task 1!\n");
         MPI_Abort(MPI_COMM_WORLD, rc);
         exit(1);
         }
      }
   }

MPI_Finalize();
exit(0);
}
