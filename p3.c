#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <mpi.h>

#define DEBUG 1

#define N 5

//AUXILIARY FUNCTIONS
//--------------------------------------------------------------------------------

void printVector(float vector[N]) {
  int i;
  for (i = 0;i < N;i++) {
    printf("%.0f ", vector[i]);
  }
  printf("\n");
}

void printMatrix(float *matrix, int size) {
  int i;
  for (i = 0;i < size;i++) {
    if (i % N == 0) {
      printf("\n");
    }
      printf("%.0f ", *(matrix + i));
  }
  printf("\n");
}

void printDebugInfo(float *matrix, int size, float vector[N], int rank) {
  printf("Process %d received matrix:", rank);
  printMatrix(matrix, size);
  printf("\n");

  printf("Process %d received vector:\n", rank);
  printVector(vector); 
  printf("\n-----------------------\n");
}


int main(int argc, char *argv[]) {

  int numprocs, rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int i, j;
  float *flatmatrix = NULL;
  float vector[N];
  int *sendcounts = NULL, *displs = NULL;
  float *finalResult = NULL;

  struct timeval  worktime1, worktime2, commtime1, commtime2;
  int worktime, commtime;

  //INITIALIZE DATA
  //--------------------------------------------------------------------------------

  if (rank == 0) { //only the master process initializes the things

    /* Initialize Matrix and Vector */
    flatmatrix = (float *)malloc(N * N * sizeof(float));

    for (i = 0;i < N;i++) {
      vector[i] = i;
      for (j = 0;j < N;j++) {
        flatmatrix[i * N + j] = i+j;
      }
    }

    //initialize the sendcounts array, that sets the number of elements to be sent to each process
    int quotient = N / numprocs;
    int remainder = N % numprocs;
    sendcounts = (int *)malloc(numprocs * sizeof(int));
    for (i = 0; i < numprocs; i++) {
      sendcounts[i] = N * (quotient + (i < remainder ? 1 : 0)); //if i is less than the remainder, add 1 to the quotient (we split evenly the remainder among the first processes)
    }

    //initialize the displs array, that sets the position in the matrix to start sending each one
    displs = (int *)malloc(numprocs * sizeof(int));
    displs[0] = 0;
    for (i = 1; i < numprocs; i++) {
      displs[i] = displs[i-1] + sendcounts[i-1];
    }

    //create array for final result
    finalResult = (float *)malloc(N * sizeof(float));

    if (DEBUG) {
      printf("Sendcounts: ");
      for (i = 0;i < numprocs;i++) {
        printf("%d ", sendcounts[i]);
      }
      printf("\n");

      printf("Initial matrix:");
      printMatrix(flatmatrix, N * N);
      printf("Initial vector:\n");
      printVector(vector);
      printf("\n\n\n");
    }
  }

  //SEND DATA TO ALL PROCESSES
  //--------------------------------------------------------------------------------
  gettimeofday(&commtime1, NULL); //start time measure
  
  MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD); //send the vector to all processes

  int elements; //number of elements that this process will receive
  MPI_Scatter(sendcounts, 1, MPI_INT, &elements, 1, MPI_INT, 0, MPI_COMM_WORLD); //send the count of elements to each process

  float *recvbuf = (float *)malloc(elements * sizeof(float));
  MPI_Scatterv(flatmatrix, sendcounts, displs, MPI_FLOAT, recvbuf, elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

  gettimeofday(&commtime2, NULL); //stop time measure
  commtime = (commtime2.tv_usec - commtime1.tv_usec) + 1000000 * (commtime2.tv_sec - commtime1.tv_sec);

  if (DEBUG) {
    for (int i = 0; i < numprocs; i++) {
      MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to reach this point
      if (rank == i) {
        printDebugInfo(recvbuf, elements, vector, rank);
        sleep(1);
      }
    }
  }


  //PERFORM CALCULATIONS
  //--------------------------------------------------------------------------------
  gettimeofday(&worktime1, NULL); //start time measure

  int rows = elements / N;
  float result[rows];
  for (i = 0;i < rows;i++) {
    result[i]=0; //set result at that position to 0
    for(j=0;j<N;j++) {
      result[i] += recvbuf[i * N + j]*vector[j]; //set the result
    }
  }

  gettimeofday(&worktime2, NULL); //stop time measure
  worktime = (worktime2.tv_usec - worktime1.tv_usec) + 1000000 * (worktime2.tv_sec - worktime1.tv_sec);

  if (DEBUG) {
    for (i = 0; i < numprocs; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == i) {
        printf("Process %d calculated result:\n", rank);
        for (j = 0;j < rows;j++) {
          printf("%.0f ", result[j]);
        }
        printf("\n");
        sleep(1);
      }
    }
  }

  //GATHER RESULTS
  //--------------------------------------------------------------------------------

  if (rank == 0) {
    gettimeofday(&worktime1, NULL); //start time measure
    
    for (i = 0;i < numprocs;i++) { //we reuse the sendcounts array for the recvcounts
      sendcounts[i] /= N; //set the number of elements that will be received from each process
      displs[i] /= N; //we can also reuse the displs
    }

    gettimeofday(&worktime2, NULL); //stop time measure
    worktime += ((worktime2.tv_usec - worktime1.tv_usec) + 1000000 * (worktime2.tv_sec - worktime1.tv_sec));
  }

  gettimeofday(&commtime1, NULL); //start time measure

  MPI_Barrier(MPI_COMM_WORLD); //wait for all processes to reach this point, since they all need to have finished calculating the result
  MPI_Gatherv(result, rows, MPI_FLOAT, finalResult, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

  gettimeofday(&commtime2, NULL); //stop time measure
  commtime += ((commtime2.tv_usec - commtime1.tv_usec) + 1000000 * (commtime2.tv_sec - commtime1.tv_sec));

  if (DEBUG) {
    if (rank == 0) {
      printf("FINAL RESULT:\n");
      printVector(finalResult);
    }
  }
  else {
    for (i = 0; i < numprocs; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == i) {
        printf("Process %d:\n", rank);
        printf("Communication time (seconds) = %lf\n", (double)commtime / 1E6); //communication time
        printf("Work time (seconds) = %lf\n\n", (double)worktime / 1E6); //working time
        sleep(1);
      }
    }

    //Total work and communication times (slowest process):
    int maxWorkTime, maxCommTime;
    MPI_Reduce(&worktime, &maxWorkTime, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&commtime, &maxCommTime, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("Total communication time (seconds) = %lf\n", (double)maxCommTime / 1E6);
      printf("Total work time (seconds) = %lf\n", (double)maxWorkTime / 1E6);
    }
  }

  //FREE MEMORY
  //--------------------------------------------------------------------------------
  if(rank == 0) {
    free(flatmatrix);
    free(sendcounts);
    free(displs);
    free(finalResult);
  }
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
