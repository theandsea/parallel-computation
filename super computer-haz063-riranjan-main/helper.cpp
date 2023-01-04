/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#ifdef _MPI_
#include <mpi.h>
#endif
#include "cblock.h"
#include "apf.h"
using namespace std;

extern control_block cb;

void printMat(const char mesg[], double *E, int m, int n);

#ifdef _MPI_
void distribute(double **_E, double **_E_prev, double *R)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Request sendIE_prev[cb.px*cb.py];
    MPI_Request sendIR[cb.px*cb.py];
    MPI_Request recvIE_prev[cb.px*cb.py];
    MPI_Request recvIR[cb.px*cb.py];

    int ri = my_rank / (cb.px);
    int rj = my_rank % (cb.px);
    int m = (cb.m/cb.py) + ((ri - cb.m%cb.py) < 0 ? 1 : 0);
    int n = (cb.n/cb.px) + ((rj - cb.n%cb.px) < 0 ? 1 : 0);
    int innerBlockRowStartIndex = (n+2)+1;
    int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

    int idx, k, l;

    if(my_rank == 0)
    {
        double *stE_prev = *_E_prev + (cb.n+2) + 1 + m*(cb.n+2);
        double *stR = R + (cb.n+2) + 1 + m*(cb.n+2);

        if(rj < cb.px-1)
        {
            stE_prev += -(m)*(cb.n+2) + n;
            stR += -(m)*(cb.n+2) + n;
        }
        else
        {
            stE_prev += n + 2 -(cb.n+2);
            stR += n + 2 -(cb.n+2); 
        }

        for(int q=1; q<cb.px*cb.py; q++)
        {
            // x -> rows, y -> columns
            int x = q / (cb.px), y = q % (cb.px);

            int cm = (cb.m/cb.py) + ((x - cb.m%cb.py) < 0 ? 1 : 0);
            int cn = (cb.n/cb.px) + ((y - cb.n%cb.px) < 0 ? 1 : 0);

            MPI_Datatype tile;
            MPI_Type_vector(cm, cn, (cb.n+2), MPI_DOUBLE, &tile);
            MPI_Type_commit(&tile);

            MPI_Isend(stE_prev, 1, tile, q, 0, MPI_COMM_WORLD, &sendIE_prev[q]);
            MPI_Isend(stR, 1, tile, q, 0, MPI_COMM_WORLD, &sendIR[q]);
            MPI_Wait(&sendIE_prev[q], MPI_STATUS_IGNORE);
            MPI_Wait(&sendIR[q], MPI_STATUS_IGNORE);

            stE_prev += cm*(cb.n+2);
            stR += cm*(cb.n+2);

            if(y < cb.px-1)
            {
                stE_prev += -(cm)*(cb.n+2) + cn;
                stR += -(cm)*(cb.n+2) + cn;
            }
            else
            {
                stE_prev += cn + 2 -(cb.n+2);
                stR += cn + 2 -(cb.n+2); 
            }
        }

        stE_prev = *_E_prev;
        stR = R;

        for(k=0; k<m+2; k++)
        {
            stE_prev[k*(n+2)] = 0;
            stR[k*(n+2)] = 0;
            for(l=1; l<n+1; l++)
            {
                stE_prev[k*(n+2) + l] = stE_prev[k*(cb.n+2) + l];
                stR[k*(n+2) + l] = stR[k*(cb.n+2) + l];
            }
            stE_prev[k*(n+2) + n+1] = 0;
            stR[k*(n+2) + n+1] = 0;
        }
    }
    else
    {
        double *tE_prev = new double[m*n];
        double *tR = new double[m*n];

        MPI_Irecv(tE_prev, m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvIE_prev[my_rank]);
        MPI_Irecv(tR, m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvIR[my_rank]);
        MPI_Wait(&recvIE_prev[my_rank], MPI_STATUS_IGNORE);
        MPI_Wait(&recvIR[my_rank], MPI_STATUS_IGNORE);

        double *stE_prev = *_E_prev + innerBlockRowStartIndex;
        double *stR = R + innerBlockRowStartIndex;

        idx = 0;

        for(k=0; k<m; k++)
        {
            for(l=0; l<n; l++, idx++)
            {
                stE_prev[k*(n+2) + l] = tE_prev[idx];
                stR[k*(n+2) + l] = tR[idx];
            }
        }
    }
}
#endif


//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
#ifdef _MPI_
void initMPI(double *E,double *E_prev,double *R,int m,int n){
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0) {
        int i;

        for (i=0; i < (m+2)*(n+2); i++)
            E_prev[i] = R[i] = 0;

        for (i = (n+2); i < (m+1)*(n+2); i++) {
            int colIndex = i % (n+2);

            if(colIndex != 0 && colIndex != (n+1) && colIndex >= ((n+1)/2+1))
                E_prev[i] = 1.0;
        }

        for (i = 0; i < (m+1)*(n+2); i++) {
            int rowIndex = i / (n+2);
            int colIndex = i % (n+2);

            if(colIndex != 0 && colIndex != (n+1) && rowIndex >= ((m+1)/2+1))
                R[i] = 1.0;
        }
    }
    else
    {
        int i;

        int rowIndex = my_rank / (cb.px);
        int colIndex = my_rank % (cb.px);

        n = (n/cb.px) + 2 + ((colIndex - n%cb.px) < 0 ? 1 : 0);
        m = (m/cb.py) + 2 + ((rowIndex - m%cb.py) < 0 ? 1 : 0);

        for (i=0; i < (m)*(n); i++)
            E_prev[i] = R[i] = 0;
    }

    distribute(&E, &E_prev, R);

    // We only print the meshes if they are small enough
// #if 1
//     printMat("E_prev",E_prev,m,n);
//     printMat("R",R,m,n);
// #endif
}
#endif

void initNoMPI(double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 1
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}

void init(double *E,double *E_prev,double *R,int m,int n){
#ifdef _MPI_
    initMPI(E, E_prev, R, m, n);
#else
    initNoMPI(E, E_prev, R, m, n);
#endif
}


// px -> columns, py -> rows
// n -> columns, m -> rows
double *alloc1DNoMPI(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

#ifdef _MPI_
double *alloc1DMPI(int m,int n){
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int nx, ny;
    double *E;

    nx=n-2, ny=m-2;

    int rowIndex = my_rank / (cb.px);
    int colIndex = my_rank % (cb.px);

    if(my_rank != 0) {
        nx=(nx/cb.px) + 2 + ((colIndex - nx%cb.px) < 0 ? 1 : 0); 
        ny=(ny/cb.py) + 2 + ((rowIndex - ny%cb.py) < 0 ? 1 : 0);
    }
    else
    {
        nx += 2, ny += 2;
    }

    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}
#endif

double *alloc1D(int m,int n){

#ifdef _MPI_
    return alloc1DMPI(m, n);
#else
    return alloc1DNoMPI(m,n);
#endif
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
