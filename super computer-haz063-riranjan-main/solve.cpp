/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#pragma GCC target("avx2")

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#ifdef _MPI_
#include <mpi.h>
#endif
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <fstream>
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

// px -> columns, py -> rows
// n -> columns, m -> row

void solveNoMPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

#ifdef _MPI_
void solveMPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
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

    int idx;//, k, l;

    // After this the boundaries are m+2, n+2
    // ri, rj contain the 2D co-ordinates of current processor

    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;

    MPI_Datatype column_type;
    MPI_Type_vector(m, 1, n+2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    /* [LEFT, RIGHT, BOTTOM, TOP] */
    int neighbours_ranks[4];

    neighbours_ranks[0] = rj == 0 ?  -1 : ri*cb.px + (rj-1);
    neighbours_ranks[1] = rj == cb.px-1 ? -1 : ri*cb.px + (rj+1);
    neighbours_ranks[2] = ri < cb.py-1 ? (ri+1)*cb.px + (rj) : -1;
    neighbours_ranks[3] = ri > 0 ? (ri-1)*cb.px + (rj) : -1;

    MPI_Request request_rB = MPI_REQUEST_NULL;
    MPI_Request request_rT = MPI_REQUEST_NULL;
    MPI_Request request_rL = MPI_REQUEST_NULL;
    MPI_Request request_rR = MPI_REQUEST_NULL;

    MPI_Request request_sB = MPI_REQUEST_NULL;
    MPI_Request request_sT = MPI_REQUEST_NULL;
    MPI_Request request_sL = MPI_REQUEST_NULL;
    MPI_Request request_sR = MPI_REQUEST_NULL;

    for (niter = 0; niter < cb.niters; niter++){

        double *recvBufB = new double[n];
        double *recvBufT = new double[n];
        double *recvBufL = new double[m];
        double *recvBufR = new double[m];

        if  (cb.debug && (niter==0)){
            stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
            if (cb.plot_freq)
            plotter->updatePlot(E, -1, m+1, n+1);
        }

        int i,j;

        // Fills in the TOP Ghost Cells
        if(ri != 0)
        {
            MPI_Isend(E_prev + innerBlockRowStartIndex, n, MPI_DOUBLE, neighbours_ranks[3], my_rank, MPI_COMM_WORLD, &request_sT);
            MPI_Irecv(recvBufT, n, MPI_DOUBLE, neighbours_ranks[3], neighbours_ranks[3], MPI_COMM_WORLD, &request_rT);
        }
        else
        {
            for (i = 0; i < (n+2); i++) {
                E_prev[i] = E_prev[i + (n+2)*2];
            }
        }

        // Fills in the BOTTOM Ghost Cells
        if(ri != cb.py-1)
        {
            MPI_Isend(E_prev + innerBlockRowEndIndex, n, MPI_DOUBLE, neighbours_ranks[2], my_rank, MPI_COMM_WORLD, &request_sB);
            MPI_Irecv(recvBufB, n, MPI_DOUBLE, neighbours_ranks[2], neighbours_ranks[2], MPI_COMM_WORLD, &request_rB);
        }
        else
        {
            for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
                E_prev[i] = E_prev[i - (n+2)*2];
            }
        }

        // Fills in the RIGHT Ghost Cells
        if(rj == cb.px-1)
        {
            for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
                E_prev[i] = E_prev[i-2];
            }
        }
        else
        {
            MPI_Isend(E_prev + innerBlockRowStartIndex + n-1, 1, column_type, neighbours_ranks[1], my_rank, MPI_COMM_WORLD, &request_sR);
            MPI_Irecv(recvBufR, m, MPI_DOUBLE, neighbours_ranks[1], neighbours_ranks[1], MPI_COMM_WORLD, &request_rR);
        }

        // Fills in the LEFT Ghost Cells
        if(rj == 0)
        {
            for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
                E_prev[i] = E_prev[i+2];
            }
        }
        else
        {
            MPI_Isend(E_prev + innerBlockRowStartIndex, 1, column_type, neighbours_ranks[0], my_rank, MPI_COMM_WORLD, &request_sL);
            MPI_Irecv(recvBufL, m, MPI_DOUBLE, neighbours_ranks[0], neighbours_ranks[0], MPI_COMM_WORLD, &request_rL);
        }

    //////////////////////////////////////////////////////////////////////////////

    #define FUSED 1

    #ifdef FUSED
        // Solve for the excitation, a PDE
        // only the inner part
        for(j = innerBlockRowStartIndex + (n+2) + 1; j <= innerBlockRowEndIndex-(n+2) + 1; j+=(n+2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for(i = 0; i < n-1; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
    #else
        // Solve for the excitation, a PDE
        for(j = innerBlockRowStartIndex + (n+2) + 1; j <= innerBlockRowEndIndex - (n+2) + 1; j+=(n+2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */

        for(j = innerBlockRowStartIndex + (n+2) + 1; j <= innerBlockRowEndIndex - (n+2) + 1; j+=(n+2)) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
    #endif

        idx = 0;

        if(ri != 0)
        {
            MPI_Wait(&request_sT, MPI_STATUS_IGNORE);
            MPI_Wait(&request_rT, MPI_STATUS_IGNORE);

            for (i = 1, idx = 0; i < (n+1); i++, idx++) {
                E_prev[i] = recvBufT[idx];
            }
        }

        if(ri != cb.py-1)
        {
            MPI_Wait(&request_sB, MPI_STATUS_IGNORE);
            MPI_Wait(&request_rB, MPI_STATUS_IGNORE);

            for (i = ((m+2)*(n+2)-(n+2)) + 1, idx = 0; i < (m+2)*(n+2) - 1; i++, idx++) {
                E_prev[i] = recvBufB[idx];
            }
        }

        if(rj != cb.px-1)
        {
            MPI_Wait(&request_rR, MPI_STATUS_IGNORE);
            MPI_Wait(&request_sR, MPI_STATUS_IGNORE);

            for (i = (n+2) + (n+1), idx=0; i < (m+1)*(n+2); i+=(n+2), idx++) {
                E_prev[i] = recvBufR[idx];
            }
        }

        if(rj != 0)
        {
            MPI_Wait(&request_sL, MPI_STATUS_IGNORE);
            MPI_Wait(&request_rL, MPI_STATUS_IGNORE);

            for (i = n+2, idx=0; i < (m)*(n+2)+1; i+=(n+2), idx++) {
                E_prev[i] = recvBufL[idx];
            }
        }

        // Compute the boundry

    #ifdef FUSED
        // Solve for the excitation, a PDE

        double *E_tmp_T = E + innerBlockRowStartIndex;
        double *E_prev_tmp_T = E_prev + innerBlockRowStartIndex;
        double *R_tmp_T = R + innerBlockRowStartIndex;

        double *E_tmp__B = E + innerBlockRowEndIndex;
        double *E_prev_tmp_B = E_prev + innerBlockRowEndIndex;
        double *R_tmp_B = R + innerBlockRowEndIndex;

        for(i = 0; i < n; i++) {
            E_tmp_T[i] = E_prev_tmp_T[i]+alpha*(E_prev_tmp_T[i+1]+E_prev_tmp_T[i-1]-4*E_prev_tmp_T[i]+E_prev_tmp_T[i+(n+2)]+E_prev_tmp_T[i-(n+2)]);
            E_tmp_T[i] += -dt*(kk*E_prev_tmp_T[i]*(E_prev_tmp_T[i]-a)*(E_prev_tmp_T[i]-1)+E_prev_tmp_T[i]*R_tmp_T[i]);
            R_tmp_T[i] += dt*(epsilon+M1* R_tmp_T[i]/(E_prev_tmp_T[i]+M2))*(-R_tmp_T[i]-kk*E_prev_tmp_T[i]*(E_prev_tmp_T[i]-b-1));

            E_tmp__B[i] = E_prev_tmp_B[i]+alpha*(E_prev_tmp_B[i+1]+E_prev_tmp_B[i-1]-4*E_prev_tmp_B[i]+E_prev_tmp_B[i+(n+2)]+E_prev_tmp_B[i-(n+2)]);
            E_tmp__B[i] += -dt*(kk*E_prev_tmp_B[i]*(E_prev_tmp_B[i]-a)*(E_prev_tmp_B[i]-1)+E_prev_tmp_B[i]*R_tmp_B[i]);
            R_tmp_B[i] += dt*(epsilon+M1* R_tmp_B[i]/(E_prev_tmp_B[i]+M2))*(-R_tmp_B[i]-kk*E_prev_tmp_B[i]*(E_prev_tmp_B[i]-b-1));
        }

        double *E_tmp_L = E + innerBlockRowStartIndex + (n+2);
        double *E_prev_tmp_L = E_prev + innerBlockRowStartIndex + (n+2);
        double *R_tmp_L = R + innerBlockRowStartIndex + (n+2);

        double *E_tmp_R = E + innerBlockRowStartIndex + n - 1 + (n+2);
        double *E_prev_tmp_R = E_prev + innerBlockRowStartIndex + n - 1 + (n+2);
        double *R_tmp_R = R + innerBlockRowStartIndex + n - 1 + (n+2);

        for(i = 0; i < ((m-3)*(n+2))+1; i+=(n+2)) {
            E_tmp_L[i] = E_prev_tmp_L[i]+alpha*(E_prev_tmp_L[i+1]+E_prev_tmp_L[i-1]-4*E_prev_tmp_L[i]+E_prev_tmp_L[i+(n+2)]+E_prev_tmp_L[i-(n+2)]);
            E_tmp_L[i] += -dt*(kk*E_prev_tmp_L[i]*(E_prev_tmp_L[i]-a)*(E_prev_tmp_L[i]-1)+E_prev_tmp_L[i]*R_tmp_L[i]);
            R_tmp_L[i] += dt*(epsilon+M1* R_tmp_L[i]/( E_prev_tmp_L[i]+M2))*(-R_tmp_L[i]-kk*E_prev_tmp_L[i]*(E_prev_tmp_L[i]-b-1));

            E_tmp_R[i] = E_prev_tmp_R[i]+alpha*(E_prev_tmp_R[i+1]+E_prev_tmp_R[i-1]-4*E_prev_tmp_R[i]+E_prev_tmp_R[i+(n+2)]+E_prev_tmp_R[i-(n+2)]);
            E_tmp_R[i] += -dt*(kk*E_prev_tmp_R[i]*(E_prev_tmp_R[i]-a)*(E_prev_tmp_R[i]-1)+E_prev_tmp_R[i]*R_tmp_R[i]);
            R_tmp_R[i] += dt*(epsilon+M1* R_tmp_R[i]/(E_prev_tmp_R[i]+M2))*(-R_tmp_R[i]-kk*E_prev_tmp_R[i]*(E_prev_tmp_R[i]-b-1));
        }

    #else
        // Solve for the excitation, a PDE

        // Top

        E_tmp = E + innerBlockRowStartIndex;
        E_prev_tmp = E_prev + innerBlockRowStartIndex;
        for(i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */
        E_tmp = E + innerBlockRowStartIndex;
        E_prev_tmp = E_prev + innerBlockRowStartIndex;
        R_tmp = R + innerBlockRowStartIndex;
        for(i = 0; i < n; i++) {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }


        // Bottom

        E_tmp = E + innerBlockRowEndIndex;
        E_prev_tmp = E_prev + innerBlockRowEndIndex;
        for(i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */
        E_tmp = E + innerBlockRowEndIndex;
        E_prev_tmp = E_prev + innerBlockRowEndIndex;
        R_tmp = R + innerBlockRowEndIndex;
        for(i = 0; i < n; i++) {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }

        // Left

        E_tmp = E + innerBlockRowStartIndex + (n+2);
        E_prev_tmp = E_prev + innerBlockRowStartIndex + (n+2);
        for(i = 0; i < ((m-3)*(n+2))+1; i+=(n+2)) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */
        E_tmp = E + innerBlockRowStartIndex + (n+2);
        E_prev_tmp = E_prev + innerBlockRowStartIndex + (n+2);
        R_tmp = R + innerBlockRowStartIndex + (n+2);
        for(i = 0; i < ((m-3)*(n+2))+1; i+=(n+2)) {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }

        // Right

        E_tmp = E + innerBlockRowStartIndex + n - 1 + (n+2);
        E_prev_tmp = E_prev + innerBlockRowStartIndex + n - 1 + (n+2);
        for(i = 0; i < ((m-3)*(n+2))+1; i+=(n+2)) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */
        E_tmp = E + innerBlockRowStartIndex + n - 1 + (n+2);
        E_prev_tmp = E_prev + innerBlockRowStartIndex + n - 1 + (n+2);
        R_tmp = R + innerBlockRowStartIndex + n - 1 + (n+2);
        for(i = 0; i < ((m-3)*(n+2))+1; i+=(n+2)) {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    #endif

        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq){
            if ( !(niter % cb.stats_freq)){
                stats(E,m,n,&mx,&sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq){
            if (!(niter % cb.plot_freq)){
                plotter->updatePlot(E,  niter, m, n);
            }
        }

        double *tmp = E; E = E_prev; E_prev = tmp;
    }

    stats(E_prev,m,n,&Linf,&sumSq);

    double fSumSq=0;
    double fLinf=0;

    MPI_Reduce(&sumSq, &fSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Linf, &fLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
    {
        Linf = fLinf;
        L2 = L2Norm(fSumSq);
    }

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}
#endif

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
#ifdef _MPI_
    solveMPI(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#else
    solveNoMPI(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#endif
}

void printMat2(const char mesg[], double *E, int m, int n){
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
