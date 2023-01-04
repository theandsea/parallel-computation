// ;-*- mode: c;-*-
// Matrix multiply device code



#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

extern __shared__ _FTYPE_ sharemem[];

#define stride_y 16 // keep strides x and y the same as bx and by in OPTIONS.TXT (block dimensions)
#define stride_x 32
#define mult_y TILEDIM_M/stride_y
#define mult_x TILEDIM_N/stride_x
#define mult_k_y TILEDIM_K/stride_y
#define mult_k_x TILEDIM_K/stride_x


#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}


#else
//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    
    //local shared storage
    // __shared__ _FTYPE_ As[TW][TW], Bs[TW][TW];

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    
    int I = by*TILEDIM_M + ty, J = bx*TILEDIM_N + tx;

    // 4 elements of Cs calculated together
    // this thread memory is also limited, but if just 0, it can ignore
    _FTYPE_ Cij[mult_y][mult_x] = {0.0f};

    _FTYPE_ * __restrict__ As = &sharemem[0];
    _FTYPE_ * __restrict__ Bs = &As[TILEDIM_M*TILEDIM_K];

    int t_K = TILEDIM_K, t_K0 = TILEDIM_K;


    // #pragma unroll // this unroll will wiredly decrease the performance
    // when used in inside loop, not difference
    for (int kk=0; kk<N; kk+=TILEDIM_K) {
        if (t_K > N-kk)  t_K0 = N-kk;

        // #pragma unroll
        for(int ii=0; ii<mult_y; ii++)
            for(int jj=0; jj<mult_k_x; jj++)
                // if is slow...do not diverge
                As[(ty+ii*stride_y)*t_K + tx + jj*stride_x] = (I+ii*stride_y<N && kk + tx + jj*stride_x<N)? A[((I+ii*stride_y)*N + kk + tx + jj*stride_x)]:0.0f ;

        //#pragma unroll
        for(int ii=0; ii<mult_k_y; ii++)
            for(int jj=0; jj<mult_x; jj++)
                Bs[(ty+ii*stride_y)*TILEDIM_N + tx + jj*stride_x] = (kk+ty+ii*stride_y <N && J+jj*stride_x <N) ? B[((kk+ty+ii*stride_y)*N + J+jj*stride_x)]:0.0f;

        __syncthreads();

        // #pragma unroll
        for (int k=0; k<t_K0; k++) 
            for(int ii=0; ii<mult_y; ii++)
                for(int jj=0; jj<mult_x; jj++)
                    Cij[ii][jj]+= As[(ty+ii*stride_y)*TILEDIM_K + k] * Bs[k*TILEDIM_N + tx + jj*stride_x];        
        __syncthreads(); // necessary otherwise As updated
    }

    // put it outside, or run out of resource, too much unrolled instructions...each thread only has 256 register as memory
    // #pragma unroll
    for(int ii=0; ii<mult_y; ii++)
        for(int jj=0; jj<mult_x; jj++)
            if (I+ii*stride_y<N && J+jj*stride_x<N) C[(I+ii*stride_y)*N + J+jj*stride_x] = Cij[ii][jj];
}

#endif
