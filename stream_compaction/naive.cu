#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <assert.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ inline void dev_swap(int*& a, int*& b)
        {
            int* temp = a;
            a = b;
            b = temp;
        }

        __global__ void kernNaiveScan(int *w, int *r, int N, int stages, int* odata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= N)
                return;

            int offset = 1;
            for (int stage = 1; stage <= stages && offset < N; ++stage, offset <<= 1)
            {
                if (k >= offset)
                {
                    w[k] = r[k - offset] + r[k];
                }
                else
                {
                    w[k] = r[k];
                }

                dev_swap(w, r);
            }

            // At return, r contains the final result due to the swap
            odata[k] = r[k];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const int BLOCK_SIZE = 128;
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            int stages = ilog2ceil(n);
            int N = 1 << stages; // next available power of two for N
            assert(stages == ilog2(N));

            // Alloc two device ping pong buffers, one for read and one for write
            int *dev_A, *dev_B;
            cudaMalloc((void**)&dev_A, sizeof(int) * N);
            cudaMalloc((void**)&dev_B, sizeof(int) * N);

            // A will be the read buffer for the first pass
            cudaMemcpy((void*)dev_A, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();         
            kernNaiveScan<<<fullBlocksPerGrid, BLOCK_SIZE>>>(dev_B, dev_A, N, stages, dev_B);
            timer().endGpuTimer();

            // Leave the first element empty for identity
            cudaMemcpy((void*)&odata[1], (const void*)dev_B, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);

            // Identity
            odata[0] = 0;

            cudaFree((void*)dev_A);
            cudaFree((void*)dev_B);
        }
    }
}
