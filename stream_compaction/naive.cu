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

        __global__ void kernNaiveScan(int *w, const int *r, int N, int idx_start)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= N)
                return;

            if (k >= idx_start)
            {
                w[k] = r[k - idx_start] + r[k];
                int stub = 42;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const int BLOCK_SIZE = 8;
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

            int idx_start = 1;
            timer().startGpuTimer();         
            for (int stage = 1; stage <= stages; ++stage)
            {
                kernNaiveScan<<<fullBlocksPerGrid, BLOCK_SIZE>>>(dev_B, dev_A, N, idx_start);
                idx_start <<= 1;
                std::swap(dev_A, dev_B); // bing bong
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_A, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree((void*)dev_A);
            cudaFree((void*)dev_B);
        }
    }
}
