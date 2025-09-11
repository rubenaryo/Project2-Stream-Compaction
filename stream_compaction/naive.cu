#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScan(int N, int* odata, const int* idata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= N)
                return;

            int d = 1;
            int log2_N = (int)ceil(log2((float)N));
            int idx_start = 1;
            while (d <= log2_N)
            {
                if (k >= idx_start)
                {
                    
                      
                }

                idx_start *= 2;
            }
            int stub = 42;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const int BLOCK_SIZE = 128;
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            timer().startGpuTimer();
            
            naiveScan<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n+1, odata, idata);

            timer().endGpuTimer();
        }
    }
}
