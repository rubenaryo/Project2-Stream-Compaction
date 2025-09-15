#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int *x, int offset)
        {
            int nextOffset = offset << 1;
            int k = ((blockIdx.x * blockDim.x) + threadIdx.x) * nextOffset;
            if (k >= (N-1))
                return;

            int leftIdx = k + offset - 1;
            int rightIdx = k + nextOffset - 1;

            x[rightIdx] = x[leftIdx] + x[rightIdx];
        }

        __global__ void kernDownSweep(int N, int* x, int offset)
        {
            int nextOffset = offset << 1;
            int k = ((blockIdx.x * blockDim.x) + threadIdx.x) * nextOffset;
            if (k >= (N - 1))
                return;

            int leftIdx = k + offset - 1;
            int rightIdx = k + nextOffset - 1;

            int t = x[leftIdx];
            x[leftIdx] = x[rightIdx];
            x[rightIdx] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int stages = ilog2ceil(n);
            int N = 1 << stages; // next available power of two for N
            int deviceMaxThreadsPerBlock = 1024;
            int deviceNumber = 0;

            int *dev_x;
            cudaMalloc((void**)&dev_x, sizeof(int) * N);
            checkCUDAError("CUDA: Fatal Error, failed to allocate dev_x");

            cudaMemset(dev_x, 0, sizeof(int) * N);
            checkCUDAError("CUDA: Error, failed to initialize dev_x");

            cudaMemcpy((void*)dev_x, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("CUDA: Fatal Error, failed to copy idata to dev_x");

            cudaGetDevice(&deviceNumber);
            checkCUDAError("CUDA: Failed to get Device Number, defaulting to 0...\n");

            cudaDeviceGetAttribute(&deviceMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceNumber);
            checkCUDAError("CUDA: Failed to get thread count per block, defaulting to 1024...\n");

            const int BLOCK_SIZE = std::min(N, deviceMaxThreadsPerBlock);
            dim3 fullBlocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();
            
            int stage = 1, offset = 1;
            for (; stage <= stages && offset < N; ++stage, offset <<= 1)
            {
                kernUpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, dev_x, offset);
            }
            
            cudaDeviceSynchronize();
            
            // Set root to 0
            cudaMemset(&dev_x[n-1], 0, sizeof(int));

            stage = ilog2(N) - 1;
            for (; stage >= 0; --stage)
            {
                offset = 1 << stage;
                kernDownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, dev_x, offset);
            }

            timer().endGpuTimer();

            cudaMemcpy((void*)odata, (const void*)dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("CUDA: Fatal Error, failed to copy dev_x to odata");

            cudaFree(dev_x);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
