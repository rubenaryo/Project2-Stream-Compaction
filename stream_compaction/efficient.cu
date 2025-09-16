#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <assert.h>

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

        __host__ void runScan(int N, int *dev_x)
        {
            assert(N % 2 == 0); // At this point we assume we are working with powers of two

            int stages = ilog2(N);

            int stage = 1, offset = 1;
            for (; stage <= stages && offset < N; ++stage, offset <<= 1)
            {
                int nextOffset = offset << 1;
                int numThreads = N / nextOffset; // N is guaranteed to be a power of two, and offset is a multiple of two.
                int blockSize = std::min(numThreads, CUDA_MAX_THREADS_PER_BLOCK);
                dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

                kernUpSweep<<<blocksPerGrid, blockSize>>>(N, dev_x, offset);
            }

            cudaDeviceSynchronize();

            // Set root to 0
            cudaMemset(&dev_x[N - 1], 0, sizeof(int));

            stage = ilog2(N) - 1;
            for (; stage >= 0; --stage)
            {
                offset = 1 << stage;
                int nextOffset = offset << 1;
                int numThreads = N / nextOffset; // N is guaranteed to be a power of two, and offset is a multiple of two.
                int blockSize = std::min(numThreads, CUDA_MAX_THREADS_PER_BLOCK);
                dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernDownSweep<<<blocksPerGrid, blockSize>>>(N, dev_x, offset);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int stages = ilog2ceil(n);
            int N = 1 << stages; // next available power of two for N
            
            int *dev_x;
            cudaMalloc((void**)&dev_x, sizeof(int) * N);
            checkCUDAError("CUDA: Fatal Error, failed to allocate dev_x");

            cudaMemset(dev_x, 0, sizeof(int) * N);
            checkCUDAError("CUDA: Error, failed to initialize dev_x");

            cudaMemcpy((void*)dev_x, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("CUDA: Fatal Error, failed to copy idata to dev_x");

            timer().startGpuTimer();
            
            runScan(N, dev_x);

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
            
            int deviceMaxThreadsPerBlock = 1024;
            int deviceNumber = 0;

            int* dev_bools;
            int* dev_idata;

            int N = 1 << ilog2ceil(n); // next available power of two for n
            cudaMalloc(&dev_bools, sizeof(int) * N);
            cudaMalloc(&dev_idata, sizeof(int) * N);

            cudaMemset(dev_idata, 0, sizeof(int) * N);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();

            int numThreads = N;
            int blockSize = std::min(numThreads, deviceMaxThreadsPerBlock);
            dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
            Common::kernMapToBoolean<<<blocksPerGrid, blockSize>>>(N, dev_bools, dev_idata);


            timer().endGpuTimer();

            cudaFree(dev_bools);
            cudaFree(dev_idata);
            return -1;
        }
    }
}
