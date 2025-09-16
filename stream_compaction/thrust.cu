#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // Create host vector from idata
            thrust::host_vector<int> hs_in(idata, idata + n);

            // Allocate on device from host vector
            thrust::device_vector<int> dv_in = hs_in;

            // Allocate blank result vector on device
            thrust::device_vector<int> dv_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

            thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }
    }
}
