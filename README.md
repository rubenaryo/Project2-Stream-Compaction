CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Ruben Young
  * [LinkedIn](https://www.linkedin.com/in/rubenaryo/), [Personal Site](https://rubenaryo.com)
* Tested on: Windows 11, AMD Ryzen 7 7800X3D, RTX 4080 SUPER (Compute Capability 8.9)

## Stream Compaction

### Overview
Stream Compaction is an algorithm with the simple goal of copying elements from one array to another if they fulfill a certain condition. For our purposes we will be filtering out all zero elements.

This serves as a convenient algorithm not just for demonstrating the capabilities of parallel processing, but also tinkering with and understanding optimization for highly parallel algorithms. 

### Naive Implementation

The naive approach is to simply add elements in stages, offsetting the read index by increasing powers of two. This simple approach is fast to implement but has many problems, such as:

* Many more addition operations than a sequential CPU approach. O(n) -> O(n * log2(n))
* Cannot be done in-place, as multiple threads are reading and writing to the same indices

<div style="margin-left: auto;
            margin-right: auto;
            width: 100%">
            
| ![](img/figure-39-2.jpg) | 
|:--:| 
| *From GPU Gems 3* |

</div>

### Work-Efficient Implementation

A much more work-efficient approach is to treat the input array instead as a balanced tree, performing an "up-sweep" stage that performs a parallel reduction that creates staggered partial sums, followed by a "down-sweep" which stitches them together to arrive at the final scan. 

This version is vastly superior to the naive approach, as it avoids the increased addition operations and can be performed in-place.

| ![](img/upsweep.png) | ![](img/downsweep.png) |
|:--:                  |:--:                    |
| Up-Sweep             |  Down-Sweep            
<p style="text-align: center;"><i>From CIS5650 at the University of Pennsylvania</i></p>


## Performance Analysis

### Block Size

Testing different block sizes up to CUDA's 1024 maximum shows no significant difference or trend in measurements.

| Block Size | Naive (POW2) | Naive (Non-POW2) | Work-Efficient (POW2) | Work-Efficient (Non-POW2) |
| ---------- | ------------ | ---------------- | --------------------- | ------------------------- |
| 128        | 27.71        | 27.19            | 10.79                 | 10.92                     |
| 256        | 27.19        | 27.71            | 10.51                 | 10.48                     |
| 512        | 27.90        | 27.43            | 10.63                 | 9.71                      |
| 1024       | 27.47        | 26.83            | 9.84                  | 10.28                     |

<p style="text-align: center;font-size:80%">
Results are in milliseconds, lower is better. <br>
Tested with N = 2<sup>26</sup> (~67M) 4-byte integers. <br>
Non-POW2 with N - 3.
</p>
