# C++ OpenMP Image Filtering Course Project Guide

This guide provides a comprehensive implementation roadmap for your High-Performance Computing systems course project, directly mapping to the research paper: **"Acceleration of digital image filtering processes on multi-core processors using OpenMP technology"** by Turdiev Ulugbek and Javliev Shakhzod.

---

## 1. Project Overview & Architecture

The goal of this project is to implement, optimize, and benchmark 3D (RGB) digital image filtering using multicore parallel programming. We implement two types of filters as detailed in the paper:
1. **Gaussian Filter (Linear)**: Used for smoothing and random noise reduction. It performs weighted averaging over a 3x3 pixel neighborhood.
2. **Median Filter (Non-linear)**: Excellent for removing salt-and-pepper noise while preserving sharp image contours. It sorts the 9 values in a 3x3 pixel neighborhood and assigns the median value.

### Data Layout: Structure of Arrays (SoA)
Color images are represented in the RGB format, requiring independent operations across three channels. Following the methodology of Section 3 and 5 of the paper, our implementation decomposes the image into three separate color channels ($R, G, B$) and represents them as independent, contiguous vectors (Structure of Arrays format) to maximize cache locality and memory throughput.

---

## 2. Code Files Provided in Your Studio Panel

We have generated and delivered three essential files to your Studio panel:
- **`image_filter.cpp`**: The complete C++ implementation containing the binary PPM image reader/writer, serial filter baselines, optimized parallel OpenMP filter implementations, correctness verification, and high-precision timing harnesses.
- **`image_converter.py`**: A python script using PIL/Pillow to easily convert standard PNG/JPEG images into binary PPM (P6) format for the C++ program, and convert the C++ outputs back to standard PNG/JPEG images for visualization.
- **`Makefile`**: A clean Unix Makefile configured to build the C++ program with aggressive optimizations (`-O3`) and OpenMP compiler directives (`-fopenmp`).

---

## 3. Step-by-Step Implementation Workflow

To execute your project, follow these steps on your multicore laptop (e.g., RTX 4050 laptop CPU) or course cluster.

### Step 1: Set up your environment
Ensure you have a standard C++ compiler (`g++` or `clang++`) with OpenMP support and Python 3 installed.
On Linux/macOS (Homebrew/WSL), install Pillow if you haven't already:
```bash
pip install Pillow
```

### Step 2: Convert an input image to PPM
Choose a high-resolution JPG or PNG image (e.g., a 1024x1024 or 3200x2400 image, matching the resolutions used in the paper's experiments) and convert it:
```bash
python3 image_converter.py to_ppm input_image_folder output_ppm_folder.ppm
```

### Step 3: Compile the C++ program
Run `make` to compile the optimized binary:
```bash
make
```
This runs:
```bash
g++ -O3 -fopenmp -Wall -Wextra image_filter.cpp -o image_filter
```

### Step 4: Run the Benchmark and Filters
Execute the binary with your input PPM image, output paths, and target thread count:
```bash
./image_filter input_test.ppm output_gaussian.ppm output_median.ppm 4
```
*Note: Replace `4` with the number of physical cores/threads on your CPU.*

### Step 5: Convert output PPM files back to PNG/JPG
```bash
python3 image_converter.py to_png input_ppm_folder output_png_folder
```

---

## 4. Key OpenMP Optimization Techniques Used

Our C++ implementation uses specific high-performance design patterns to ensure maximum scalability and speedup on multi-core processors:

### A. Fine-Grained Loop Parallelism & Collapse Clause
Instead of parallelizing the three RGB channels using `#pragma omp sections` (which bounds parallelism to exactly 3 threads, leaving higher core counts underutilized), our code uses loop-level parallelization inside each filter with the **`collapse(2)`** clause:
```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        // Pixel calculations
    }
}
```
- **Why this works**: `collapse(2)` combines the `y` and `x` loop iterations into a single massive, flat iteration space. This allows OpenMP to load-balance the work perfectly across 8, 12, or 16 threads, bypassing the 3-thread limit of channel-level split.

### B. Eliminating Heap Allocations inside Parallel Loops
In the sequential and parallel Median Filter implementations, we must sort a 9-element array for each pixel. A naive implementation allocating a `std::vector` inside the pixel loop would result in millions of expensive heap allocations and memory contention between threads.
To solve this, our code uses a **fast stack-allocated 9-element array** and a **custom, inlined sorting function (`sort9`)**:
```cpp
unsigned char window[9];
// populate window...
sort9(window); // custom inlined insertion sort
```
This ensures zero thread contention for dynamic memory, delivering a massive boost in performance and scaling.

---

## 5. Analyzing Performance & Writing Your Report

In the paper, the authors noted up to a **5.5× speedup** for Gaussian filtering and a **4× speedup** for Median filtering on a 10-core/16-thread system when processing a high-resolution 3200x2400 image.

When writing your course report, address these three topics to demonstrate a deep, professional understanding:

1. **Why Gaussian Filter scales better than Median Filter**:
   - The Gaussian filter is a linear filter performing static weighted sums; it has predictable data access and instruction streams.
   - The Median filter involves conditional sorting branches inside the loop (`sort9`), which causes CPU branch mispredictions and is highly dependent on input pixel values.
2. **Memory Bottlenecks vs. Compute Bound**:
   - Image filtering is inherently memory-bandwidth-bound because we perform relatively few computations per byte read from DRAM.
   - Restructuring your data in a continuous memory structure ensures spatial cache locality (sequential cache-line loading), which is crucial to prevent threads from stalling on memory reads.
3. **Loop Scheduling Trade-offs**:
   - We used `schedule(static)` because the computational load per pixel is uniform. Explain in your report why `static` scheduling has less overhead than `dynamic` scheduling when work per pixel is balanced, and under what circumstances (like variable-size kernels) `dynamic` or `guided` might be superior.

---

## 6. High-Grade Extension Ideas

To elevate your project from a basic implementation to an outstanding, high-grade submission, consider implementing these extensions:

- **Experiment with OpenMP Loop Schedulers**: Benchmark the difference between `schedule(static)`, `schedule(dynamic)`, and `schedule(guided)` using different image resolutions and thread counts. Present the speedup curves in a graph.
- **Implement Thread Affinity & Pinning**: Benchmark execution times with different thread affinity settings using environment variables:
  ```bash
  export OMP_PROC_BIND=true
  export OMP_PLACES=cores
  ```
  Analyze how socket-pinning prevents threads from migrating across CPU cores, reducing L1/L2 cache misses.
- **Vectorization (SIMD) Analysis**: Use compiler vectorization flags like `-ftree-vectorize` or `-fopt-info-vec` to check if your compiler auto-vectorizes the loop. Document how the compiler handles the SIMD layout of your RGB channels.
