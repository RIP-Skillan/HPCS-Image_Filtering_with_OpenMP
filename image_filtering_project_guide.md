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

## 2. Dataset Used for Testing: 
[Salt and Pepper Noise Dataset:Clean vs Noisy Image](https://www.kaggle.com/datasets/rajneesh231/salt-and-pepper-noise-images)

---

## 3. Code Files Provided

We have 4 essential code files:
- **`image_filter.cpp`**: The complete C++ implementation containing the binary PPM image reader/writer, serial filter baselines, optimized parallel OpenMP filter implementations, correctness verification, and high-precision timing harnesses.
- **`image_converter.py`**: A python script using PIL/Pillow to easily convert standard PNG/JPEG images into binary PPM (P6) format for the C++ program, and convert the C++ outputs back to standard PNG/JPEG images for visualization.
- **`Makefile`**: A clean Unix Makefile configured to build the C++ program with aggressive optimizations (`-O3`) and OpenMP compiler directives (`-fopenmp`).
- **`plot_benchmark.py`**: A python script using pyplot to visualize the comparison results between sequential and parallel time taken.

---

## 4. Step-by-Step Implementation Workflow

To execute this project, follow these steps on your multicore system.

### Step 1: Set up your environment
Ensure you have a standard C++ compiler (`g++` or `clang++`) with OpenMP support and Python 3 installed.
On Linux/macOS (Homebrew/WSL), install Pillow if you haven't already:
```bash
pip install Pillow
```

### Step 2: Convert an input image to PPM
Choose a high-resolution JPG or PNG image (e.g., a 1024x1024 or 3200x2400 image, matching the resolutions used in the paper's experiments) and convert it for best results:
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

### Step 4: Run the Filters
Execute the binary with your input PPM images, output paths, and target thread count:
```bash
./image_filter ppm_input_folder_path output_folder_path 4
```
*Note: Replace `4` with the number of physical cores/threads on your CPU.*

### Step 5: Convert output PPM files back to PNG/JPG (Optional)
```bash
python3 image_converter.py to_png input_ppm_folder output_png_folder
```

### Step 6: Run Benchmark to Visualize the Results
```bash
python3 plot_benchmark.py
```

---

## 5. Key OpenMP Optimization Techniques Used

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
