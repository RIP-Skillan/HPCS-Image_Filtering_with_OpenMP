#include<bits/stdc++.h>
#include<cuda_runtime.h>

// Coordinate clamping to handle image boundaries robustly
__device__ inline int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Optimized insertion sort for a static 9-element array (avoids dynamic memory allocations)
__device__ inline void sort9(unsigned char arr[9]) {
    for (int i = 1; i < 9; ++i) {
        unsigned char key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// Simple structure to hold RGB image channels independently (Structure of Arrays)
struct Image {
    int width;
    int height;
    std::vector<unsigned char> r;
    std::vector<unsigned char> g;
    std::vector<unsigned char> b;
};

// PPM P6 (Binary RGB) Image Loader
bool read_ppm(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string format;
    file >> format;
    if (format != "P6") {
        std::cerr << "Error: Only P6 (binary RGB PPM) format is supported" << std::endl;
        return false;
    }

    // Skip whitespace and comments
    char c = file.get();
    while (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        c = file.get();
    }
    if (c == '#') {
        std::string comment;
        std::getline(file, comment);
        file >> c;
    }
    file.unget();

    file >> img.width >> img.height;
    int max_val;
    file >> max_val;
    if (max_val != 255) {
        std::cerr << "Error: Only 8-bit PPM files (max value 255) are supported" << std::endl;
        return false;
    }

    // Skip single byte of whitespace after max_val
    file.get();

    int num_pixels = img.width * img.height;
    img.r.resize(num_pixels);
    img.g.resize(num_pixels);
    img.b.resize(num_pixels);

    std::vector<unsigned char> buffer(num_pixels * 3);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (int i = 0; i < num_pixels; ++i) {
        img.r[i] = buffer[i * 3 + 0];
        img.g[i] = buffer[i * 3 + 1];
        img.b[i] = buffer[i * 3 + 2];
    }

    file.close();
    return true;
}

// PPM P6 Image Writer
bool write_ppm(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    file << "P6\n" << img.width << " " << img.height << "\n255\n";

    int num_pixels = img.width * img.height;
    std::vector<unsigned char> buffer(num_pixels * 3);

    for (int i = 0; i < num_pixels; ++i) {
        buffer[i * 3 + 0] = img.r[i];
        buffer[i * 3 + 1] = img.g[i];
        buffer[i * 3 + 2] = img.b[i];
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    file.close();
    return true;
}

// --- FILTER IMPLEMENTATIONS ---

// Gaussian Filter for Cuda
__global__ void gaussian_filter_parallel(int width, int height, unsigned char* input, unsigned char* output) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                int weight = (ky == 0 && kx == 0) ? 4 : ((ky == 0 || kx == 0) ? 2 : 1);
                sum += input[py * width + px] * weight;
            }
        }
        output[y * width + x] = static_cast<unsigned char>(sum / 16);
    }
}

// Median Filter for Cuda
__global__ void median_filter_parallel(int width, int height, unsigned char* input, unsigned char* output) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        unsigned char window[9];
        int count = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                window[count++] = input[py * width + px];
            }
        }
        sort9(window);
        output[y * width + x] = window[4];
    }
}

bool exportToCSV(const std::string& filename,
                const std::vector<double>& method_a,
                const std::vector<double>& method_b)
{
    // 1. Open the output file stream
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }

    // 2. Write the CSV header line
    file << "Number of Images,Cuda Guass Filter,Cuda Median Filter\n";

    // 3. Find the minimum size among all vectors to avoid out-of-bounds access
    size_t num_rows = std::min({method_a.size(), method_b.size()});

    // 4. Write data row by row
    for (size_t i = 0; i < num_rows; ++i) {
        file << i << ","
             << method_a[i] << ","
             << method_b[i] << "\n";
    }

    file.close();
    std::cout << "Successfully saved data to " << filename << " (" << num_rows << " rows)\n";
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <out_gaussian.ppm> <out_median.ppm> [num_threads]" << std::endl;
        return 1;
    }

    std::string input_folder = argv[1];
	std::string output_folder = argv[2];
    std::filesystem::path file_path;
    std::string input_file;
    std::string out_gauss_file;
    std::string out_median_file;
    
    //std::vector<double> gauss_seq = {0};
    std::vector<double> gauss_pl = {0};
    //std::vector<double> med_seq = {0};
    std::vector<double> med_pl = {0};

    std::cout << "===== CUDA Image Filtering Benchmark =====" << std::endl;
    std::cout << "Gauss Filter\t\t\t\t|\tMedian Filter" << std::endl;
    try {
        if (std::filesystem::exists(input_folder) && std::filesystem::is_directory(input_folder)) {

            cudaEvent_t start,stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float gauss_time, med_time, temp;

            for (const auto& entry: std::filesystem::directory_iterator(input_folder)) {
                if (std::filesystem::is_regular_file(entry.status()) && entry.path().extension() == ".ppm") {
                    file_path = entry.path();
                    input_file = file_path.string();
                    out_gauss_file = output_folder + (file_path.stem().string() + "_gauss" + file_path.extension().string());
                    out_median_file = (output_folder) + (file_path.stem().string() + "_median" + file_path.extension().string());

                    Image input_img;
                    if (!read_ppm(input_file, input_img)) {
                        return 1;
                    }

                    int w = input_img.width;
                    int h = input_img.height;

                    // Allocate output structures
                    Image out_gauss_parallel = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};

                    Image out_median_parallel = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};

                    // Allocate Device Memory
                    size_t size = w * h * sizeof(unsigned char);
                    unsigned char* d_input = nullptr;
                    unsigned char* d_output = nullptr;
                    cudaMalloc(&d_input, size);
                    cudaMalloc(&d_output, size);
                    dim3 dimBlock(16, 16);
                    dim3 dimGrid((w + 15) / 16, (h + 15) / 16);

                    //time_setup
                    gauss_time = 0;
                    med_time = 0;

                    //Parallel

                    cudaEventRecord(start);
                    cudaMemcpy(d_input, input_img.r.data(), size, cudaMemcpyHostToDevice);
                    gaussian_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    gauss_time += temp;
                    cudaMemcpy(out_gauss_parallel.r.data(), d_output, size, cudaMemcpyDeviceToHost);

                    cudaEventRecord(start);
                    median_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    med_time += temp;
                    cudaMemcpy(out_median_parallel.r.data(), d_output, size, cudaMemcpyDeviceToHost);

                    cudaMemcpy(d_input, input_img.g.data(), size, cudaMemcpyHostToDevice);
                    cudaEventRecord(start);
                    gaussian_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    gauss_time += temp;
                    cudaMemcpy(out_gauss_parallel.g.data(), d_output, size, cudaMemcpyDeviceToHost);

                    median_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    med_time += temp;
                    cudaEventRecord(start);
                    cudaMemcpy(out_median_parallel.g.data(), d_output, size, cudaMemcpyDeviceToHost);

                    cudaMemcpy(d_input, input_img.b.data(), size, cudaMemcpyHostToDevice);
                    cudaEventRecord(start);
                    gaussian_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    gauss_time += temp;
                    cudaMemcpy(out_gauss_parallel.b.data(), d_output, size, cudaMemcpyDeviceToHost);

                    cudaEventRecord(start);
                    median_filter_parallel<<<dimGrid, dimBlock>>>(w, h, d_input, d_output);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&temp, start, stop);
                    med_time += temp;
                    cudaMemcpy(out_median_parallel.b.data(), d_output, size, cudaMemcpyDeviceToHost);
                    
                    std::cout << "Gauss TIME: " << gauss_time << " ms\t\tMedian TIME: " << med_time << std::endl;
                    gauss_pl.push_back(gauss_pl.back() + gauss_time);
                    med_pl.push_back(med_pl.back() + med_time);

                    cudaFree(d_input);
                    cudaFree(d_output);
                    
                    write_ppm(out_gauss_file, out_gauss_parallel);
                    write_ppm(out_median_file, out_median_parallel);
                }
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    std::cout << "============================================" << std::endl;
    exportToCSV("cuda_benchmark_results.csv", gauss_pl, med_pl);

	return 0;
}







