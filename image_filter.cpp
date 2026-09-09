#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <filesystem>

// Coordinate clamping to handle image boundaries robustly
inline int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Optimized insertion sort for a static 9-element array (avoids dynamic memory allocations)
inline void sort9(unsigned char arr[9]) {
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

// 1. Gaussian Filter (3x3 Kernel: weights [1 2 1; 2 4 2; 1 2 1] / 16)
void gaussian_filter_serial(int width, int height, const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = clamp(x + kx, 0, width - 1);
                    int py = clamp(y + ky, 0, height - 1);
                    int weight = (ky == 0 && kx == 0) ? 4 : 
                                 ((ky == 0 || kx == 0) ? 2 : 1);
                    sum += input[py * width + px] * weight;
                }
            }
            output[y * width + x] = static_cast<unsigned char>(sum / 16);
        }
    }
}

void gaussian_filter_parallel(int width, int height, const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = clamp(x + kx, 0, width - 1);
                    int py = clamp(y + ky, 0, height - 1);
                    int weight = (ky == 0 && kx == 0) ? 4 : 
                                 ((ky == 0 || kx == 0) ? 2 : 1);
                    sum += input[py * width + px] * weight;
                }
            }
            output[y * width + x] = static_cast<unsigned char>(sum / 16);
        }
    }
}

// 2. Median Filter (3x3 Window)
void median_filter_serial(int width, int height, const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
}

void median_filter_parallel(int width, int height, const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
}

// Helper to verify that two image channels are identical
bool verify_channels(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const std::string& name) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "Verification failed for " << name << " at index " << i 
                      << " (Serial: " << (int)a[i] << ", Parallel: " << (int)b[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

bool exportToCSV(const std::string& filename,
                const std::vector<double>& method_a,
                const std::vector<double>& method_b,
                const std::vector<double>& method_c,
                const std::vector<double>& method_d)
{
    // 1. Open the output file stream
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }

    // 2. Write the CSV header line
    file << "Number of Images,Sequential Gauss Filter,Parallel Guass Filter,Sequential Median Filter,Parallel Median Filter\n";

    // 3. Find the minimum size among all vectors to avoid out-of-bounds access
    size_t num_rows = std::min({method_a.size(), method_b.size(), method_c.size(), method_d.size()});

    // 4. Write data row by row
    for (size_t i = 0; i < num_rows; ++i) {
        file << i << ","
             << method_a[i] << ","
             << method_b[i] << ","
             << method_c[i] << ","
             << method_d[i] << "\n";
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
    
    std::vector<double> gauss_seq = {0};
    std::vector<double> gauss_pl = {0};
    std::vector<double> med_seq = {0};
    std::vector<double> med_pl = {0};
    int num_threads = omp_get_max_threads();
    if (argc >= 4) {
        num_threads = std::stoi(argv[4]);
    }
    omp_set_num_threads(num_threads);
    
    std::cout << "===== OpenMP Image Filtering Benchmark =====" << std::endl;
    //std::cout << "Threads configured: " << num_threads << " / " << omp_get_max_threads() << " max" << std::endl << std::endl;
    std::cout << "Gauss Filter\t\t\t\t|\tMedian Filter" << std::endl;
    std::cout << "Sequential Time\tParallel Time\t| Sequential Time\tParallel Time" << std::endl;
    try {
        if (std::filesystem::exists(input_folder) && std::filesystem::is_directory(input_folder)) {
            for (const auto& entry: std::filesystem::directory_iterator(input_folder)) {
                if (std::filesystem::is_regular_file(entry.status()) && entry.path().extension() == ".ppm") {
                    file_path = entry.path(); 
                    input_file = file_path.string();
                    out_gauss_file = output_folder + (file_path.stem().string() + "_gauss" + file_path.extension().string());
                    out_median_file = (output_folder) + (file_path.stem().string() + "_median" + file_path.extension().string());
                
                    
                    //std::cout << "===== OpenMP Image Filtering Benchmark =====" << std::endl;
                    // std::cout << "Loading image: " << input_file << "..." << std::endl;
                    
                    Image input_img;
                    if (!read_ppm(input_file, input_img)) {
                        return 1;
                    }

                    int w = input_img.width;
                    int h = input_img.height;
                    //std::cout << "Resolution: " << w << "x" << h << " (" << (w * h / 1000000.0) << " Megapixels)" << std::endl;
                    //std::cout << "Threads configured: " << num_threads << " / " << omp_get_max_threads() << " max" << std::endl << std::endl;

                    // Allocate output structures
                    Image out_gauss_serial = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};
                    Image out_gauss_parallel = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};
                    
                    Image out_median_serial = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};
                    Image out_median_parallel = {w, h, std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h), std::vector<unsigned char>(w * h)};

                    // ==========================================
                    // 1. BENCHMARK GAUSSIAN FILTER
                    // ==========================================
                    //std::cout << "--> Running 3x3 Gaussian Filter..." << std::endl;

                    // Serial
                    double start_time = omp_get_wtime();
                    gaussian_filter_serial(w, h, input_img.r, out_gauss_serial.r);
                    gaussian_filter_serial(w, h, input_img.g, out_gauss_serial.g);
                    gaussian_filter_serial(w, h, input_img.b, out_gauss_serial.b);
                    double gauss_serial_time = (omp_get_wtime() - start_time) * 1000.0; // ms
                    std::cout << gauss_serial_time << " ms\t";
                    gauss_seq.push_back(gauss_seq.back() + gauss_serial_time);

                    // Parallel
                    start_time = omp_get_wtime();
                    gaussian_filter_parallel(w, h, input_img.r, out_gauss_parallel.r);
                    gaussian_filter_parallel(w, h, input_img.g, out_gauss_parallel.g);
                    gaussian_filter_parallel(w, h, input_img.b, out_gauss_parallel.b);
                    double gauss_parallel_time = (omp_get_wtime() - start_time) * 1000.0; // ms
                    std::cout << gauss_parallel_time << " ms \t";
                    gauss_pl.push_back(gauss_pl.back() + gauss_parallel_time);

                    // Verify
                    bool gauss_ok = verify_channels(out_gauss_serial.r, out_gauss_parallel.r, "Gaussian Red") &&
                                    verify_channels(out_gauss_serial.g, out_gauss_parallel.g, "Gaussian Green") &&
                                    verify_channels(out_gauss_serial.b, out_gauss_parallel.b, "Gaussian Blue");
                    if (gauss_ok) {
                        //std::cout << "    Verification    : PASSED" << std::endl;
                        //std::cout << "    Speedup         : " << (gauss_serial_time / gauss_parallel_time) << "x" << std::endl;
                        write_ppm(out_gauss_file, out_gauss_parallel);
                    } else {
                        //std::cout << "    Verification    : FAILED!" << std::endl;
                    }
                    //std::cout << std::endl;

                    // ==========================================
                    // 2. BENCHMARK MEDIAN FILTER
                    // ==========================================
                    //std::cout << "--> Running 3x3 Median Filter..." << std::endl;

                    // Serial
                    start_time = omp_get_wtime();
                    median_filter_serial(w, h, input_img.r, out_median_serial.r);
                    median_filter_serial(w, h, input_img.g, out_median_serial.g);
                    median_filter_serial(w, h, input_img.b, out_median_serial.b);
                    double median_serial_time = (omp_get_wtime() - start_time) * 1000.0; // ms
                    std::cout << "| " << median_serial_time << " ms\t";
                    med_seq.push_back(med_seq.back() + median_serial_time);

                    // Parallel
                    start_time = omp_get_wtime();
                    median_filter_parallel(w, h, input_img.r, out_median_parallel.r);
                    median_filter_parallel(w, h, input_img.g, out_median_parallel.g);
                    median_filter_parallel(w, h, input_img.b, out_median_parallel.b);
                    double median_parallel_time = (omp_get_wtime() - start_time) * 1000.0; // ms
                    std::cout << median_parallel_time << " ms" << std::endl;
                    med_pl.push_back(med_pl.back() + median_parallel_time);
                    
                    // Verify
                    bool median_ok = verify_channels(out_median_serial.r, out_median_parallel.r, "Median Red") &&
                                     verify_channels(out_median_serial.g, out_median_parallel.g, "Median Green") &&
                                     verify_channels(out_median_serial.b, out_median_parallel.b, "Median Blue");
                    if (median_ok) {
                        //std::cout << "    Verification    : PASSED" << std::endl;
                        //std::cout << "    Speedup         : " << (median_serial_time / median_parallel_time) << "x" << std::endl;
                        write_ppm(out_median_file, out_median_parallel);
                    } else {
                        //std::cout << "    Verification    : FAILED!" << std::endl;
                    }
                    //std::cout << std::endl;

                   // std::cout << "============================================" << std::endl;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    std::cout << "============================================" << std::endl;
    exportToCSV("benchmark_results.csv", gauss_seq, gauss_pl, med_seq, med_pl);

	return 0;
}
