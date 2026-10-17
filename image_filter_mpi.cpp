#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mpi.h>

// Coordinate clamping for horizontal boundaries
inline int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Optimized insertion sort for a static 9-element array
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

// Structure for RGB Image (Structure of Arrays)
struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> r;
    std::vector<unsigned char> g;
    std::vector<unsigned char> b;
};

// PPM P6 (Binary RGB) Image Loader - Executed by Rank 0
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

    file.get(); // Skip single whitespace character after header

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

// PPM P6 Image Writer - Executed by Rank 0
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

// Local 3x3 Gaussian Kernel operating on padded slice [1 .. local_h]
void gaussian_kernel_local(int width, int local_h, const std::vector<unsigned char>& padded_in, std::vector<unsigned char>& local_out) {
    for (int y = 1; y <= local_h; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = clamp(x + kx, 0, width - 1);
                    int py = y + ky;
                    int weight = (ky == 0 && kx == 0) ? 4 :
                    ((ky == 0 || kx == 0) ? 2 : 1);
                    sum += padded_in[py * width + px] * weight;
                }
            }
            local_out[(y - 1) * width + x] = static_cast<unsigned char>(sum / 16);
        }
    }
}

// Local 3x3 Median Kernel operating on padded slice [1 .. local_h]
void median_kernel_local(int width, int local_h, const std::vector<unsigned char>& padded_in, std::vector<unsigned char>& local_out) {
    for (int y = 1; y <= local_h; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char window[9];
            int count = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = clamp(x + kx, 0, width - 1);
                    int py = y + ky;
                    window[count++] = padded_in[py * width + px];
                }
            }
            sort9(window);
            local_out[(y - 1) * width + x] = window[4];
        }
    }
}

// Ghost Row / Halo Exchange among adjacent MPI ranks
void exchange_halo_rows(int rank, int size, int width, int local_h, std::vector<unsigned char>& padded_channel) {
    MPI_Status status;

    // 1. Top Halo Exchange
    if (rank > 0) {
        MPI_Sendrecv(
            padded_channel.data() + 1 * width, width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     padded_channel.data() + 0 * width, width, MPI_UNSIGNED_CHAR, rank - 1, 1,
                     MPI_COMM_WORLD, &status
        );
    } else {
        std::memcpy(padded_channel.data() + 0 * width, padded_channel.data() + 1 * width, width);
    }

    // 2. Bottom Halo Exchange
    if (rank < size - 1) {
        MPI_Sendrecv(
            padded_channel.data() + local_h * width, width, MPI_UNSIGNED_CHAR, rank + 1, 1,
                     padded_channel.data() + (local_h + 1) * width, width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, &status
        );
    } else {
        std::memcpy(padded_channel.data() + (local_h + 1) * width, padded_channel.data() + local_h * width, width);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Hello" << std::endl;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::filesystem::directory_iterator it;
    std::filesystem::directory_iterator end_it;

    std::filesystem::path input_folder = argv[1];
    std::filesystem::path output_folder = argv[2];

    if (rank == 0) {
        try {
            if (std::filesystem::exists(input_folder) && std::filesystem::is_directory(input_folder)) {
                it = std::filesystem::directory_iterator(input_folder);
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error reading directory: " << e.what() << std::endl;
        }
    }

    while (true) {
        int has_file = 0;
        std::string input_file, out_gauss_file, out_median_file;

        if (rank == 0) {
            while (it != end_it && !(std::filesystem::is_regular_file(it->status()) && it->path().extension() == ".ppm")) {
                ++it;
            }

            if (it != end_it) {
                has_file = 1;
                std::filesystem::path file_path = it->path();
                input_file = file_path.string();
                out_gauss_file = (output_folder / (file_path.stem().string() + "_gauss" + file_path.extension().string())).string();
                out_median_file = (output_folder / (file_path.stem().string() + "_median" + file_path.extension().string())).string();
                std::cout << input_file << std::endl;
                ++it;
            }
        }

        // Broadcast file existence status to all ranks
        MPI_Bcast(&has_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!has_file) break; // Exit loop when no more files remain

        Image input_img;
        int width = 0, height = 0;

        if (rank == 0) {
            if (!read_ppm(input_file, input_img)) {
                std::cerr << "Failed to read PPM: " << input_file << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            width = input_img.width;
            height = input_img.height;
        }

        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 1D Row Domain Decomposition
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);
        std::vector<int> rows_per_rank(size);

        int base_rows = height / size;
        int remainder = height % size;

        for (int r = 0; r < size; ++r) {
            rows_per_rank[r] = base_rows + (r < remainder ? 1 : 0);
            int start_row = r * base_rows + std::min(r, remainder);
            sendcounts[r] = rows_per_rank[r] * width;
            displs[r] = start_row * width;
        }

        int local_h = rows_per_rank[rank];

        int padded_num_bytes = (local_h + 2) * width;
        std::vector<unsigned char> padded_r(padded_num_bytes);
        std::vector<unsigned char> padded_g(padded_num_bytes);
        std::vector<unsigned char> padded_b(padded_num_bytes);

        int local_num_bytes = local_h * width;
        std::vector<unsigned char> local_out_r(local_num_bytes);
        std::vector<unsigned char> local_out_g(local_num_bytes);
        std::vector<unsigned char> local_out_b(local_num_bytes);

        unsigned char* scatter_dest_r = padded_r.data() + 1 * width;
        unsigned char* scatter_dest_g = padded_g.data() + 1 * width;
        unsigned char* scatter_dest_b = padded_b.data() + 1 * width;

        // 1. GAUSSIAN FILTER
        MPI_Barrier(MPI_COMM_WORLD);
        double start_gauss_mpi = MPI_Wtime();

        MPI_Scatterv(rank == 0 ? input_img.r.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     scatter_dest_r, local_num_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? input_img.g.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     scatter_dest_g, local_num_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? input_img.b.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     scatter_dest_b, local_num_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        exchange_halo_rows(rank, size, width, local_h, padded_r);
        exchange_halo_rows(rank, size, width, local_h, padded_g);
        exchange_halo_rows(rank, size, width, local_h, padded_b);

        gaussian_kernel_local(width, local_h, padded_r, local_out_r);
        gaussian_kernel_local(width, local_h, padded_g, local_out_g);
        gaussian_kernel_local(width, local_h, padded_b, local_out_b);

        Image mpi_gauss_result = {width, height, std::vector<unsigned char>(width * height),
            std::vector<unsigned char>(width * height), std::vector<unsigned char>(width * height)};

            MPI_Gatherv(local_out_r.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                        rank == 0 ? mpi_gauss_result.r.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_out_g.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                        rank == 0 ? mpi_gauss_result.g.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_out_b.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                        rank == 0 ? mpi_gauss_result.b.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            double end_gauss_mpi = MPI_Wtime();

            if (rank == 0) {
                write_ppm(out_gauss_file, mpi_gauss_result);
            }

            // 2. MEDIAN FILTER
            MPI_Barrier(MPI_COMM_WORLD);
            double start_median_mpi = MPI_Wtime();

            exchange_halo_rows(rank, size, width, local_h, padded_r);
            exchange_halo_rows(rank, size, width, local_h, padded_g);
            exchange_halo_rows(rank, size, width, local_h, padded_b);

            median_kernel_local(width, local_h, padded_r, local_out_r);
            median_kernel_local(width, local_h, padded_g, local_out_g);
            median_kernel_local(width, local_h, padded_b, local_out_b);

            Image mpi_median_result = {width, height, std::vector<unsigned char>(width * height),
                std::vector<unsigned char>(width * height), std::vector<unsigned char>(width * height)};

                MPI_Gatherv(local_out_r.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                            rank == 0 ? mpi_median_result.r.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
                MPI_Gatherv(local_out_g.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                            rank == 0 ? mpi_median_result.g.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
                MPI_Gatherv(local_out_b.data(), local_num_bytes, MPI_UNSIGNED_CHAR,
                            rank == 0 ? mpi_median_result.b.data() : NULL, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

                MPI_Barrier(MPI_COMM_WORLD);
                double end_median_mpi = MPI_Wtime();

                if (rank == 0) {
                    double mpi_total_time = (end_median_mpi - start_median_mpi) * 1000.0;
                    std::cout << "Processed file: " << input_file << " in " << mpi_total_time << " ms\n";
                    write_ppm(out_median_file, mpi_median_result);
                }
    }

    MPI_Finalize();
    return 0;
}
