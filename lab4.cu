#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>

using namespace std;

__global__ void matrixMulKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

vector<vector<int>> generate_matrix(int size, int seed) {
    vector<vector<int>> matrix(size, vector<int>(size));

    std::mt19937 engine(static_cast<unsigned int>(time(nullptr)) + seed);
    std::uniform_int_distribution<int> dist(0, RAND_MAX);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dist(engine);
        }
    }

    return matrix;
}

void write_to_file(const vector<vector<int>>& matrix, const string& path) {
    ofstream out(path);
    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << endl;
    }
}

vector<vector<int>> read_from_file(const string& path) {
    ifstream in(path);
    vector<vector<int>> matrix;
    string line;

    while (getline(in, line)) {
        istringstream iss(line);
        vector<int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

vector<vector<int>> multiply_matrices_cuda(const vector<vector<int>>& matrix1,
                                           const vector<vector<int>>& matrix2) {
    size_t N = matrix1.size();

    vector<int> flatA(N * N);
    vector<int> flatB(N * N);
    vector<int> flatC(N * N, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = matrix1[i][j];
            flatB[i * N + j] = matrix2[j][i]; // Transpose B for better memory access pattern
        }
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, flatA.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(flatC.data(), d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    vector<vector<int>> result(N, vector<int>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = flatC[i * N + j];
        }
    }

    return result;
}

int main(int argc, char** argv) {
    vector<int> counts = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                          1000, 1500, 2000, 3000};

    for (const auto& count : counts) {
        for (int i = 1; i < 3; ++i) {
            vector<vector<int>> matrix = generate_matrix(count, i);
            string path = to_string(i) + "_" + to_string(count) + ".txt";
            write_to_file(matrix, path);
        }
    }

    vector<double> times(counts.size(), 0.0);

    for (size_t i = 0; i < counts.size(); ++i) {
        int count = counts[i];
        string path_1 = "1_" + to_string(count) + ".txt";
        string path_2 = "2_" + to_string(count) + ".txt";
        string result_path = "result_" + to_string(count) + ".txt";

        auto start_time = chrono::steady_clock::now();

        vector<vector<int>> matrix_1 = read_from_file(path_1);
        vector<vector<int>> matrix_2 = read_from_file(path_2);

        vector<vector<int>> result = multiply_matrices_cuda(matrix_1, matrix_2);

        write_to_file(result, result_path);

        auto end_time = chrono::steady_clock::now();
        times[i] = chrono::duration<double, milli>(end_time - start_time).count();
        cout << "Size " << count << " multiplication took " << times[i] << " ms" << endl;
    }

    ofstream out("stats.txt");
    for (size_t i = 0; i < counts.size(); ++i) {
        out << counts[i] << ": " << times[i] << "\n";
    }

    return 0;
}