#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;

vector<vector<int>> generate_matrix(int size) {
    vector<vector<int>> matrix(size, vector<int>(size));

#pragma omp parallel
    {
        std::mt19937 engine(static_cast<unsigned int>(time(nullptr)) + omp_get_thread_num());
        std::uniform_int_distribution<int> dist(0, RAND_MAX);

#pragma omp for
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = dist(engine);
            }
        }
    }
    return matrix;
}

void write_to_file(const vector<vector<int>>& matrix, const string& path) {
    std::ofstream out(path);
    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << std::endl;
    }
}

vector<vector<int>> read_from_file(const string& path) {
    std::ifstream in(path);
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

vector<vector<int>> mul_matrix(const vector<vector<int>>& matrix1, const vector<vector<int>>& matrix2) {
    size_t size = matrix1.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(size); ++i) {
        for (int k = 0; k < static_cast<int>(size); ++k) {
            int temp = matrix1[i][k];
            for (int j = 0; j < static_cast<int>(size); ++j) {
                result[i][j] += temp * matrix2[k][j];
            }
        }
    }
    return result;
}

int main() {
    vector<int> counts = { 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000 };
    for (const auto& count : counts) {
        for (int i = 1; i < 3; ++i) {
            vector<vector<int>> matrix = generate_matrix(count);
            string path = std::to_string(i) + "_" + std::to_string(count) + ".txt";
            write_to_file(matrix, path);
        }
    }
    vector<double> times(counts.size(), 0.0);
    for (size_t i = 0; i < counts.size(); ++i) {
        int count = counts[i];
        string path_1 = "1_" + std::to_string(count) + ".txt";
        string path_2 = "2_" + std::to_string(count) + ".txt";
        string result_path = "result_" + std::to_string(count) + ".txt";

        auto start_time = chrono::steady_clock::now();
        vector<vector<int>> matrix_1 = read_from_file(path_1);
        vector<vector<int>> matrix_2 = read_from_file(path_2);
        vector<vector<int>> result = mul_matrix(matrix_1, matrix_2);
        write_to_file(result, result_path);
        auto end_time = chrono::steady_clock::now();

        times[i] = chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cout << "Size " << count << " multiplication took " << times[i] << " ms" << std::endl;
    }
    std::ofstream out("stats.txt");
    for (size_t i = 0; i < counts.size(); ++i) {
        out << counts[i] << ": " << (times[i]) << "\n";
    }
}
