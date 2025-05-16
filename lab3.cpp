#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;

vector<vector<int>> generate_matrix(int size, int rank) {
    vector<vector<int>> matrix(size, vector<int>(size));

    std::mt19937 engine(static_cast<unsigned int>(time(nullptr)) + rank);
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

vector<vector<int>> mul_matrix(const vector<vector<int>>& matrix1,
                               const vector<vector<int>>& matrix2,
                               int rank, int size) {
    size_t n = matrix1.size();
    vector<vector<int>> result(n, vector<int>(n, 0));

    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    for (int i = start_row; i < end_row; ++i) {
        for (int k = 0; k < n; ++k) {
            int temp = matrix1[i][k];
            for (int j = 0; j < n; ++j) {
                result[i][j] += temp * matrix2[k][j];
            }
        }
    }

    if (rank != 0) {
        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(result[i].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_process;
            int p_end = (p == size - 1) ? n : p_start + rows_per_process;

            for (int i = p_start; i < p_end; ++i) {
                MPI_Recv(result[i].data(), n, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> counts = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                          1000, 1500, 2000, 3000};

    if (rank == 0) {
        for (const auto& count : counts) {
            for (int i = 1; i < 3; ++i) {
                vector<vector<int>> matrix = generate_matrix(count, rank);
                string path = to_string(i) + "_" + to_string(count) + ".txt";
                write_to_file(matrix, path);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    vector<double> times(counts.size(), 0.0);

    if (rank == 0) {
        for (size_t i = 0; i < counts.size(); ++i) {
            int count = counts[i];
            string path_1 = "1_" + to_string(count) + ".txt";
            string path_2 = "2_" + to_string(count) + ".txt";
            string result_path = "result_" + to_string(count) + ".txt";

            auto start_time = chrono::steady_clock::now();

            vector<vector<int>> matrix_1 = read_from_file(path_1);
            vector<vector<int>> matrix_2 = read_from_file(path_2);

            int n = matrix_1.size();
            for (int p = 1; p < size; ++p) {
                MPI_Send(&n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

                for (const auto& row : matrix_1) {
                    MPI_Send(row.data(), n, MPI_INT, p, 0, MPI_COMM_WORLD);
                }

                for (const auto& row : matrix_2) {
                    MPI_Send(row.data(), n, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }

            vector<vector<int>> result = mul_matrix(matrix_1, matrix_2, rank, size);

            write_to_file(result, result_path);

            auto end_time = chrono::steady_clock::now();
            times[i] = chrono::duration<double, milli>(end_time - start_time).count();
            cout << "Size " << count << " multiplication took " << times[i] << " ms" << endl;
        }

        ofstream out("stats.txt");
        for (size_t i = 0; i < counts.size(); ++i) {
            out << counts[i] << ": " << times[i] << "\n";
        }
    } else {
        for (size_t i = 0; i < counts.size(); ++i) {
            int n;

            MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vector<vector<int>> matrix_1(n, vector<int>(n));
            for (int i = 0; i < n; ++i) {
                MPI_Recv(matrix_1[i].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            vector<vector<int>> matrix_2(n, vector<int>(n));
            for (int i = 0; i < n; ++i) {
                MPI_Recv(matrix_2[i].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            mul_matrix(matrix_1, matrix_2, rank, size);
        }
    }

    MPI_Finalize();
    return 0;
}