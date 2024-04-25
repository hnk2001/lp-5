#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Example matrices
    const int N = 2;
    int** A;
    int** B;
    int** C;
    
    // Allocate host memory for matrices
    cudaMallocHost(&A, N * sizeof(int*));
    cudaMallocHost(&B, N * sizeof(int*));
    cudaMallocHost(&C, N * sizeof(int*));
    
    for (int i = 0; i < N; i++) {
        cudaMallocHost(&A[i], N * sizeof(int));
        cudaMallocHost(&B[i], N * sizeof(int));
        cudaMallocHost(&C[i], N * sizeof(int));
    }

    // Initialize matrices A and B
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;

    B[0][0] = 5; B[0][1] = 6;
    B[1][0] = 7; B[1][1] = 8;

    // Perform matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    // Print the result
    std::cout << "Result of matrix multiplication:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        cudaFreeHost(A[i]);
        cudaFreeHost(B[i]);
        cudaFreeHost(C[i]);
    }
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
