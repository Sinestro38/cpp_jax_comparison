/* To compile run:
g++ mat_mul.cc -o mat_mul -lcublas
*/
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <cublas_v2.h>

void matrix_multiply_cublas(cublasHandle_t handle, float *A, float *B, float *C, int N) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);

}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle); // ~150 ms

    const int N = 1000;

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // allocate an array of 1000 length to store time elapsed for each run
    float times_mean = 0;
    const int N_iterations = 1000;

    for (int i = 0; i < N_iterations; i++) {
        for (int i = 0; i < N * N; i++) {
            A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        matrix_multiply_cublas(handle, A, B, C, N); // ~ 8 microseconds
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        times_mean += duration.count();
        // print the duration.count if i < 10
        if (i < 25) {
            std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
        }
    }
    times_mean /= N_iterations;
    // take the mean of the times array

    std::cout << "Time taken: " << times_mean << " microseconds" << std::endl;
    // Print some values of matrix C
    // std::cout << "C[0][0] = " << C[0] << std::endl;
    // std::cout << "C[N/2][N/2] = " << C[(N/2)*N + N/2] << std::endl;
    // std::cout << "C[N-1][N-1] = " << C[(N-1)*N + N-1] << std::endl;

    cublasDestroy(handle);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}








// #include <iostream>
// #include <chrono>
// #include <cuda_runtime.h>
// using namespace std;
// using namespace std::chrono;

// const int N = 1000;

// __global__ void matrix_multiply(int *A, int *B, int *C) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < N && col < N) {
//         int sum = 0;
//         for (int i = 0; i < N; i++) {
//             sum += A[row * N + i] * B[i * N + col];
//         }
//         C[row * N + col] = sum;
//     }
// }

// int main() {
//     int *A, *B, *C;
//     int *d_A, *d_B, *d_C;
//     int size = N * N * sizeof(int);

//     // Allocate memory on the CPU
//     A = (int*)malloc(size);
//     B = (int*)malloc(size);
//     C = (int*)malloc(size);

//     // Initialize matrices A and B with random values
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             A[i * N + j] = rand() % 100;
//             B[i * N + j] = rand() % 100;
//         }
//     }

//     // Allocate memory on the GPU
//     cudaMalloc(&d_A, size);
//     cudaMalloc(&d_B, size);
//     cudaMalloc(&d_C, size);

//     // Start the timer
//     auto start_time = high_resolution_clock::now();

//     // Copy matrices A and B from CPU to GPU
//     cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

//     // Set the number of threads per block
//     int threadsPerBlock = 16;

//     // Calculate the number of blocks in the x and y dimensions
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     // Launch the kernel to multiply matrices A and B on the GPU
//     matrix_multiply<<<dim3(blocksPerGrid, blocksPerGrid), dim3(threadsPerBlock, threadsPerBlock)>>>(d_A, d_B, d_C);

//     // Copy matrix C from GPU to CPU
//     cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

//     // End the timer
//     auto end_time = high_resolution_clock::now();

//     // Calculate the elapsed time and print it in microseconds
//     auto elapsed_time = duration_cast<microseconds>(end_time - start_time).count();
//     cout << "Elapsed time: " << elapsed_time << " mus" << endl;

//     // // Print matrix C
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < N; j++) {
//     //         cout << C[i * N + j] << " ";
//     //     }
//     //     cout << endl;
//     // }

//     // Free memory on the GPU
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     // Free memory on the CPU
//     free(A);
//     free(B);
//     free(C);

//     return 0;
// }
