/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <map>
#include <string>
#include <iostream>

using namespace std;

// Performance Benchmark
#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 10000
#ifdef USE_CUSPARSE
#define CUSPARSE_ITERATION 10
#endif
// Problem size
//#define M_GLOBAL    (36*1024)                               //
//#define K_GLOBAL    (9*1024)                                // must be 64X
//#define N_GLOBAL    16                                     // must be 8X
//#define SPLIT_K     3
//#define MATRIX_A_PRUNING_PERCENTAGE 80

//printf("\n\
//        CUSPARSE_SPMM_ALG_DEFAULT      = 0,\n\
//        CUSPARSE_SPMM_COO_ALG1         = 1,\n\
//        CUSPARSE_SPMM_COO_ALG2         = 2,\n\
//        CUSPARSE_SPMM_COO_ALG3         = 3,\n\
//        CUSPARSE_SPMM_COO_ALG4         = 5,\n\
//        CUSPARSE_SPMM_CSR_ALG1         = 4,\n\
//        CUSPARSE_SPMM_CSR_ALG2         = 6,\n\
//        CUSPARSE_SPMM_CSR_ALG3         = 12,\n\
//        CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13\n");

void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",                                             \
                   __LINE__,                                                                                           \
                   cusparseGetErrorString(status),                                                                     \
                   status);                                                                                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }
#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess) {                                                                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

// void checkCusparseError(cusparseStatus_t status, int line) {
//    if (status != CUSPARSE_STATUS_SUCCESS) {
//        printf("CuSparse Error at line %d, Error Code: %d\n", line, status);
//        exit(EXIT_FAILURE);
//    }
//}

#define CUDA_CALL(code)                                                                                                \
    do {                                                                                                               \
        cudaError_t status = code;                                                                                     \
        std::string err    = cudaGetErrorString(status);                                                               \
        CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err;                                                        \
    } while (0)

void checkCudaError(cudaError_t error, int line)
{
    if (error != cudaSuccess) {
        printf("Cuda Error at line %d, Error Code: %d\n", line, error);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int rand_int()
{
    // rand int between 0 and 3
    return rand() % 4;
}

__host__ half __inline__ rand_half()
{
    return __float2half_rn(static_cast<float>((rand() / (float)RAND_MAX)));
}

__host__ void
init_host_matrices(half* a, half* b, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            int r = rand() % 100;
            if (r >= MATRIX_A_PRUNING_PERCENTAGE)
                a[j + i * K_GLOBAL] = __float2half_rn(1.0f);
                // a[j + i * K_GLOBAL] = rand_half();
            else
                a[j + i * K_GLOBAL] = __float2half_rn(0.0f);
        }
    }

    for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++){
        b[i] = __float2half_rn((static_cast<float>((rand() % 5)) / 5 - 0.5f)*10);
        // b[i] = __float2half_rn(2.0f);
    }
}


__host__ void init_host_structure_sparsity(
    half* a, half* b, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE)
{
    // randomize the values

    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j += 4) {
            a[i * K_GLOBAL + j]     = 0.0;
            a[i * K_GLOBAL + j + 1] = 0.0;
            a[i * K_GLOBAL + j + 2] = 0.0;
            a[i * K_GLOBAL + j + 3] = 0.0;

            // int value1 = rand_int();
            // int value2 = rand_int();
            int value1 = 0;
            int value2 = 1;

            value2 = value2 == value1 ? (value2 + 1) % 4 : value2;

            // a[i + value1] = rand_half();
            // a[i + value2] = rand_half();
            // if (rand() % 100 >= MATRIX_A_PRUNING_PERCENTAGE) {
            //     a[i * K_GLOBAL + j + 0] = rand_half();
            // }
            // if (rand() % 100 >= MATRIX_A_PRUNING_PERCENTAGE) {
            //     a[i * K_GLOBAL + j + 1] = rand_half();
            // }
            // if (rand() % 100 >= MATRIX_A_PRUNING_PERCENTAGE) {
            //     a[i * K_GLOBAL + j + 2] = rand_half();
            // }
            // if (rand() % 100 >= MATRIX_A_PRUNING_PERCENTAGE) {
            //     a[i * K_GLOBAL + j + 3] = rand_half();
            // }
            // else {
                a[i * K_GLOBAL + j + 0] = __float2half(i + 1);
                a[i * K_GLOBAL + j + 1] = __float2half(i + 1);
                a[i * K_GLOBAL + j + 2] = __float2half(i + 1);
                a[i * K_GLOBAL + j + 3] = __float2half(i + 1);
            // }
                // a[i * K_GLOBAL + j + value1] = __float2half(1.0f);
                // a[i * K_GLOBAL + j + value2] = __float2half(1.0f);
            // a[i * K_GLOBAL + j + 2] = __float2half(1.0f);
        }
    }

    for (int j = 0; j < N_GLOBAL; j++) {
        for (int i = 0; i < K_GLOBAL; i++) {
            // b[j * K_GLOBAL + i] = __float2half_rn((static_cast<float>((rand() % 5)) / 5 - 0.5f)*10);
            // b[j * K_GLOBAL + i] = rand_half();
            // b[j * K_GLOBAL + i] = __float2half(j + 1 + i * N_GLOBAL);
            b[j * K_GLOBAL + i] = __float2half(1.0f);
        }
    }
//     for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
// //        b[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
//             b[i] = __float2half(1.0f);
//         //   b[i] = rand_half();
}

void ComputeError(half* CuBlas, half* Other, int m, int n, double* AvgError, double* MaxError)
{
    double totalError = 0.0;
    double AllError = 0.0;
    double maxError   = 0.0;
    for (int i = 0; i < m * n; i++) {
        double error = fabs(__half2float(CuBlas[i]) - __half2float(Other[i]));
        if (__half2float(CuBlas[i]) != 0.0)
            totalError += fabs(error / __half2float(CuBlas[i]));
        AllError += fabs(error);
        if (error > maxError)
            maxError = error;
        // if (error > 0.01 && i < 10) printf("i=%d, => %d %d CuBlas=%f, Other=%f %f\n", i,  i / m,  i % m, __half2float(CuBlas[i]), __half2float(Other[i]), error);
    }
    printf("Total Error: %f\n", AllError);
    *AvgError = totalError * 100 / (m * n);
    *MaxError = maxError;
}

void PrintMismatch(const char* KernelName,
                   int         MaxNumMismatch,
                   float       ErrorThreshold,
                   half*       CuBlas,
                   half*       Other,
                   int         M_GLOBAL,
                   int         N_GLOBAL)
{
    printf("First 10 Mismatches between Cublas and %s:\n", KernelName);
    int count = 0;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            if (fabs(__half2float(CuBlas[i + j * M_GLOBAL]) - __half2float(Other[i + j * M_GLOBAL])) > ErrorThreshold) {
                count++;
                printf("(%d,%d) CuBlas=%f %s=%f\n",
                       i,
                       j,
                       __half2float(CuBlas[i + j * M_GLOBAL]),
                       KernelName,
                       __half2float(Other[i + j * M_GLOBAL]));
            }
            if (count == MaxNumMismatch)
                break;
        }
        if (count == MaxNumMismatch)
            break;
    }
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double avgError, double maxError)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t AvgError/%%: %.2lf \t MaxError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           avgError,
           maxError);
}
