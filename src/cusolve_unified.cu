#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>


#define CheckCudaError(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
  do {                                                                                           \
      cusolverStatus_t err_ = (err);                                                             \
      if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
          printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
          throw std::runtime_error("cusolver error");                                            \
      }                                                                                          \
  } while (0)


// Timer setup
#include <sys/time.h>
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Initialize matrix from file input
void InitMat(FILE* fp, double* ary, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (fscanf(fp, "%lf", &ary[size * j + i]) != 1) {
                perror("Failed to read the matrix");
                exit(1);
            }
        }
    }
}

// Initialize array from file input
void InitAry(FILE* fp, double* ary, int size) {
    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%lf", &ary[i]) != 1) {
            perror("Failed to read the array");
            exit(1);
        }
    }
}

void PrintMat(double* ary, int size)
{
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf("%8.2f ", *(ary + size * i + j));
        }
        printf("\n");
    }
    printf("\n");
}

void PrintAry(double* ary, int size)
{
    for (int i = 0; i < size; i++) {
        printf("%8.2f ", *(ary + i));
    }
}

void InitProblemOnce(char* filename, int* size, double** a, double** b, double** slnVec, double** X) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Unable to open the file");
        exit(1);
    }

    printf("Read input from is: %s\n", filename);

    if (fscanf(fp, "%d", size) != 1) {
        perror("Failed to read the size");
        exit(1);
    }
    printf("The input matrix A's size is: %d\n", *size);

    // Allocate memory for a and b
    cudaMallocManaged(a, *size * *size * sizeof(double));
    cudaMallocManaged(b, *size * sizeof(double));

    // Advise the CUDA memory manager to set the preferred location for a and b
    cudaMemAdvise(a, *size * *size * sizeof(double), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, *size * sizeof(double), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    InitMat(fp, *a, *size);
    InitAry(fp, *b, *size);
    *slnVec = (double*)malloc((*size) * sizeof(double));
    InitAry(fp, *slnVec, *size);
}

int main(int argc, char* argv[]) {
    //begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    int size;
    double* A, * B, * slnVec, * X;
    int id = cudaGetDevice(&id);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        exit(1);
    }

    char* filename = argv[1];

    InitProblemOnce(filename, &size, &A, &B, &slnVec, &X);

    // PrintMat(A, size);
    // PrintAry(B, size);
    // printf("\n");

    // printf("Solution: ");
    // PrintAry(slnVec, size);
    // printf("\n");

    // Initialize cuSolver
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    // Allocate memory for devIpiv and devInfo
    int* devIpiv, * devInfo;
    cudaMallocManaged(&devIpiv, size * sizeof(int));
    cudaMallocManaged(&devInfo, sizeof(int));

    // Advise the CUDA memory manager to set the preferred location for devIpiv, devInfo, A and B
    cudaMemAdvise(devIpiv, size * sizeof(int), cudaMemAdviseSetPreferredLocation, id);
    cudaMemAdvise(devInfo, sizeof(int), cudaMemAdviseSetPreferredLocation, id);
    cudaMemAdvise(A, size * size * sizeof(double), cudaMemAdviseSetPreferredLocation, id);
    cudaMemAdvise(B, size * sizeof(double), cudaMemAdviseSetPreferredLocation, id);

    // Asynchronously prefetch A and B to GPU
    cudaMemPrefetchAsync(A, size * size * sizeof(double), id);
    cudaMemPrefetchAsync(B, size * sizeof(double), id);

    // Allocate memory for workspace
    double* workspace;
    int workspace_size;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, size, size, A, size, &workspace_size));
    cudaDeviceSynchronize();
    cudaMallocManaged(&workspace, workspace_size * sizeof(double));
    cudaMemAdvise(workspace, workspace_size * sizeof(double), cudaMemAdviseSetPreferredLocation, id);

    // LU factorization
    double start = cpuSecond();
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, size, size, A, size, workspace, devIpiv, devInfo));
    cudaDeviceSynchronize();
    double stop = cpuSecond();
    printf("Time for LU factorization: %f sec\n", stop - start);

    // Solve Ax = B
    start = cpuSecond();
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, size, 1, A, size, devIpiv, B, size, devInfo));
    cudaDeviceSynchronize();
    stop = cpuSecond();
    printf("Time for solving Ax = B: %f sec\n", stop - start);

    // Asynchronously prefetch B to CPU
    cudaMemPrefetchAsync(B, size * sizeof(double), cudaCpuDeviceId);

    // Print solution
    // printf("Solution:\n");
    bool isCorrect = true;
    for (int i = 0; i < size; i++) {
        // printf("%f ", B[i]);
        if (fabs(B[i] - slnVec[i]) > 1e-5) {
            isCorrect = false;
            // break;
        }
    }

    printf("\n");
    std::cout << "Results are " << (isCorrect ? "correct" : "incorrect") << std::endl;

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    free(slnVec);
    cudaFree(workspace);
    cudaFree(devIpiv);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);

    //end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);

    return 0;
}
