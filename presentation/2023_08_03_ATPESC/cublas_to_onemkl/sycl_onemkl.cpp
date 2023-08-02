////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the ONEMKL library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL ONEMKL API IN A REVERSE ORDER: onemklSegemm(B, A)!
// The reason is explained as follows:

// ONEMKL library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to ONEMKL, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like onemklSgemm(A, B)  because of
// implicit transpose. The actual result of onemklSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based onemkl matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In onemkl format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Onemkl matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// ONEMKL provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <assert.h>
#include <cstdio>
#include <cmath>
#include <chrono>

// To compile: clang++ -std=c++17 -sycl-std=2020 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -L$MKLROOT/lib -lonemkl sycl_onemkl.cpp -o sycl_onemkl.out

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

inline bool
sdkCompareL2fe(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    assert(epsilon >= 0);

    float error = 0;
    float ref = 0;

    for (unsigned int i = 0; i < len; ++i)
    {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);

    if (fabs(ref) < 1e-7)
    {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG

    if (! result)
    {
        std::cerr << "ERROR, l2-norm error "
                  << error << " is greater than epsilon " << epsilon << "\n";
    }

#endif

    return result;
}

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple,
                    sMatrixSize &matrix_size) {
    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    // use a larger block size (32) for Fermi and above
    int block_size = 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using ONEMKL
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    sycl::queue q(sycl::gpu_selector_v);
    int block_size = 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_ONEMKL = (float *) malloc(mem_size_C);

    d_A = (float *)sycl::malloc_device(mem_size_A, q);
    d_B = (float *)sycl::malloc_device(mem_size_B, q);
    q.memcpy(d_A, h_A, mem_size_A).wait();
    q.memcpy(d_B, h_B, mem_size_B).wait();
    d_C = (float *)sycl::malloc_device(mem_size_C, q);

    // setup execution parameters
    sycl::range<3> threads(1, block_size, block_size);
    sycl::range<3> grid(1, matrix_size.uiHC / threads[1],
                        matrix_size.uiWC / threads[2]);

    // create and start timer
    printf("Computing result using ONEMKL...");

    // execute the kernel
    int nIter = 30;

    // ONEMKL version
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        std::chrono::time_point<std::chrono::steady_clock> start;
        std::chrono::time_point<std::chrono::steady_clock> stop;

        //Perform warmup operation with onemkl
        oneapi::mkl::blas::column_major::gemm(
            q, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans, matrix_size.uiWB,
            matrix_size.uiHA, matrix_size.uiWA, alpha, d_B, matrix_size.uiWB,
            d_A, matrix_size.uiWA, beta, d_C, matrix_size.uiWB).wait();

        // Record the start event
        start = std::chrono::steady_clock::now();

        for (int j = 0; j < nIter; j++)
        {
            //note onemkl is column primary!
            //need to transpose the order
	    oneapi::mkl::blas::column_major::gemm(
		q, oneapi::mkl::transpose::nontrans,
		oneapi::mkl::transpose::nontrans, matrix_size.uiWB,
		matrix_size.uiHA, matrix_size.uiWA, alpha, d_B,
		matrix_size.uiWB, d_A, matrix_size.uiWA, beta, d_C,
		matrix_size.uiWB).wait();
        }

        printf("done.\n");

        // Record the stop event
	//q.wait();
        stop = std::chrono::steady_clock::now();


        float msecTotal = 0.0f;
	msecTotal = std::chrono::duration<float, std::milli>(stop - start).count();

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // copy result from device to host
	q.memcpy(h_ONEMKL, d_C, mem_size_C).wait();

    }

    // compute reference solution
    printf("Computing result using host CPU...");
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    printf("done.\n");

    // check result (ONEMKL)
    bool resONEMKL = sdkCompareL2fe(reference, h_ONEMKL, size_C, 1.0e-6f);
    if (resONEMKL != true)
    {
        printDiff(reference, h_ONEMKL, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing ONEMKL Matrix Multiply with CPU results: %s\n", (true == resONEMKL) ? "PASS" : "FAIL");

    printf("\nNOTE: The Samples are not meant for performance measurements.\n");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    if (resONEMKL == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[Matrix Multiply ONEMKL] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

    return matrix_result;
}
