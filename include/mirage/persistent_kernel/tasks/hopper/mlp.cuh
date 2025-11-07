#pragma once
#include "tasks/common/common_header.cuh"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>





namespace kernel {

    template <typename T>
    __device__ void add_kernel(void const *A_ptr, void const *B_ptr, void *C_ptr, int rows, int cols) {
        int tid = threadIdx.x;
        int total = rows * cols;

        T const *A = static_cast<T const*>(A_ptr);
        T const *B = static_cast<T const*>(B_ptr);
        T *C = static_cast<T*>(C_ptr);

        if (tid < total) {
            C[tid] = A[tid] + B[tid];
        }
    }

    template <typename T>
    __device__ void subtract_kernel(void const *A_ptr, void const *B_ptr, void *C_ptr, int rows, int cols) {
        int tid = threadIdx.x;
        int total = rows * cols;

        T const *A = static_cast<T const*>(A_ptr);
        T const *B = static_cast<T const*>(B_ptr);
        T *C = static_cast<T*>(C_ptr);

        if (tid < total) {
            C[tid] = A[tid] - B[tid];
        }
    }

    template <typename T>
    __device__ void transpose_kernel(void const *A_ptr, void *At_ptr, int rows, int cols) {
        int tid = threadIdx.x;
        int total = rows * cols;

        T const *A = static_cast<T const*>(A_ptr);
        T *At = static_cast<T*>(At_ptr);

        if (tid < total) {
            int r = tid / cols;
            int c = tid % cols;
            At[c * rows + r] = A[r * cols + c];
        }
    }


}

