// Includes, system
// #include <stdio.h>
// #include <stdlib.h>

// Includes, cuda
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// Includes, cuda helper functions
// #include <helper_cuda.h>

// For the functors
#include "detail/ctc_helper.h"
#include "ctc.h"

const int warp_size = 32;

template<int NT, typename T, typename Rop>
struct CTAReduce;


// Reduce in tiled. The shared mem capacity is same to tiled size.
// Rop: Reduce OP
// T: element type
// NT: num of T elements. shared memory capacity. e.g 128
template<int NT, typename T, typename Rop>
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    __device__ static T reduce(int tid, T x, Storage& storage, int count, Rop g) {
        // tid, thread id
        // x, input T element
        // storage, shared memory, capacity is NT
        // count, total count of elements
        // g, gather func

        T* s = storage.shared;
        s[tid] = x;
        __syncthreads(); // sync threads in the thread block

        // Fold the data in half with each pass. tree-reduction.
        // 1. reduce in tiled.
#pragma unroll 
//https://www.ibm.com/docs/en/zos/2.3.0?topic=descriptions-pragma-unroll
        for(int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if(tid < offset && tid + offset < count) {
                // Read from the right half and store to the left half using gather func.
                x = g(x, s[tid + offset]);
                s[tid] = x;
            }
            __syncthreads();  // sync threads in the thread block
        }

        // 2. reduce in warp
        T shuff;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
            shuff = __shfl_down_sync(0xFFFFFFFF, x, offset);
#else
            shuff = __shfl_down(x, offset);
#endif
            if (tid < offset && tid + offset < count)
                x = g(x, shuff);
        }
        return x;
    }
};

// reduce cols in global matrix
// Iop: element-wise/inplace op
// Rop: Reduce op
// NT: tile size，shared memory size.
// input: row major mem, (R, C). (T,B,V) mem layout is same to (T*B, V)
// ouput: (R,), reduce on C axis
// num_cols: C
// num_rows: R
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_cols(Iop f, Rop g, const T* input, T* output,
                            int num_cols, int num_rows) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x; // 128, on C axis
    int idx = tid;
    int row = blockIdx.x; // T*B, on R axis
    T curr;

    // Each block works on a column
    // 1. reduce between tiled_partition
    if (idx < num_cols)
        curr = f(input[row * num_cols + idx]);
    idx += NT;

    while (idx < num_cols) {
        curr = g(curr, f(input[row * num_cols + idx]));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    // 2. reduce in tiled and warp
    curr = R::reduce(tid, curr, storage, num_cols, g);

    // Store result in out
    if (tid == 0)
        output[row] = curr;
}


// reduce rows in tiled partition, tile size is 128.
// NT: shared memory size
// Iop: inplace func
// Rop: reduce func
// T: element type
// input: row major mem, (R, C=warp_size).
// output: (C,), reduce on axis R.
// num_cols: C
// num_rows, R
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T* input, T* output,
                            int num_cols, int num_rows) {
    __shared__ T s[NT]; 

    // threadIdx is 2dim (warp_size, 128/warp_size)
    // blockIndx is 1dim, ((num_rows + warp_size - 1) / warp_size)
    int warps_per_block = NT / warp_size;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = threadIdx.y; // 128/warp_size
    T curr;

    // reduce on row axis tiled, tile size = blockDim.y = 128/warp_size
    if (row < num_rows && col < num_cols) {
        curr = f(input[row * num_cols + col]);
        row += blockDim.y; // 128/warp_size
        while (row < num_rows) {
            curr = g(curr, f(input[row * num_cols + col]));
            row += blockDim.y;
        }
    }

    //using s as shape (warp_size, 128/warp_size)
    s[threadIdx.x * warps_per_block + threadIdx.y] = curr;
    __syncthreads();

    // Reduce
    if (threadIdx.y == 0 && col < num_cols) {
#pragma unroll
        for (int i = 1; i < warps_per_block && i < num_rows; ++i)
            curr = g(curr, s[threadIdx.x * warps_per_block + i]);
        output[col] = curr;
    }
}


struct ReduceHelper {
    // T - type
    // Iof - element/inplcae func
    // Rof - gather func
    template<typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T* input, T* output, int num_cols, int num_rows, bool axis, gpuStream_t stream) {
        // f = inplcae of func
        // g = gather of func
        // input, (T,B,V)
        // output, (T*B,)
        // num_cols, (V,)
        // num_rows, (T*B,)
        // axis=1

        // work on matrix, (T*B, V)
        int grid_size;  // num blocks

        // if axis = True, reduce on axis=1, else reduce on axis=0
        if (axis) {
            // reduce on cols, (num_rows, 1)
            // softmax reduce_max
            grid_size = num_rows; // num_blocks
            reduce_cols<128 /*tile size*/><<<grid_size, 128, 0, stream>>>
               (f, g, input, output, num_cols, num_rows);
        } else {
            // reduce on rows, (1, num_cols)
            dim3 tpb(warp_size, 128 / warp_size); // num_threads
            grid_size = (num_rows + warp_size - 1) / warp_size; // num_blocks
            reduce_rows<128 /*tile size*/><<<grid_size, tpb, 0, stream>>>
                (f, g, input, output, num_cols, num_rows);
        }
    }
};


// T - type
// Iof - element/inplcae func
// Rof - gather func
template<typename T, typename Iof, typename  Rof>
ctcStatus_t reduce(Iof f, Rof g, const T* input, T* output, int cols, int rows, bool axis, gpuStream_t stream) {
    ReduceHelper::impl(f, g, input, output, cols, rows, axis, stream);

#ifdef __HIPCC__
    hipStreamSynchronize(stream);
    gpuError_t err = hipGetLastError();
#else
    cudaStreamSynchronize(stream);
    gpuError_t err = cudaGetLastError();
#endif

    if (err != gpuSuccess)
        return CTC_STATUS_EXECUTION_FAILED;

    return CTC_STATUS_SUCCESS;
}

template<typename T>
ctcStatus_t reduce_negate(const T *input, T *output, int cols, int rows, bool axis, gpuStream_t stream) {
    return reduce(ctc_helper::negate<T>(), ctc_helper::add<T>(), input, output, cols, rows, axis, stream);
}
template ctcStatus_t reduce_negate<float>(const float *input, float *output, int cols, int rows, bool axis, gpuStream_t stream);
template ctcStatus_t reduce_negate<double>(const double *input, double *output, int cols, int rows, bool axis, gpuStream_t stream);

template<typename T>
ctcStatus_t reduce_exp(const T *input, T *output, int cols, int rows, bool axis, gpuStream_t stream) {
    return reduce(ctc_helper::exponential<T>(), ctc_helper::add<T>(), input, output, cols, rows, axis, stream);
}
template ctcStatus_t reduce_exp<float>(const float *input, float *output, int cols, int rows, bool axis, gpuStream_t stream);
template ctcStatus_t reduce_exp<double>(const double *input, double *output, int cols, int rows, bool axis, gpuStream_t stream);

template<typename T>
ctcStatus_t reduce_max(const T *input, T *output, int cols, int rows, bool axis, gpuStream_t stream) {
    return reduce(ctc_helper::identity<T>(), ctc_helper::maximum<T>(),input, output, cols, rows, axis, stream);
}
template ctcStatus_t reduce_max<float>(const float *input, float *output, int cols, int rows, bool axis, gpuStream_t stream);
template ctcStatus_t reduce_max<double>(const double *input, double *output, int cols, int rows, bool axis, gpuStream_t stream);
