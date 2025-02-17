#pragma once

template <typename T>
ctcStatus_t reduce_negate(const T* input, T* output, int cols, int rows, bool axis, gpuStream_t stream);
template <typename T>
ctcStatus_t reduce_exp(const T* input, T* output, int cols, int rows, bool axis, gpuStream_t stream);
template <typename T>
ctcStatus_t reduce_max(const T* input, T* output, int cols, int rows, bool axis, gpuStream_t stream);
