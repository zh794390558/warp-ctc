#pragma once

#include <contrib/moderngpu/include/device/ctascan.cuh>
#include <contrib/moderngpu/include/device/ctamerge.cuh>

#include "ctc_helper.h"

using namespace mgpu;

// Computes forward probabilities. This fills in a T * S matrix.
// The computation starts at t=1 (2nd row) and ends at t=T-1 (last row). Each row has
// S elements where S = 2L + 1.
//
// We only need to read in probabilities corresponding to the labels, thus a sparse
// set of values are read from the probs matrix since the character set is much smaller
// than the labels. This is much more true for Mandarin than English.
template<typename ProbT, int NT, int VT>
__global__
void compute_alpha_kernel (const ProbT* probs, const int *label_sizes,
                           const int *input_lens, const int *repeats_in_labels,
                           const int *labels_without_blanks, const int *label_offsets,
                           int *labels_with_blanks, ProbT *alphas,
                           ProbT* nll_forward, int stride, int out_dim,
                           int S_memoffset, int T_memoffset, int blank_label) {
    // probs, (T, B, V)
    // alphas, (B, T, S)
    // label_sizes, (B,)
    // input_lens, (B,)
    // repeats_in_labels, (B,)
    // label_offsets, (B,)
    // labels_without_blanks, (B, L)
    // labels_with_blanks, (B, S)
    // nll_forward, (B,)
    // stride = B
    // out_dim = V
    // S_memoffset = S
    // T_memoffset = T
    // grid_dim = B
    // block_dim = NT=128
    // VT=10
    ctc_helper::log_plus<ProbT> log_plus_f;

    const int tid = threadIdx.x;
    // b=blockIdx.x example
    const int L = label_sizes[blockIdx.x]; // label_len
    const int T = input_lens[blockIdx.x]; // input_len
    const int S = 2*L + 1;
    const int prob_offset = blockIdx.x * out_dim; // b * V
    const int repeats = repeats_in_labels[blockIdx.x];

    const int NV = NT * VT; //shared mem size, NT*VT >= S_
    __shared__ int label[NV];

    if ((L + repeats) > T) {
        // invalid example
        return;
    }

    // Generate labels with blanks from labels without blanks
    {
        const int label_start_offset = label_offsets[blockIdx.x];
        for (int idx = tid; idx < L; idx += blockDim.x) {
            const int offset = (blockIdx.x * S_memoffset) + 2 * idx;
            labels_with_blanks[offset] = blank_label;
            labels_with_blanks[offset+1] = labels_without_blanks[label_start_offset + idx];
        }
        if (tid == 0) {
            labels_with_blanks[(blockIdx.x * S_memoffset) + 2 * L] = blank_label;
        }
    }
    __syncthreads();

    const int *labels = labels_with_blanks;
    const int* label_global = &labels[blockIdx.x * S_memoffset];
    ProbT* alpha = &alphas[blockIdx.x * (T_memoffset * S_memoffset)]; // (1, T, S)

    // Set the first row of alpha neg_inf - it is much more efficient to do it
    // here than outside
    #pragma unroll
    for (int idx = tid; idx < min(S, NV); idx += blockDim.x) {
        // alpha point to (1, 1, 1)
        alpha[idx] = ctc_helper::neg_inf<ProbT>();
    }

    // Load labels into shared memory
    #pragma unroll
    for (int i = tid; i < S; i += NT) {
        label[i] = label_global[i];
    }

    __syncthreads();

    int start =  (L + repeats < T) ? 0 : 1; // see Line. 124
    int end = S > 1 ? 2 : 1;

    // Initialize the first row corresponding to t=0;
    for(int i = tid; i < (end - start); i += blockDim.x)
        alpha[start + i] = log(probs[prob_offset + label[start + i]]);

    __syncthreads();

    // Fill in the rest of matrix, one row at a time (outer loop).
    for(int t = 1; t < T; ++t) {

        // Start offsets into the current and previous row
        const int start_cur_row = t * S;
        const int start_prev_row = (t - 1) * S;

        // The prob is a 2D row major array, with probabilites for each t strided
        // by (out_dim * stride), where stride is the minibatch size
        const int start_prob_col = t * (stride * out_dim);

        // This is the first row and in this case there is nothing left of it
        if (tid == 0) {
            if (start == 0) { // start from blank(s=0)
                alpha[start_cur_row] = alpha[start_prev_row] +
                                       log(probs[start_prob_col + prob_offset + blank_label]);
            }
            else if (start == 1) { // start from l'_1(s=1)
                alpha[start_cur_row] = alpha[start_prev_row];
            }
        }

        __syncthreads();

        // Fill in the elements in each row. There is no loop dependence here since our
        // input is the row above. We sum either two or three adjacent values from the
        // row above depending on whether we have a blank or repeated characters. Finally
        // we add the probability corresponding to this label at time t
        #pragma unroll
        for (int idx = (tid+1); idx < S; idx += blockDim.x) {

            ProbT prev_sum = log_plus_f(alpha[start_prev_row + idx], alpha[start_prev_row + (idx-1)]);

            // Skip two if not on blank and not on repeat.
            if ((label[idx] != blank_label) &&
                (idx != 1) && (label[idx] != label[idx-2]))
                prev_sum = log_plus_f(prev_sum, alpha[start_prev_row + (idx-2)]);

            alpha[start_cur_row + idx] =
                prev_sum + log(probs[start_prob_col + prob_offset + label[idx]]);
        } // end for idx

        __syncthreads();
    } // end for t

    if (tid == 0) {
        // Add and return the rightmost two/one element(s) in the last row.
        ProbT loglikelihood = ctc_helper::neg_inf<ProbT>();

        // This is the total increment for s_inc and e_inc through the loop
        const int val = 2 * (L-1) + 1 - (((L + repeats) == T) ? 1 : 0);

        start = (val * (L!=0) + start);
        end = (val * (L!=0) + end);

        for(int i = start; i < end; ++i)
            loglikelihood = log_plus_f(loglikelihood, alpha[(T - 1) * S + i]);

        // nll
        nll_forward[blockIdx.x] = -loglikelihood;
    }
}


template<int NT, int VT, typename T, typename KeyT, typename Op>
struct CTASegReduce {

    enum {NV = NT * VT};

    union Storage {
        typename CTAScan<NT>::Storage scanStorage;
        int indices[NV];
    };

    //adapted from global kernel KernelReduceByKeyPreprocess
    __device__ static void preprocessKeys(KeyT *keys, int count,
                                          int *numUniqueLabels, int seg_start[VT],
                                          int seg_end[VT], int *scanout) {
        __shared__ Storage shared;

        const int tid = threadIdx.x;
        // Compare adjacent keys within each thread and mark discontinuities
        int endFlags = 0;
        T key = keys[VT * tid];
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
            int index = VT * tid + 1 + i;
            T next = keys[index];
            if(index == count || (index < count && key != next)) {
                endFlags |= 1 << i;
            }
            key = next;
        }

        __syncthreads();

        //Count the number of encountered end flags
        int scan = CTAScan<NT>::Scan(tid, popc(endFlags), shared.scanStorage, numUniqueLabels);

        __syncthreads();

        //output the unique keys
        //use indices as scratch space
        int outputPos = scan;
        #pragma unroll
        for (int i = 0; i < VT; ++i) {

            if ( (endFlags >> i) & 1) {
                shared.indices[outputPos] = keys[VT * tid + i];
                scanout[outputPos] = VT * tid + i;
                outputPos++;
            }
        }

        __syncthreads();

        // Create start and end
        for (int idx = tid, j = 0; idx < (*numUniqueLabels); idx += blockDim.x, ++j) {
            seg_start[j] = (idx == 0) ? 0 : (scanout[idx-1] + 1);
            seg_end[j] = scanout[idx];
        }

        __syncthreads();

        //copy from the scratch space back into the keys
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
            keys[i * NT + tid] = shared.indices[i * NT + tid];
        }

        __syncthreads();
    }
};


// Computes backward probabilities. This also fills in a T * S matrix
// See comments above compute_alphas for more context.
template<typename ProbT, int NT, int VT>
__global__
void compute_betas_and_grad_kernel (const ProbT* probs, const int *label_sizes,
                                    const int *input_lens, const int *repeats_in_labels,
                                    const int *labels_with_blanks, ProbT *alphas,
                                    const ProbT* nll_forward, ProbT *nll_backward,
                                    ProbT *grads, int stride, int out_dim,
                                    int S_memoffset, int T_memoffset, int blank_label) {
    // probs, (T, B, V)
    // alphas, (B, T, S)
    // grads, (T, B, V)
    // label_sizes, (B,)
    // input_lens, (B,)
    // repeats_in_labels, (B,)
    // label_offsets, (B,)
    // labels_without_blanks, (B, L)
    // labels_with_blanks, (B, S)
    // nll_forward, (B,)
    // nll_backward, (B,)
    // stride = B
    // out_dim = V
    // S_memoffset = S
    // T_memoffset = T
    // blank_label = 0

    // grid_dim = B
    // block_dim = NT=128
    // VT=10, used for Segment Reduce
    // shared mem (NT, VT)
    ctc_helper::log_plus<ProbT> log_plus_f; // logsumexp
    typedef CTASegReduce<NT, VT, ProbT, int, ctc_helper::log_plus<ProbT>> SegReduce;

    const int tid = threadIdx.x;
    const int L = label_sizes[blockIdx.x]; // batch b
    const int T = input_lens[blockIdx.x]; // batch b
    const int repeats = repeats_in_labels[blockIdx.x]; // batch b
    const ProbT log_partition = -nll_forward[blockIdx.x]; // batch b, log likelihood
    const int S = 2*L + 1;
    const int prob_offset = blockIdx.x * out_dim; // b * V
   
    const int* labels = labels_with_blanks;
    const int* label_global = &labels[blockIdx.x * S_memoffset]; // batch b
    ProbT* alpha = &alphas[blockIdx.x * (T_memoffset * S_memoffset)]; // batch b

    const int NV = NT * VT;  //shared mem size, NT*VT >= S_
    union TempStorage { // S
        ProbT beta[NV];
        int result[NV];
    };
    __shared__ TempStorage temp_buffer;
    __shared__ int label[NV];

    // Temporaries needed for segmented reduce
    // TODO: see if we can combine the shared memory requirements
    __shared__ int keys_shared[NV];
    __shared__ int gather_indices[NV];
    __shared__ ProbT output[NV]; // output(s) = \alpha_t(s) * \beat_t(s)

    ProbT beta_val[VT];

    if ((L + repeats) > T) // invalid example
        return;

    // (L + repeats) <= T

    // backward
    // start, end are index in S
    // end is sentinel
    int start = S > 1 ? (S - 2) : 0;
    int end = (L + repeats < T) ? S : S-1;

    // Setup shared memory buffers
    #pragma unroll
    for (int idx = tid; idx < NV; idx += NT) {
        // load label in shared mem
        label[idx] = (idx < S) ? label_global[idx] : INT_MAX;
    }

    __syncthreads();

    // int flags;
    int uniquelabels;
    int seg_start[VT];
    int seg_end[VT];

    // Sort labels and record indices from which to gather from
    {
        int key[VT];
        int gather_val[VT];

        // (NT, VT), tid in NT
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
            const int idx = tid * VT + i;
            gather_val[i] = idx;
            key[i] = label[idx];
        }

        __syncthreads();

        // Caller provides the keys in shared memory. This functions sorts the first
        // count elements.
        CTAMergesort<NT, VT, true, true, int, int, mgpu::less<int>>
            (key, gather_val, keys_shared, gather_indices, S, tid, mgpu::less<int>());

        __syncthreads();

        for (int i = 0; i < VT; ++i) {
            const int idx = tid * VT + i;
            gather_indices[idx] = gather_val[i];
        }

        __syncthreads();

        SegReduce::preprocessKeys(keys_shared, S, &uniquelabels, seg_start, seg_end,
                                  temp_buffer.result);
        __syncthreads();
    }

    // TODO: probably not necessary
    __syncthreads();

    // Load labels back
    #pragma unroll
    for (int idx = tid; idx < NV; idx += NT) {
        // init beta with -inf
        temp_buffer.beta[idx] = ctc_helper::neg_inf<ProbT>();
    }
    __syncthreads();

    // Initialize the two rightmost values in the last row (assuming L non-zero)
    for(int i = tid; i < (end-start); i += blockDim.x)
        // init beta bin [start, end)
        temp_buffer.beta[start + i] =
            log(probs[(T - 1) * (stride * out_dim) + prob_offset + label[start + i]]);

    __syncthreads();

    // Load output data in registers through the transpose trick - should really be a function
    #pragma unroll
    for (int idx = tid; idx < S; idx += NT) {
        // \alpha_{T-1}(s) * \beat_{T-1}(s)
        output[idx] = alpha[(T - 1) * S + idx] + temp_buffer.beta[idx];
    }

    __syncthreads();

    // Start at the second to last row and backward in time
    for(int t = T - 1; t >= 0; --t) {

        // Start offsets into the current and next row
        const int start_cur_row = t * S;

        // Starting offset of column that we read from the probs array
        const int start_prob_col = t * (stride * out_dim);

        if (t < T-1) {

            // Filling up one row at one time but going back in time from the last row
            // to the first. As in the forward pass, there is no loop dependence and we
            // do a variable length filter of maximum filter size of 3
            #pragma unroll
            for(int idx = tid, i = 0; idx < (S-1); idx += NT, i++) {
                ProbT next_sum = log_plus_f(temp_buffer.beta[idx], temp_buffer.beta[idx+1]);

                // Skip two if not on blank and not on repeat.
                if ((label[idx] != blank_label) &&
                    (idx != (S-2)) && (label[idx] != label[idx+2]))
                    next_sum = log_plus_f(next_sum, temp_buffer.beta[idx+2]);

                beta_val[i] = next_sum + log(probs[start_prob_col + prob_offset + label[idx]]);
            }

            __syncthreads();

            // Initialize values for the rightmost column since there is nothing to the right
            // Update input buffer for next iteration
            if ((tid == 0) && (end == S))
                temp_buffer.beta[(S-1)] = temp_buffer.beta[(S-1)] +
                                          log(probs[start_prob_col + prob_offset + blank_label]);

            #pragma unroll
            for(int idx = tid, i = 0; idx < (S-1); idx += NT, i++) {
               temp_buffer.beta[idx] = beta_val[i];
            }

            __syncthreads();

            // Beta Computation done - add to alpha and update the gradient. Reload
            // the gradient back for segmented reduce later on
            #pragma unroll
            for(int idx = tid; idx < S; idx += NT) {
               // \alpha_{t}(s) * \beat_{t}(s)
               output[idx] = alpha[start_cur_row + idx] + temp_buffer.beta[idx];
            }

            __syncthreads();

        } // end if t < T-1

        __syncthreads();

        // Compute segmented reduction of output by using label as key
        {
            // Somewhat faster key value reduce
            ProbT accum[VT];

            for (int idx = tid, j = 0; idx < uniquelabels; idx += blockDim.x, ++j) {

                accum[j] = ctc_helper::neg_inf<ProbT>();
                for (int i = seg_start[j]; i <= seg_end[j]; ++i) {
                    accum[j] = log_plus_f(accum[j], output[gather_indices[i]]);
                }
            }
            __syncthreads();

            // Write accumulated value into output since that is not used
            for (int idx = tid, j = 0; idx < uniquelabels; idx += blockDim.x, ++j) {
                output[idx] = accum[j];
            }
            __syncthreads();

            for (int idx = tid; idx < out_dim; idx += blockDim.x) {
                const int grads_offset = start_prob_col + prob_offset + idx;
                grads[grads_offset] = probs[grads_offset];
            }

            __syncthreads();

            for (int idx = tid; idx < uniquelabels; idx += blockDim.x) {
                const int grads_offset = start_prob_col + prob_offset + keys_shared[idx];

                ProbT grad = output[idx];

                if ((grad == 0.0) || (probs[grads_offset] == 0.0) ||
                    (grad == ctc_helper::neg_inf<ProbT>())) {
                } else {
                    grads[grads_offset] =
                        probs[grads_offset] - exp(grad - log(probs[grads_offset]) - log_partition);
                }
            }

            __syncthreads();
        }

        // Output backward log likelihood
        if ((t == 0) && (tid == 0)) {
            ProbT loglikelihood = ctc_helper::neg_inf<ProbT>();

            const int val = 2 * (L-1) + 1 - (((L + repeats) == T) ? 1 : 0);

            start = (-val * (L != 0) + start);
            end = (-val * (L != 0) + end);

            // Sum and return the leftmost one/two value(s) in first row
            for(int i = start; i < end; ++i)
                loglikelihood = log_plus_f(loglikelihood, temp_buffer.beta[i]);

            nll_backward[blockIdx.x] = -loglikelihood;
        }

        // For some reason this is important
        __syncthreads();

    } // end for t

}

// probs minus max
template <typename ProbT, int VT = 1, typename Op>
__global__ void prepare_stable_SM_kernel_probs_minus_max(Op f, ProbT* probs,
                                         const ProbT* const col_max,
                                         int alphabet_size,
                                         int count) {
    // probs, (T,B,V)
    // col_max, (T*B,)
    // alphabet_size, V
    // count, T*B*V
    // block_dim, NT=128
    // grid_dim, count/128
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#pragma unroll
    for(int i = 0; i < VT; i++) {
        if (idx < count) {
            const int column_idx = idx / alphabet_size;
            probs[idx] = f(probs[idx] - col_max[column_idx]);
        }
        idx += stride;
    }
}

// exp probs and  div denom
template <typename ProbT, int VT = 1, typename Op>
__global__ void compute_probs_kernel_exp_probs_div_denom(Op f, ProbT* probs,
                                     const ProbT* const denom,
                                     int alphabet_size,
                                     int count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#pragma unroll
    for(int i = 0; i < VT; i++) {
        if (idx < count) {
            const int column_idx = idx / alphabet_size;
            probs[idx] = f(probs[idx]) / denom[column_idx];
        }
        idx += stride;
    }
}

// max(probs, min_T)
template <typename ProbT, int VT = 1>
__global__ void clip_min_probs_kernel(ProbT* probs, int count) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    ProbT min_T = numeric_limits<ProbT>::min();
#pragma unroll
    for(int i = 0; i < VT; i++) {
        if (idx < count) {
            if (min_T > probs[idx]) {
                probs[idx] = min_T;
            }
        }
        idx += stride;
    }
}

