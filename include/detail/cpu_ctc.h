#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#if !defined(CTC_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "ctc_helper.h"


template<typename ProbT>
class CpuCTC {
public:
    // Noncopyable
    CpuCTC(int alphabet_size /*vocab size, V*/, int minibatch /*batchsize, B*/, 
           void* workspace /*memory block*/, int num_threads,
           int blank_label /*blank id, default 0*/) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            num_threads_(num_threads), workspace_(workspace),
            blank_label_(blank_label) {
#if defined(CTC_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    // non-copy and non-assignment
    CpuCTC(const CpuCTC&) = delete;
    CpuCTC& operator=(const CpuCTC&) = delete;

    // compute loss and grad 
    ctcStatus_t cost_and_grad(const ProbT* const activations /*logits, (T, B, V)*/,
                              ProbT *grads, /*out*/
                              ProbT* costs, /*out*/
                              const int* const flat_labels /*lables in flat form, L1+...+LB=sum(label_lengths)*/,
                              const int* const label_lengths /*label len, (B,)*/,
                              const int* const input_lengths /*logits len, (B,)*/);

    // compute \alpha score
    ctcStatus_t score_forward(const ProbT* const activations /*logits, (T, B, V)*/,
                              ProbT* costs, /*out*/
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths /*logits len, (B,)*/);

private:

    class CpuCTC_metadata {
        private:
            int setup_labels(const int* const labels, int blank_label, int L, int S);

        public:
            // mb = minibatch = batch_size
            // L = |l|
            // S = 2|l| + 1
            // T = times
            // alphabet_size = V
            // blank_label = blank_id, default = 0
            // labels = l
            CpuCTC_metadata(int L, int S, int T, int mb, int alphabet_size,
                            void* workspace, size_t bytes_used, int blank_label,
                            const int* const labels);

            ProbT* alphas; //forward probs, (T,S)
            ProbT* betas; // bacwrad probs, (S,)
            int* labels_w_blanks; // 2L + 1 = S, (S,)
            int* e_inc;  // e(backward) max increase step in S space, (S,)
            int* s_inc;  // s(forward) max increase step in S space, (S,)
            ProbT* output; // (V,)
            int repeats; // number of continue same token in label l
    }; // CpuCTC_metadata

    int alphabet_size_; // Number of characters plus blank, vocab_size + 1 = V
    int minibatch_; // B
    int num_threads_; 
    int blank_label_; // blank_id
    void* workspace_;

    // activations to probs, (T, B, V)
    void softmax(const ProbT* const activations, ProbT* probs /*out*/,
                 const int* const input_lengths);

    std::tuple<ProbT, bool>
            cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                 const int* const labels, int T, int L,
                                 int mb, size_t bytes_used);

    ProbT compute_alphas(const ProbT* probs, int repeats, int S, int T,
                         const int* const e_inc,
                         const int* const s_inc,
                         const int* const labels,
                         ProbT* alphas);

    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                 ProbT log_partition, int repeats,
                                 int S, int T, const int* const e_inc,
                                 const int* const s_inc,
                                 const int* const labels,
                                 ProbT* alphas,
                                 ProbT* betas,
                                 ProbT* output);
};

template<typename ProbT>
CpuCTC<ProbT>::CpuCTC_metadata::CpuCTC_metadata(int L, int S, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                int blank_label,
                                                const int* const labels) {
    // bytes_used = offset in workspace

    // 1. alphas, (T, S), -inf, prob in log domain
    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S * T;
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());

    // 2. betas, (S), -inf, prob in log domain
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S;
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());

    // 3. labels_w_blanks, (S,)
    labels_w_blanks = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;

    // 4. e max increase index, (S,)
    e_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;

    // 5. s max increase index, (S,)
    s_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;

    // 6. output, (V,)
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

    // 7. repeats = continuous repeat charactor num, label l expand to l'
    repeats = setup_labels(labels, blank_label, L, S);
}

template<typename ProbT>
int CpuCTC<ProbT>::CpuCTC_metadata::setup_labels(const int* const labels,
                                                 int blank_label, int L, int S) {
    // L, label, abbc
    // s_inc,    1211
    // e_inc,     2111     
    // S, label with blank, ab-b-c

    // repeats: continuous repeat charactor num, e.g. abbc=1, abbbc=2
    // asssume L + repeats = T
    // s_counter `L+repeats - remain time step` to go forward, Fig.3 top-right circule
    // e_counter time step have go, Fig.3 left-bottom circlue
    int e_counter = 0; 
    int s_counter = 0; // s index of s_inc

    
    s_inc[s_counter++] = 1; // s_inc[0] = 1, start from blank

    int repeats = 0; // continuous repeat charactor num

    for (int i = 1; i < L; ++i) {
        if (labels[i-1] == labels[i]) {
            // repeat label, max step is 1
            // need go into blank, then to next label.
            s_inc[s_counter++] = 1;
            s_inc[s_counter++] = 1;
            e_inc[e_counter++] = 1;
            e_inc[e_counter++] = 1;
            ++repeats;
        }
        else {
            // not repeat label, max step is 2
            // can go to blank, or next label
            s_inc[s_counter++] = 2;
            e_inc[e_counter++] = 2;
        }
    }
    e_inc[e_counter++] = 1; // end to blank

    // fill label with blank
    for (int i = 0; i < L; ++i) {
        labels_w_blanks[2 * i] = blank_label;
        labels_w_blanks[2 * i + 1] = labels[i];
    }
    labels_w_blanks[S - 1] = blank_label; // last is blank

    return repeats;
}

// compute softmax on logits
// stalbe softmax = {shifx = x - np.max(x); exps = np.exp(shiftx); retrun exps / np.sum(exps);}
template<typename ProbT>
void
CpuCTC<ProbT>::softmax(const ProbT* const activations /*(T, B, V)*/, ProbT* probs /*out*/,
                       const int* const input_lengths /*(B,)*/) {
    ProbT min_T = std::numeric_limits<ProbT>::min();

    // minibatch_= B, index mb, col index
    // input_lengths = (B,), index c, row index
    // alphabet_size_ = V, index r
    // activations = logits,  (T, B, D)
#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for(int c = 0; c < input_lengths[mb]; ++c) {
            // col offset = offset in B dim
            int col_offset = (c * minibatch_ + mb) * alphabet_size_;
            
            ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
            for(int r = 0; r < alphabet_size_; ++r)
                max_activation = std::max(max_activation, activations[col_offset + r]);

            ProbT denom = ProbT(0.);
            for(int r = 0; r < alphabet_size_; ++r) {
                probs[col_offset + r] = std::exp(activations[col_offset + r] - max_activation);
                denom += probs[col_offset + r];
            }

            for(int r = 0; r < alphabet_size_; ++r) {
                probs[col_offset + r] /= denom;
                if (probs[col_offset + r] < min_T) {
                    probs[col_offset + r] = min_T;
                }
            }
        }
    }
}

// compute loss and grad for one example
template<typename ProbT>
std::tuple<ProbT, bool>
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {
    // grad (T, V), probs (T, V)
    
    // bytes_used = bytes offset in workspace
    const int S = 2*L + 1; // Number of labels with blanks

    CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_, bytes_used, blank_label_, labels);

    // foward and backward loss diff less than 0.1
    bool over_threshold = false;

    if (L + ctcm.repeats > T) { 
        // L is the label len. 
        // count L with expand blank between repeat label
        // l=aab  \bar{l} =a-ab, repeats is the count of blank should be insert between repeat label
        // \bar(l) is the min count of expansion of l.
        return std::make_tuple(ProbT(0), over_threshold); // TODO, not right to return 0
    }

    // forward log likelihood, alphas
    ProbT llForward = compute_alphas(probs, ctcm.repeats, S, T, ctcm.e_inc,
                                     ctcm.s_inc, ctcm.labels_w_blanks,
                                     ctcm.alphas);

    // backward log likelihook, betas, output, grad
    ProbT llBackward = compute_betas_and_grad(grad, probs, llForward, ctcm.repeats,
                                              S, T, ctcm.e_inc, ctcm.s_inc,
                                              ctcm.labels_w_blanks,
                                              ctcm.alphas,
                                              ctcm.betas,
                                              ctcm.output);

    // foward ll and backward ll diff more than 0.1 will set over_threshold=True
    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
        over_threshold = true;
    }

    // return (nll, whether diff over 0.1)
    return std::make_tuple(-llForward, over_threshold);
}

// Computes forward probabilities for one example, impelement of Eq.6
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_alphas(const ProbT* probs, int repeats, int S, int T,
                                    const int* const e_inc,
                                    const int* const s_inc,
                                    const int* const label_w_blank,
                                    ProbT* alphas) {
    // probs real shape (T, B, V), 
    // probs (T, 1, V), point to t0 probs.
    // lables = label_w_blank, (S,)
    // e_inc/s_inc, (S,)
    // alphas (T, S)

    // S/2 = int((2L+1)/2 ) = L
    // repeats = blank should be insert between repeat label, e.g abb, ab-b-
    

    // See Fig. 3, start and end are index of S, from 0-S
    // start = 0, if L + repeats < T
    // start = 1, if L + repeats = T, special case, e.g l=aaa, T=6. only can start from l'_1
    // L + repeats > T is not valid,  and is filter by `cost_and_grad_kernel`

    // end = 2, if S > 1, i.e L \neq 0, end is sentinel, i.e i < end
    // end = 1, if S <=1, i.e L = 0
    int start =  (((S /2) + repeats - T) < 0) ? 0 : 1,
            end = S > 1 ? 2 : 1;

    // alpha, beta init with -inf in CpuCTC_metadata
    // init \alpha_1(1) and \alpha_1(2)
    for (int i = start; i < end; ++i) {
        alphas[i] = std::log(probs[label_w_blank[i]]);
    }

    // \alpha_2 - \alpha_{T-1}
    for(int t = 1; t < T; ++t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= 0)
        {   // remian time step less than L+repeats, start should start inc step, not horizontal jump, 
            // or will not be valid path to go throught all sym in L'
            // condtion in Fig.3, right-top circle
            start += s_inc[remain];
        }

        if(t <= (S / 2) + repeats) {
            // condtion Fig.3, left-bottom circle
            // if t <= L+repeats end should inc step, or will horizontal jump.
            // maybe first L+repeats will eat all sym of L'
            end += e_inc[t - 1];
        }
           
        // idx1 t, idx2 t-1 of alpha ;idx3 t of global probs
        int startloop = start;
        int idx1 = t * S, idx2 = (t - 1) * S, idx3 = t * (minibatch_ * alphabet_size_);

        if (start == 0) {
            // in Fig.3,  process s=0 line, blank horizontal jump
            alphas[idx1] = alphas[idx2] + std::log(probs[idx3 + blank_label_]);
            // forward compute of alpha recurse from s=1
            startloop += 1;
        }

        for(int i = startloop; i < end; ++i) {
            // normal condtion
            // prev_sum = alpha_{t-1}(i) + alpha_{t-1}(i-1)
            ProbT prev_sum = ctc_helper::log_plus<ProbT>()(alphas[idx2 + i], alphas[idx2 + (i-1)]);

            // jump two if not on blank and not on repeat.
            if (label_w_blank[i] != blank_label_ && i != 1 && label_w_blank[i] != label_w_blank[i-2]) {
                // in this for loop, lowest i is 1, which only can jump one step from pre state
                prev_sum = ctc_helper::log_plus<ProbT>()(prev_sum, alphas[idx2 + (i-2)]);
            }
                
            // alpha_t(i) = prev_sum + probs_t(i)
            alphas[idx1 + i] = prev_sum + std::log(probs[idx3 + label_w_blank[i]]);
        }
    }

    ProbT loglikelihood = ctc_helper::neg_inf<ProbT>();
    // log-likelihood = \alpha_T(|l'|-1) + \alpha_T(|l'|) 
    for(int i = start; i < end; ++i) {
        loglikelihood = ctc_helper::log_plus<ProbT>()(loglikelihood, alphas[(T - 1) * S + i]);
    }

    return loglikelihood;
}


// Starting from T, we sweep backward over the alpha array computing one column
// of betas as we go.  At each position we can update product alpha * beta and then
// sum into the gradient associated with each label.
// NOTE computes gradient w.r.t UNNORMALIZED final layer activations(i.e. logits).
// Assumed passed in grads are already zeroed!
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                            ProbT log_partition, int repeats,
                                            int S, int T, const int* const e_inc,
                                            const int* const s_inc,
                                            const int* const labels_w_blanks,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    // grad, (T, B, V), zeroed, input point to (T, 1, V)
    // probs (T, 1, V)
    // log_partition = loglikelihood of Forward (1,)
    // labels = labels_w_blanks, (S,)
    // output, (V,), `sum_{s \in lab(z, k)} \alpha_t(s) \beta_t(s)`
    // beats, (S,)

    // backward start/end index, end is sentinel not include
    // start <= i < end
    int start = S > 1 ? (S - 2) : 0,
            end = (T > (S / 2) + repeats) ? S : S-1;

    // init output with -inf
    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    //set the starting values in the beta column at the very right edge
    // Compute Beta/output at T-1, 
    for (int i = start; i < end; ++i) {
        betas[i] = std::log(probs[(T - 1) * (minibatch_ * alphabet_size_) + labels_w_blanks[i]]);

        //compute alpha * beta in log space at this position in (S, T) space
        alphas[(T - 1) * S + i] += betas[i];

        //update the gradient associated with this label
        //essentially performing a reduce-by-key in a sequential manner
        output[labels_w_blanks[i]] =
                ctc_helper::log_plus<ProbT>()( output[labels_w_blanks[i]], alphas[(T - 1) * S + i] );
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
        int idx3 = (T - 1) * alphabet_size_ * minibatch_ + i;

        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            probs[idx3] == 0.0) {
            grad[idx3] = probs[idx3];
        } else {
            grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                std::log(probs[idx3]) - log_partition);
        }
    }

    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= -1)
            start -= s_inc[remain + 1];
        if(t < (S / 2) + repeats)
            end -= e_inc[t];

        int endloop = end == S ? end - 1 : end;
        int idx1 = t * S, idx3 = t * (minibatch_ * alphabet_size_);

        std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

        // process line <= S-2
        for(int i = start; i < endloop; ++i) {
            ProbT next_sum = ctc_helper::log_plus<ProbT>()(betas[i], betas[(i+1)]);
            // Skip two if not on blank and not on repeat.
            if (labels_w_blanks[i] != blank_label_ && i != (S-2) && labels_w_blanks[i] != labels_w_blanks[i+2]){
                next_sum = ctc_helper::log_plus<ProbT>()(next_sum, betas[(i+2)]);
            }
            betas[i] = next_sum + std::log(probs[idx3 + labels_w_blanks[i]]);

            //compute alpha * beta in log space
            alphas[i + idx1] += betas[i];

            //update the gradient associated with this label
            output[labels_w_blanks[i]] =
                    ctc_helper::log_plus<ProbT>()( output[labels_w_blanks[i]], alphas[idx1 + i] );
        }

        // process line S-1
        if (end == S) {
            betas[(S-1)] = betas[(S-1)] + std::log(probs[idx3 + blank_label_]);
            alphas[idx1 + (S-1)] += betas[(S-1)];

            output[labels_w_blanks[S-1]] =
                    ctc_helper::log_plus<ProbT>()( output[labels_w_blanks[S-1]], alphas[idx1 + (S-1)] );
        }

        //go over the unique labels and compute the final grad
        // wrt to each one at this time step
        for (int i = 0; i < alphabet_size_; ++i) {
            if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
                probs[idx3] == 0.0) {
                grad[idx3] = probs[idx3];
            } else {
                grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                    std::log(probs[idx3]) - log_partition);
            }
            ++idx3;
        }
    }

    // log-likelihood in backward.
    ProbT loglikelihood = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglikelihood = ctc_helper::log_plus<ProbT>()(loglikelihood, betas[i]);
    }

    return loglikelihood;
}

template<typename ProbT>
ctcStatus_t
CpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths) {
    // activations = logits, (T, B, V)
    // input_lengths, (B,)
    // label_lengths, (B,)
    // minibatch_ = B
    // grads, (T, B, V)
    // probs, (T, B, V)
    if (activations == nullptr ||
        grads == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    // first element in workspace is probs, (T, B, V)
    ProbT* probs = static_cast<ProbT *>(workspace_);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);
    size_t bytes_used = sizeof(ProbT) * maxT * minibatch_ * alphabet_size_;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);
    int maxS = 2 * maxL + 1;

    //output, (V,)
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas, (T, S)
    per_minibatch_bytes += sizeof(float) * maxS * maxT;

    //betas, (S,)
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc, (S,)
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    // batch softmax
    softmax(activations, probs, input_lengths);

    // compute loss and grad one by one
#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription

        bool mb_status;

        std::tie(costs[mb], mb_status) =
                cost_and_grad_kernel(grads + mb * alphabet_size_ /*(T, 1, V)*/,
                                     probs + mb * alphabet_size_ /*(T, 1, V)*/,
                                     flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                                     T, L, mb,
                                     bytes_used + mb * per_minibatch_bytes);
    }

    return CTC_STATUS_SUCCESS;
}

// compute forward log-likelihood
template<typename ProbT>
ctcStatus_t CpuCTC<ProbT>::score_forward(const ProbT* const activations,
                                         ProbT* costs,
                                         const int* const flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) {
    // activations = logits, (T, B, V)
    // cost, (B,)
    // input_lengths, (B,)
    // label_lengths, (B,)
    // minibatch_ = B
    // grads, (T, B, V)
    // probs, (T, B, V)
    if (activations == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    // probs is first element of workspace 
    ProbT* probs = static_cast<ProbT *>(workspace_);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);
    size_t bytes_used = sizeof(ProbT) * maxT * minibatch_ * alphabet_size_;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);
    int maxS = 2 * maxL + 1;

    //output, sum_{s \in lab(z,k)} \alpha_t(s) \beta_t(s)
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxT * maxS;

    //betas
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    // batch softmax
    softmax(activations, probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription
        const int S = 2*L + 1; // Number of labels with blanks

        CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_,
                             bytes_used + mb * per_minibatch_bytes, blank_label_,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));


        if (L + ctcm.repeats > T)
            // \bar{l} > T, invalid, set loss to zero
            costs[mb] = ProbT(0);
        else {
            // nll of forward as loss
            costs[mb] = -compute_alphas(probs + mb * alphabet_size_, ctcm.repeats, S, T,
                                        ctcm.e_inc, ctcm.s_inc, ctcm.labels_w_blanks,
                                        ctcm.alphas);
        }
    }

    return CTC_STATUS_SUCCESS;
}
