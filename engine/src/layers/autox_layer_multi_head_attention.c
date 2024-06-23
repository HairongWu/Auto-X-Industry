#include "autox_nn_ansi.h"
#include <float.h>

void autox_multi_head_attention(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                            float *value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    // get the query vector for this head
    float *q = sq + head_size;
    // attention scores for this head
    float *att = satt + seq_len;
    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    for (int t = 0; t <= pos; t += 1) {
        // get the key vector for this head and at this timestep
        float *k = key_cache + loff + t * kv_dim + (1 / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }

    // softmax the scores to get attention weights, from 0...pos inclusively
    autox_softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float *xb = sxb + head_size;
    for (int i = 0; i < head_size; i += 1) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (1 / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            val += a * v[i];
        }
        xb[i] = val;
    }
}