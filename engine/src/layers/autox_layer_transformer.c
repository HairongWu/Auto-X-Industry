#include "../kernels/autox_nn.h"
#include "autox_layers.h"
#include <string.h>

float* autox_transformer(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

	uint32_t x_dims[1];
	uint32_t y_dims[2];
	uint32_t o_dims[1];
    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        uint32_t loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;	
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wq + l * dim*dim, s->q, x_dims, 1,y_dims, 2, o_dims, 1,false, true, 1.0);
		x_dims[0] = dim;
		y_dims[0] = kv_dim;
		y_dims[1] = dim;
		o_dims[0] = kv_dim;
		autox_matmul(s->xb, w->wk + l * dim*kv_dim, s->k, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
		autox_matmul(s->xb, w->wv + l * dim*kv_dim, s->v, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);


        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;

        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            autox_softmax(att, 1, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wo + l * dim*dim, s->xb2, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
		x_dims[0] = dim;
		y_dims[0] = hidden_dim;
		y_dims[1] = dim;
		o_dims[0] = hidden_dim;
		autox_matmul(s->xb, w->w1 + l * dim*hidden_dim, s->hb, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
		autox_matmul(s->xb, w->w3 + l * dim*hidden_dim, s->hb2, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);

        // SwiGLU non-linearity
		autox_swiglu(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
		x_dims[0] = hidden_dim;
		y_dims[0] = dim;
		y_dims[1] = hidden_dim;
		o_dims[0] = dim;
		autox_matmul(s->hb, w->w2 + l * dim*hidden_dim, s->xb, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    autox_rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
	x_dims[0] = p->dim;
	y_dims[0] = p->vocab_size;
	y_dims[1] = p->dim;
	o_dims[0] = p->vocab_size;
	autox_matmul(x, w->wcls, s->logits, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
    return s->logits;
}

float *autox_transformer3(Transformer *transformer, int token, int pos) {
    // a few convenience variables
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
	uint32_t x_dims[1];
	uint32_t y_dims[2];
	uint32_t o_dims[1];
    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wq + l * dim*dim, s->q, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
		x_dims[0] = dim;
		y_dims[0] = kv_dim;
		y_dims[1] = dim;
		o_dims[0] = kv_dim;
		autox_matmul(s->xb, w->wk + l * dim*kv_dim, s->k, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
		autox_matmul(s->xb, w->wv + l * dim*kv_dim, s->v, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        autox_rope_rotation(pos, s->q, s->k, dim, kv_dim, head_size);

        // multihead attention. iterate over all heads
        autox_multi_head_attention(p->n_heads,pos, p->seq_len, s->q, s->att, s->xb,
			s->key_cache, s->value_cache, kv_dim, kv_mul,
			head_size, loff);

        // final matmul to get the output of the attention
		x_dims[0] = dim;
		y_dims[0] = dim;
		y_dims[1] = dim;
		o_dims[0] = dim;
		autox_matmul(s->xb, w->wo + l * dim*dim, s->xb2, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);

        // residual connection back into x
        autox_accum(x, s->xb2, dim);

        // ffn rmsnorm
        autox_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
		x_dims[0] = dim;
		y_dims[0] = hidden_dim;
		y_dims[1] = dim;
		o_dims[0] = hidden_dim;
		autox_matmul(s->xb, w->w1 + l * dim*hidden_dim, s->hb, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
		autox_matmul(s->xb, w->w3 + l * dim*hidden_dim, s->hb2, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);

        // SwiGLU non-linearity
        autox_swiglu(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
		x_dims[0] = hidden_dim;
		y_dims[0] = dim;
		y_dims[1] = hidden_dim;
		o_dims[0] = dim;
		autox_matmul(s->hb, w->w2 + l * dim*hidden_dim, s->xb, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);

        // residual connection
        autox_accum(x, s->xb, dim);
    }

    // final rmsnorm
    autox_rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
	x_dims[0] = p->dim;
	y_dims[0] = p->vocab_size;
	y_dims[1] = p->dim;
	o_dims[0] = p->vocab_size;
	autox_matmul(x, w->wcls, s->logits, x_dims, 1, y_dims, 2, o_dims, 1, false, true, 1.0);
    return s->logits;
}
