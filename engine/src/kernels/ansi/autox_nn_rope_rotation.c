#include "autox_nn_ansi.h"
#include <float.h>

void autox_rope_rotation_ansi(int pos, float *sq, float *sk, int dim, int kv_dim, int head_size) {
	for (int i = 0; i < dim/2 ; i+=2)
	{
		int head_dim = i % head_size;
		float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
		float val = pos * freq;
		float fcr = cosf(val);
		float fci = sinf(val);
		int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
		for (int v = 0; v < rotn; v++) {
			float *vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
			float v0 = vec[i];
			float v1 = vec[i + 1];
			vec[i] = v0 * fcr - v1 * fci;
			vec[i + 1] = v0 * fci + v1 * fcr;
		}
	}
}