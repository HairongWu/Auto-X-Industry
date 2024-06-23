#include "autox_nn_ansi.h"
#include <float.h>

void autox_swiglu_ansi(float* hb, float* hb2, uint32_t hidden_dim) {
	// SwiGLU non-linearity
	for (int i = 0; i < hidden_dim; i++) {
		float val = hb[i];
		// silu(x)=x*��(x), where ��(x) is the logistic sigmoid
		val *= (1.0f / (1.0f + expf(-val)));
		// elementwise multiply with w3(x)
		val *= hb2[i];
		hb[i] = val;
	}
}
