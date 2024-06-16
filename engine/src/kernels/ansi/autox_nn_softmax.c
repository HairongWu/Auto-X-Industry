#include "autox_nn_ansi.h"
#include <float.h>

void autox_softmax_ansi(float* x, uint32_t height, uint32_t width) {

	for (uint32_t h = 0; h < height; h++)
	{
		// get max value of input data
		float in_max = -FLT_MAX;

		for (uint32_t i = 0; i < width; ++i) {
			in_max = max(in_max, x[i + h * width]);
		}
		// y = exp(x - in_max)
		for (uint32_t i = 0; i < width; ++i) {
			x[i + h * width] = expf(x[i + h * width] - in_max);
		}
		// y = y / sum(y[i], y[i + stride], y[i + stride + stride] ...)
		float sum = 0.f;
		for (uint32_t j = 0; j < width; ++j) {
			sum += x[j + h * width];
		}
		for (uint32_t j = 0; j < width; ++j) {
			x[j + h * width] /= sum;
		}
	}
}