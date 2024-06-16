#include "autox_nn_ansi.h"

void autox_argmax_ansi(const float *X, uint32_t *Out, uint32_t size)
{
	float first = X[0];
	uint32_t second = 0;
	for (uint32_t i = 1; i < size; i++) {
		if (X[i] > first) {
			first = X[i];
			second = i;
		}
	}
	*Out = second;
}