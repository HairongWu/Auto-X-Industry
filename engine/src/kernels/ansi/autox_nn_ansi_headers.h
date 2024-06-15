#pragma once

#include <stdint.h>

void autox_rmsnorm_ansi(float* o, float* x, float* weight, uint32_t size);
void autox_matmul_ansi(float* xout, float* x, float* w, uint32_t n, uint32_t d);
void autox_softmax_ansi(float* x, uint32_t size);
int autox_argmax_ansi(float* probabilities, uint32_t n);