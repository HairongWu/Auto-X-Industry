#pragma once

#include <stdint.h>
#include <stdlib.h>

void autox_rmsnorm_ansi(float* o, float* x, float* weight, uint32_t size);
void autox_matmul_ansi(const float* X, const float* Y, float* Out, uint32_t *x_dims, uint16_t x_dims_size,
	uint32_t *y_dims, uint16_t y_dims_size, uint32_t *o_dims, uint16_t o_dims_size,
	int8_t x_transpose, int8_t y_transpose, float alpha);
void autox_softmax_ansi(float* x, uint32_t height, uint32_t width);
void autox_argmax_ansi(const float *X, uint32_t *Out, uint32_t size);
void autox_swiglu_ansi(float* hb, float* hb2, uint32_t hidden_dim);

void autox_accum_ansi(float *a, float *b, int size);
void autox_rope_rotation_ansi(int pos, float *sq, float *sk, int kv_dim, int head_size);