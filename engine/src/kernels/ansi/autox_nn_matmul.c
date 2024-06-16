#include "autox_nn_ansi.h"
#include <stdbool.h>

static uint32_t count(const uint32_t* data_, uint16_t start, uint16_t end)
{
	start = max(start, 0);
	if (end < start) {
		return 0;
	}
	uint32_t sum = 1;
	for (uint16_t i = start; i < end; ++i) {
		sum *= data_[i];
	}
	return sum;
}
#define INIT_PARAM                                                           \
  int m=0, n=0, k=0;                                                               \
  int lda=0, ldb=0, ldc=0;                                                         \
  if ((x_dims_size >= 2 && y_dims_size >= 2) &&                          \
      (x_dims_size != 2 || y_dims_size != 2)) {                          \
    if (!x_transpose) {                                                      \
      m = x_dims[x_dims_size - 2];                                         \
      k = x_dims[x_dims_size - 1];                                         \
      lda = k;                                                               \
    } else {                                                                 \
      m = x_dims[x_dims_size - 1];                                         \
      k = x_dims[x_dims_size - 2];                                         \
      lda = m;                                                               \
    }                                                                        \
    if (!y_transpose) {                                                      \
      n = y_dims[y_dims_size - 1];                                         \
      ldb = n;                                                               \
    } else {                                                                 \
      n = y_dims[y_dims_size - 2];                                         \
      ldb = k;                                                               \
    }                                                                        \
    ldc = n;                                                                 \
  } else if ((x_dims_size == 2 && y_dims_size == 2) ||                   \
             (x_dims_size == 2 && y_dims_size == 1)) {                   \
    if (!x_transpose) {                                                      \
      m = x_dims[0];                                                         \
      k = x_dims[1];                                                         \
      lda = k;                                                               \
    } else {                                                                 \
      m = x_dims[1];                                                         \
      k = x_dims[0];                                                         \
      lda = m;                                                               \
    }                                                                        \
    if (!y_transpose) {                                                      \
      if (y_dims_size > 1) {                                               \
        n = y_dims[1];                                                       \
      } else {                                                               \
        n = 1;                                                               \
      }                                                                      \
      ldb = n;                                                               \
    } else {                                                                 \
      if (y_dims_size > 1) {                                               \
        n = y_dims[0];                                                       \
      } else {                                                               \
        n = 1;                                                               \
      }                                                                      \
      ldb = k;                                                               \
    }                                                                        \
    ldc = n;                                                                 \
  } else if (x_dims_size >= 2 && y_dims_size == 1) {                     \
    n = 1;                                                                   \
    k = y_dims[0];                                                           \
    if (!x_transpose) {                                                      \
      m = count(x_dims, 0, x_dims_size - 1);                                \
    } else {                                                                 \
      m = count(x_dims, 1, x_dims_size - 1);                                \
    }                                                                        \
    lda = k;                                                                 \
    ldb = n;                                                                 \
    ldc = n;                                                                 \
  } else if (y_dims_size >= 2 && x_dims_size == 1) {                     \
    m = 1;                                                                   \
    k = x_dims[0];                                                           \
    if (!y_transpose) {                                                      \
      n = count(y_dims, 1, y_dims_size);                                    \
    } else {                                                                 \
      n = count(y_dims, 0, y_dims_size - 1);                                \
    }                                                                        \
    lda = n;                                                                 \
    ldb = k;                                                                 \
    ldc = n;                                                                 \
  } else if (x_dims_size == 1 && y_dims_size == 1) {                     \
    m = 1;                                                                   \
    n = 1;                                                                   \
    k = x_dims[0];                                                           \
    if (x_transpose == true && y_transpose == true) {                        \
      m = x_dims[0];                                                         \
      k = 1;                                                                 \
      n = y_dims[0];                                                         \
    }                                                                        \
    lda = k;                                                                 \
    ldb = n;                                                                 \
    ldc = n;                                                                 \
  }

static void gemm_nn(int M, int N, int K, float ALPHA,
	const float *A, int lda,
	const float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[i*lda + k];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}

static void gemm_nt(int M, int N, int K, float ALPHA,
	const float *A, int lda,
	const float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}

static void gemm_tn(int M, int N, int K, float ALPHA,
	const float *A, int lda,
	const float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[k*lda + i];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}

static void gemm_tt(int M, int N, int K, float ALPHA,
	const float *A, int lda,
	const float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}

static void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	const float *A, int lda,
	const float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	int i, j;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] *= BETA;
		}
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void autox_matmul_ansi(const float* X, const float* Y, float* Out, uint32_t *x_dims, uint16_t x_dims_size,
	uint32_t *y_dims, uint16_t y_dims_size, uint32_t *o_dims, uint16_t o_dims_size,
	int8_t x_transpose, int8_t y_transpose, float alpha) {
	INIT_PARAM;
	const float* x_data = X;
	const float* y_data = Y;
	float* o_data = Out;
	if ((x_dims_size >= 2 && y_dims_size >= 2) &&
		(x_dims_size != 2 || y_dims_size != 2)) {
		// x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
		// x: [B, M, K], y: [K, N], out: [B, M, N]
		// or
		// x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
		// x: [M, K], y: [B, K, N], out: [B, M, N]
		int x_inner = x_dims[x_dims_size - 2] * x_dims[x_dims_size - 1];
		int y_inner = y_dims[y_dims_size - 2] * y_dims[y_dims_size - 1];
		int out_inner = o_dims[o_dims_size - 2] * o_dims[o_dims_size - 1];

		if (x_dims_size > 2 && y_dims_size > 2) {
			for (size_t i = 0; i < count(x_dims, 0, x_dims_size - 2); ++i) {
				gemm_cpu(x_transpose,
					y_transpose,
					m,
					n,
					k,
					alpha,
					x_data + i * x_inner,
					lda,
					y_data + i * y_inner,
					ldb,
					0.f,
					o_data + i * out_inner,
					ldc);
			}
		}
		else if (x_dims_size > 2 && y_dims_size == 2) {
			for (size_t i = 0; i < count(x_dims, 0, x_dims_size - 2); ++i) {
				gemm_cpu(x_transpose,
					y_transpose,
					m,
					n,
					k,
					alpha,
					x_data + i * x_inner,
					lda,
					y_data,
					ldb,
					0.f,
					o_data + i * out_inner,
					ldc);
			}
		}
		else if (x_dims_size == 2 && y_dims_size > 2) {
			for (size_t i = 0; i < count(y_dims, 0, y_dims_size - 2); ++i) {
				gemm_cpu(x_transpose,
					y_transpose,
					m,
					n,
					k,
					alpha,
					x_data,
					lda,
					y_data + i * y_inner,
					ldb,
					0.f,
					o_data + i * out_inner,
					ldc);
			}
		}
	}
	else if (x_dims_size == 2 && y_dims_size == 2) {
		// x: [M, K], y: [K, N], out: [M, N]
		gemm_cpu(x_transpose,
			y_transpose,
			m,
			n,
			k,
			alpha,
			x_data,
			lda,
			y_data,
			ldb,
			0.f,
			o_data,
			ldc);
	}
	else if (x_dims_size >= 2 && y_dims_size == 1) {
		// x: [B, M, K], y: [K], out: [B, M]
		gemm_cpu(x_transpose,
			false,
			m,
			n,
			k,
			alpha,
			x_data,
			lda,
			y_data,
			ldb,
			0.f,
			o_data,
			ldc);
	}
	else if (y_dims_size >= 2 && x_dims_size == 1) {
		// y: [B, K, N], x: [K], out: [B, N]
		gemm_cpu(false,
			y_transpose,
			m,
			n,
			k,
			alpha,
			x_data,
			lda,
			y_data,
			ldb,
			0.f,
			o_data,
			ldc);
	}
	else if (x_dims_size == 1 && y_dims_size == 1) {
		// x: [K], y: [K], out: [1]
		if (x_transpose == false && y_transpose == false) {
			o_data[0] = 0.;
			for (size_t i = 0; i < x_dims[0]; ++i) {
				o_data[0] += x_data[i] * y_data[i] * alpha;
			}
		}
		else if (x_transpose == true && y_transpose == true) {
			gemm_cpu(false,
				false,
				m,
				n,
				k,
				alpha,
				x_data,
				lda,
				y_data,
				ldb,
				0.f,
				o_data,
				ldc);
		}
	}
}
