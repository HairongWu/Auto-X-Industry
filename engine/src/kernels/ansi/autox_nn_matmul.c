#include "autox_nn_ansi.h"

void autox_matmul_ansi(float* xout, float* x, float* w, uint32_t n, uint32_t d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    uint32_t i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
