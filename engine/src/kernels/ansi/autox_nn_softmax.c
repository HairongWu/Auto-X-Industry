#include "autox_nn_ansi.h"

void autox_softmax_ansi(float* x, uint32_t size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (uint32_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (uint32_t i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
