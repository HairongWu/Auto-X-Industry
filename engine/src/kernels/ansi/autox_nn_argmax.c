#include "autox_nn_ansi.h"

int autox_argmax_ansi(float* probabilities, uint32_t n) {
    // return the index that has the highest probability
    uint32_t max_i = 0;
    float max_p = probabilities[0];
    for (uint32_t i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}