#include "autox_nn_ansi.h"
#include <float.h>

void autox_accum_ansi(float *a, float *b, int size) {
    for (int i=0;i < size;i++) {
        a[i] += b[i];
    }
}