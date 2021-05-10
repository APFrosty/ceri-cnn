#include "Headers.h"

namespace CNN {
    int8_t* applyPadding(int8_t* input, int inputSize, int inputDepth, int padding);
    int8_t* convolution(int8_t* input, int inputSize, int inputDepth, int8_t* filters, int8_t* biases, int filterSize, int filterCount, int stride, int padding);
    int8_t* maxPooling(int8_t* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize);
    int8_t* fullyConnected(int8_t* input, int inputLength, int8_t* weights, int weightCount);
}