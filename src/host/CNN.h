#include "Headers.h"

namespace CNN {

    namespace Tools {
        int8_t* applyPadding(int8_t* input, int inputSize, int inputDepth, int padding);

        namespace Convolution {
            int* createBiasIndices(int outputSize, int outputDepth);
            int* createXArray(int size, int depth);
            int* createYArray(int size, int depth);
            int* createFilterIndices(int outputSize, int outputDepth, int filterSize, int inputDepth);
        }

        namespace Pooling {
            int* createDepthsArray(int size, int depth);
            int* createXArray(int outputSize, int outputDepth, int stride);
            int* createYArray(int outputSize, int outputDepth, int stride);
        }
    }

    int8_t* convolution(int8_t* input, int inputSize, int inputDepth, int8_t* filters, int8_t* biases, int filterSize, int filterCount, int stride, int padding);
    int8_t* maxPooling(int8_t* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize);
    int8_t* fullyConnected(int8_t* input, int inputLength, int8_t* weights, int weightCount);
    float* softmax(float* input, int inputLength);
}