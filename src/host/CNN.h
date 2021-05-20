#include "Headers.h"

namespace CNN {

    namespace Tools {
        float* applyPadding(float* input, int inputSize, int inputDepth, int padding);

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

    float* convolution(float* input, int inputSize, int inputDepth, float* filters, float* biases, int filterSize, int filterCount, int stride, int padding);
    float* maxPooling(float* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize);
    float* fullyConnected(float* input, int inputLength, float* weights, float* biases, int neuronCount);
    //float* fullyConnected2(float* input, int inputLength, float* weights, float* biases, int neuronCount);
}