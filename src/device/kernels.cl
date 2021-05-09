/*__kernel void vector_add(__global const int* restrict a, __global const int* restrict b, __global int* restrict c) {
    int index = get_global_id(0);
    c[index] = a[index] + b[index];
}

__kernel void applyFilter(__global const char* restrict weights, int filterSize, __global const char* restrict input, int size, int depth, int stride, __global char* restrict output) {
    int outputIndex = 0;
    for(int x = 0; x < size; x += stride) {
        for(int y = 0; y < size; y += stride) {
            char sum = 0;
            for(int z = 0; z < depth; ++z) {
                int inputIndex = y * size + x + z * size * size;
                for(int i = 0; i < filterSize * filterSize; ++i) {
                    int filterIndex = z * filterSize * filterSize;
                    sum += weights[filterIndex + i] * input[inputIndex + i + size * (i / filterSize)];
                }
            };
            output[outputIndex] = sum;
            ++outputIndex;
        }
    }
}*/

int determineInputEntry(int outputIndex, int outputSize, int inputSize, int stride, int depth) {
    int x = outputIndex % (outputSize * outputSize) % outputSize;
    int y = outputIndex % (outputSize * outputSize) / outputSize;
    int offset = depth * inputSize * inputSize;
    int entry = offset + x * stride + y * stride * inputSize;
    return entry;
}

__kernel void convolution(__global const char* input, int inputSize, int inputDepth, __global const char* filters, __global const char* restrict biases, int filterSize, int filterCount, int stride, int padding, __global char* output) {
    int i = get_global_id(0); 
    //i = 9;

    int outputSize = (inputSize - filterSize) / stride + 1;
    output[i] = 0;

    __global const char* filter = &filters[i / (outputSize*outputSize) * filterSize * filterSize * inputDepth];
    char bias = biases[i / (outputSize*outputSize)];

    for(int filterDepth = 0; filterDepth < inputDepth; ++filterDepth) {
        int inputEntry = determineInputEntry(i, outputSize, inputSize, stride, filterDepth);
        for(int index = 0; index < filterSize * filterSize; ++index) {
            output[i] += filter[index + filterDepth*filterSize*filterSize] * input[inputEntry + index % filterSize + index / filterSize * inputSize];
            //printf("%d * %d -> %d\n", input[inputEntry + index % filterSize + index / filterSize * inputSize], filter[index + filterDepth*filterSize*filterSize], output[i]);
        }
    }
    output[i] += bias;

    //printf("%d: \t\t%d\n", i, (char)output[i]);
}