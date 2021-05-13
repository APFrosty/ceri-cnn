int determineInputEntry(int outputIndex, int outputSize, int inputSize, int stride, int depth, __global int* restrict Xs, __global int* restrict Ys) {
    int x = Xs[outputIndex];
    int y = Ys[outputIndex];
    int offset = depth * inputSize * inputSize;
    int entry = offset + x * stride + y * stride * inputSize;
    return entry;
}

__kernel void convolution(__global const char* restrict input, int inputSize, int inputDepth, __global const char* restrict filters, __global const char* restrict biases, int filterSize, int filterCount, int stride, int padding, __global char* restrict output, __global int* restrict Xs, __global int* restrict Ys, __global int* restrict biasesIndices, int outputSize) {
    int i = get_global_id(0);

    output[i] = 0;

    __global const char* filter = &filters[i / (outputSize*outputSize) * filterSize * filterSize * inputDepth];
    char bias = biases[i / (outputSize*outputSize)];

    for(int filterDepth = 0; filterDepth < inputDepth; ++filterDepth) {
        int offset = filterDepth * inputSize * inputSize;
        int inputEntry = determineInputEntry(i, outputSize, inputSize, stride, filterDepth, Xs, Ys);
        for(int index = 0; index < filterSize * filterSize; ++index) {
            output[i] += filter[index + filterDepth*filterSize*filterSize] * input[inputEntry + index % filterSize + index / filterSize * inputSize];
        }
    }
    output[i] += bias;
}

__kernel void maxPooling(__global const char* restrict input, int inputSize, int inputDepth, int stride, int padding, int poolingSize, __global char* restrict output, __global int* restrict Xs, __global int* restrict Ys, __global int* restrict depths, int outputSize) {
    int i = get_global_id(0);

    int depth = depths[i];
    int x = Xs[i];
    int y = Ys[i];
    int entry = (depth * inputSize * inputSize) + y * inputSize + x;
    char maxValue = 0x00;
    for(int j = 0; j < poolingSize * poolingSize; ++j) {
        x = j % poolingSize;
        y = j / poolingSize;
        maxValue = max(maxValue, input[entry + y * inputSize + x]);
    }
    output[i] = maxValue;
}

__kernel void fullyConnected(__global char* restrict input, int inputLength, __global char* restrict weights, int weightCount, __global char* restrict output) {
    int i = get_global_id(0);
    output[i] = 0;
    for(int j = 0; j < inputLength; ++j) {
        output[i] += weights[i] * input[j];
    }
}