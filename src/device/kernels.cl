int determineInputEntry(int outputIndex, int outputSize, int inputSize, int stride, int depth, __global int* restrict Xs, __global int* restrict Ys) {
    int x = Xs[outputIndex];
    int y = Ys[outputIndex];
    int offset = depth * inputSize * inputSize;
    int entry = offset + x * stride + y * stride * inputSize;
    return entry;
}

__kernel void convolution(__global const float* restrict input, int inputSize, int inputDepth, __global const float* restrict filters, __global const float* restrict biases, int filterSize, int filterCount, int stride, int padding, __global float* restrict output, __global int* restrict Xs, __global int* restrict Ys, __global int* restrict biasesIndices, int outputSize) {
    int i = get_global_id(0);

    output[i] = biases[i / (outputSize*outputSize)];

    __global const float* filter = &filters[i / (outputSize*outputSize) * filterSize * filterSize * inputDepth];

    for(int filterDepth = 0; filterDepth < inputDepth; ++filterDepth) {
        int offset = filterDepth * inputSize * inputSize;
        int inputEntry = determineInputEntry(i, outputSize, inputSize, stride, filterDepth, Xs, Ys);
        for(int index = 0; index < filterSize * filterSize; ++index) {
            float left = filter[index + filterDepth*filterSize*filterSize];
            float right = input[inputEntry + index % filterSize + index / filterSize * inputSize];
            float result = left * right;
            output[i] += left * right;
        }
    }
    output[i] = max(0.0f, output[i]);
}

__kernel void maxPooling(__global const float* restrict input, int inputSize, int inputDepth, int stride, int padding, int poolingSize, __global float* restrict output, __global int* restrict Xs, __global int* restrict Ys, __global int* restrict depths, int outputSize) {
    int i = get_global_id(0);

    int depth = depths[i];
    int x = Xs[i];
    int y = Ys[i];
    int entry = (depth * inputSize * inputSize) + y * inputSize + x;
    float maxValue = 0.0f;
    for(int j = 0; j < poolingSize * poolingSize; ++j) {
        x = j % poolingSize;
        y = j / poolingSize;
        maxValue = max(maxValue, input[entry + y * inputSize + x]);
    }
    output[i] = maxValue;
}

__kernel void fullyConnected(__global float* restrict input, int inputLength, __global float* restrict weights, __global float* restrict biases, int neuronCount, __global float* restrict output, int second) {
    int i = get_global_id(0);
    
    output[i] = biases[i];
    for(int j = 0; j < inputLength; ++j) {
        float weight = weights[i * inputLength + j];
        output[i] += weights[i * inputLength + j] * input[j];
    }
    if(second == 0) {
        output[i] = max(0.0f, output[i]);
    }
}