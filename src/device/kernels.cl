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

/*char max(char a, char b) {
    if(a > b) {
        return a;
    }
    return b;
}*/

__kernel void convolution(__global const char* input, int inputSize, int inputDepth, __global const char* filters, __global const char* restrict biases, int filterSize, int filterCount, int stride, int padding, __global char* output) {
    int i = get_global_id(0); 

    int outputSize = (inputSize - filterSize) / stride + 1;
    output[i] = 0;

    __global const char* filter = &filters[i / (outputSize*outputSize) * filterSize * filterSize * inputDepth];
    char bias = biases[i / (outputSize*outputSize)];

    for(int filterDepth = 0; filterDepth < inputDepth; ++filterDepth) {
        int inputEntry = determineInputEntry(i, outputSize, inputSize, stride, filterDepth);
        for(int index = 0; index < filterSize * filterSize; ++index) {
            output[i] += filter[index + filterDepth*filterSize*filterSize] * input[inputEntry + index % filterSize + index / filterSize * inputSize];
        }
    }
    output[i] += bias;
}

__kernel void maxPooling(__global const char* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize, __global char* output) {
    int i = get_global_id(0);

    int outputSize = (inputSize - poolingSize) / stride + 1;

    int depth = i / (outputSize * outputSize);
    int position = i % (outputSize * outputSize);
    int x = (position % outputSize) * stride;
    int y = (position / outputSize) * stride;
    int entry = (depth * inputSize * inputSize) + y * inputSize + x;
    char maxValue = 0x00;
    for(int j = 0; j < poolingSize * poolingSize; ++j) {
        x = j % poolingSize;
        y = j / poolingSize;
        maxValue = max(maxValue, input[entry + y * inputSize + x]);
    }
    output[i] = maxValue;
}

__kernel void fullyConnected(__global char* restrict input, int inputLength, __global char* restrict weights, int weightCount, __global char* output) {
    int i = get_global_id(0);
    output[i] = 0;
    for(int j = 0; j < inputLength; ++j) {
        output[i] += weights[i] * input[j];
    }
}