__kernel void vector_add(__global const int* restrict a, __global const int* restrict b, __global int* restrict c) {
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
}