#include "Main.h"
#include "Helper.h"

Main* Main::singleton = NULL;

int main(int argc, char const *argv[]) {
    Main main;
    main.initOpenCL();
    main.run();
    return 0;
}

Main::Main() {
    singleton = this;
}

int8_t* Main::convolution(int8_t* input, int inputSize, int inputDepth, int8_t* filters, int8_t* biases, int filterSize, int filterCount, int stride, int padding) {

    auto determineInputEntry = [](int outputIndex, int outputSize, int inputSize, int stride, int depth) {
        int x = outputIndex % (outputSize * outputSize) % outputSize;
        int y = outputIndex % (outputSize * outputSize) / outputSize;
        int offset = depth * inputSize * inputSize;
        int entry = offset + x * stride + y * stride * inputSize;
        return entry;
    };

    if(padding != 0) {
        int size = inputSize + padding * 2;
        int8_t* used = new int8_t[size*size*inputDepth];
        memset(used, 0, size*size*inputDepth);


        for(int i = 0; i < size*size*inputDepth; ++i) {
            int x = i % (size*size) % size;
            int y = i % (size*size) / size;
            int depth = i / (size*size);
            if(x < padding || x >= size - padding || y < padding || y >= size - padding) {
                
            } else {
                x -= padding;
                y -= padding;
                int index = depth * inputSize * inputSize + x + y * inputSize;
                used[i] = input[index];
            }
        }

        input = used;
        inputSize = size;
    }
    
    int outputSize = (inputSize - filterSize) / stride + 1;
    int outputDepth = filterCount;

    int8_t* output = new int8_t[outputSize*outputSize*outputDepth];

    // CPU implementation
    int i = 0;
    while(i < outputSize*outputSize*outputDepth) {
        output[i] = 0;

        int8_t* filter = &filters[i / (outputSize*outputSize) * filterSize * filterSize * inputDepth];
        int8_t bias = biases[i / (outputSize*outputSize)];

        for(int filterDepth = 0; filterDepth < inputDepth; ++filterDepth) {
            int inputEntry = determineInputEntry(i, outputSize, inputSize, stride, filterDepth);
            for(int index = 0; index < filterSize * filterSize; ++index) {
                output[i] += filter[index + filterDepth*filterSize*filterSize] * input[inputEntry + index % filterSize + index / filterSize * inputSize];
            }
        }
        output[i] += bias;
        ++i;
    }

    return output;
}

void Main::initOpenCL() {
    cl_int result = clGetPlatformIDs(1, &platform, &numPlatforms);
    Helper::assertResult(result, __FILE__, __LINE__);

    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);

    commandQueue = clCreateCommandQueue(context, device, 0, &result);
    Helper::assertResult(result, __FILE__, __LINE__);

    cl_program program = Helper::createProgram("kernels");

    //applyFilterKernel = clCreateKernel(program, "applyFilter", &result);
    //BHelper::assertResult(result, __FILE__, __LINE__);
}

void Main::run() {
    int8_t input[] = {
        2, 0, 0, 2, 1,
        1, 0, 1, 1, 1,
        1, 1, 0, 0, 2,
        0, 0, 0, 1, 0,
        1, 1, 1, 1, 2,

        2, 2, 1, 1, 0,
        0, 0, 0, 2, 1,
        2, 0, 1, 1, 1,
        2, 2, 0, 0, 0,
        1, 2, 1, 2, 1,

        0, 0, 2, 1, 1,
        0, 2, 2, 2, 2,
        0, 0, 2, 1, 1,
        0, 1, 2, 2, 2,
        0, 1, 1, 2, 0,
    };

    int8_t filters[] = {
        1, 1, 1, 0, 0, 0, -1, -1, 0,
        -1, -1, -1, -1, -1, 0, 1, 1, 0,
        0, -1, -1, 1, -1, 0, 1, -1, 0,

        1, -1, -1, 0, 1, -1, 1, 1, 1,
        1, 1, 0, -1, -1, 0, -1, 0, 0,
        -1, -1, -1, 0, 1, 0, -1, 0, 0,
    };

    int8_t biases[] = {1, 0};

    int STRIDE = 2;
    int PADDING = 1;

    int8_t* output = convolution(input, 5, 3, filters, biases, 3, 2, STRIDE, PADDING);

    for(int i = 0; i < 3*3*2; ++i) {
        std::cout << (int)output[i];
        if(i % 3 != 2) {
            std::cout << ", ";
        } else {
            std::cout << "\n";
        }
    }
}

cl_device_id Main::getDevice() {
    return singleton->device;
}

cl_context Main::getContext() {
    return singleton->context;
}

cl_command_queue Main::getCommandQueue() {
    return singleton->commandQueue;
}

cl_kernel Main::getApplyFilterKernel() {
    return singleton->applyFilterKernel;
}