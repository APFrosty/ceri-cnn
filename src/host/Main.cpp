#include "Main.h"
#include "Helper.h"
#include "Testing.h"
#include "Data.h"
#include "CNN.h"

Main* Main::singleton = NULL;

int main(int argc, char const *argv[]) {
    Main main;
    main.initOpenCL();
    std::string filename;
    if(argc < 2) {
        filename = "zero.png";
    } else {
        filename = argv[1];
    }
    main.run(filename);
    return 0;
}

Main::Main() {
    singleton = this;
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

    convolutionKernel = clCreateKernel(program, "convolution", &result);
    Helper::assertResult(result, __FILE__, __LINE__);

    maxPoolingKernel = clCreateKernel(program, "maxPooling", &result);
    Helper::assertResult(result, __FILE__, __LINE__);

    fullyConnectedKernel = clCreateKernel(program, "fullyConnected", &result);
    Helper::assertResult(result, __FILE__, __LINE__);
}

void Main::run(std::string filename) {
    float* input = Helper::imageToInput(filename);

    // Conv1 Layer
    int filterCount = 32;
    int padding = 1;
    int filterSize = 3;
    int stride = 1;
    int inputSize = 28;
    int inputDepth = 1;
    float* conv1Output = CNN::convolution(input, inputSize, inputDepth, conv1Weights, conv1Bias, filterSize, filterCount, stride, padding);
    delete[] input;

    // Pool1 Layer
    inputSize = 28;
    inputDepth = 32;
    padding = 0;
    stride = 2;
    int poolSize = 2;
    float* pool1Output = CNN::maxPooling(conv1Output, inputSize, inputDepth, stride, padding, poolSize);
    delete[] conv1Output;

    // Conv2 Layer
    inputSize = 14;
    inputDepth = 32;
    padding = 1;
    stride = 1;
    filterSize = 3;
    filterCount = 64;
    float* conv2Output = CNN::convolution(pool1Output, inputSize, inputDepth, conv2Weights, conv2Bias, filterSize, filterCount, stride, padding);
    delete[] pool1Output;

    // Pool2 Layer
    inputSize = 14;
    inputDepth = 64;
    stride = 2;
    padding = 0;
    poolSize = 2;
    float* pool2Output = CNN::maxPooling(conv2Output, inputSize, inputDepth, stride, padding, poolSize);
    delete[] conv2Output;

    // FC1 Layer
    int inputLength = 3136;
    int neuronCount = 128;
    float* fc1Output = CNN::fullyConnected(pool2Output, inputLength, linear1Weights, linear1Bias, neuronCount);

    // FC2 Layer
    inputLength = 128;
    neuronCount = 10;
    float* fc2Output = CNN::fullyConnected(fc1Output, inputLength, linear2Weights, linear2Bias, neuronCount);
    delete[] fc1Output;

    std::multimap<float, int> results;
    for(int i = 0; i < neuronCount; ++i) {
        results.insert({fc2Output[i], i});
    }

    std::cout << "For file: " << filename << std::endl;
    for(auto& iterator : results) {
        std::cout << iterator.second << ": " << iterator.first << std::endl;
    }

    delete[] fc2Output;
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

cl_kernel Main::getConvolutionKernel() {
    return singleton->convolutionKernel;
}

cl_kernel Main::getMaxPoolingKernel() {
    return singleton->maxPoolingKernel;
}

cl_kernel Main::getFullyConnectedKernel() {
    return singleton->fullyConnectedKernel;
}