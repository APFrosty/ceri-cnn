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
    /*int i = 0;
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
    }*/

    // OpenCL implementation
    cl_int result;
    // Set input
    cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputSize*inputSize*inputDepth*sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, inputSize*inputSize*inputDepth*sizeof(int8_t), input, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(convolutionKernel, 0, sizeof(cl_mem), (void*)& clInput);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set input size
    result = clSetKernelArg(convolutionKernel, 1, sizeof(int), &inputSize);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set input depth
    result = clSetKernelArg(convolutionKernel, 2, sizeof(int), &inputDepth);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set filters
    cl_mem clFilters = clCreateBuffer(context, CL_MEM_READ_WRITE, filterSize*filterSize*filterCount*inputDepth*sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clEnqueueWriteBuffer(commandQueue, clFilters, CL_TRUE, 0, filterSize*filterSize*filterCount*inputDepth*sizeof(int8_t), filters, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(convolutionKernel, 3, sizeof(cl_mem), (void*)& clFilters);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set biases
    cl_mem clBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, filterCount*sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clEnqueueWriteBuffer(commandQueue, clBiases, CL_TRUE, 0, filterCount*sizeof(int8_t), biases, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(convolutionKernel, 4, sizeof(cl_mem), (void*)& clBiases);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set filter size
    result = clSetKernelArg(convolutionKernel, 5, sizeof(int), &filterSize);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set filter count
    result = clSetKernelArg(convolutionKernel, 6, sizeof(int), &filterCount);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set stride
    result = clSetKernelArg(convolutionKernel, 7, sizeof(int), &stride);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set padding
    result = clSetKernelArg(convolutionKernel, 8, sizeof(int), &padding);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set output
    cl_mem clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize*outputSize*outputDepth*sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(convolutionKernel, 9, sizeof(cl_mem), (void*)& clOutput);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Execute
    size_t globalItemSize = outputSize*outputSize*outputDepth;
    //globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, convolutionKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, outputSize * outputSize * outputDepth * sizeof(int8_t), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

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

    convolutionKernel = clCreateKernel(program, "convolution", &result);
    Helper::assertResult(result, __FILE__, __LINE__);
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
    return singleton->convolutionKernel;
}