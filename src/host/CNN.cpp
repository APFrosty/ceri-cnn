#include "CNN.h"
#include "Helper.h"
#include "Main.h"

int8_t* CNN::applyPadding(int8_t* input, int inputSize, int inputDepth, int padding) {
    int size = inputSize * padding + 2;
    int8_t* padded = new int8_t[size*size*inputDepth];
    memset(padded, 0, size*size*inputDepth);

    for(int i = 0; i < size*size*inputDepth; ++i) {
        int x = i % (size*size) % size;
        int y = i % (size*size) / size;
        int depth = i / (size*size);
        if(x < padding || x >= size - padding || y < padding || y >= size - padding) {
            
        } else {
            x -= padding;
            y -= padding;
            int index = depth * inputSize * inputSize + x + y * inputSize;
            padded[i] = input[index];
        }
    }

    return padded;
}

int8_t* CNN::convolution(int8_t* input, int inputSize, int inputDepth, int8_t* filters, int8_t* biases, int filterSize, int filterCount, int stride, int padding) {

    if(padding != 0) {
        input = applyPadding(input, inputSize, inputDepth, padding);
        inputSize = inputSize + padding * 2;
    }
    
    int outputSize = (inputSize - filterSize) / stride + 1;
    int outputDepth = filterCount;

    int8_t* output = new int8_t[outputSize*outputSize*outputDepth];

    // OpenCL implementation
    cl_context context = Main::getContext();
    cl_command_queue commandQueue = Main::getCommandQueue();
    cl_kernel convolutionKernel = Main::getConvolutionKernel();
    cl_int result;
    cl_mem clOutput;
    {
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
        clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize*outputSize*outputDepth*sizeof(int8_t), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 9, sizeof(cl_mem), (void*)& clOutput);
        Helper::assertResult(result, __FILE__, __LINE__);
    }

    // Execute
    size_t globalItemSize = outputSize*outputSize*outputDepth;
    //globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, convolutionKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, outputSize * outputSize * outputDepth * sizeof(int8_t), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    if(padding != 0) {
        delete[] input;
    }

    return output;
}

int8_t* CNN::maxPooling(int8_t* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize) {
    
    int outputSize = (inputSize - poolingSize) / stride + 1;
    int8_t* output = new int8_t[inputDepth * outputSize * outputSize];

    if(padding != 0) {
        input = applyPadding(input, inputSize, inputDepth, padding);
        inputSize = inputSize + padding * 2;
    }

    // CPU implementation
    for(int i = 0; i < outputSize * outputSize * inputDepth; ++i) {
        int depth = i / (outputSize * outputSize);
        int position = i % (outputSize * outputSize);
        int x = (position % outputSize) * stride;
        int y = (position / outputSize) * stride;
        int entry = (depth * inputSize * inputSize) + y * inputSize + x;
        int8_t max = 0;
        for(int j = 0; j < poolingSize * poolingSize; ++j) {
            x = j % poolingSize;
            y = j / poolingSize;
            max = std::max(max, input[entry + y * inputSize + x]);
        }
        output[i] = max;
    }

    if(padding != 0) {
        delete[] input;
    }

    return output;
}