#include "Filter.h"
#include "Main.h"
#include "Helper.h"

Filter::Filter(int8_t* weights, int size, int depth) {
    this->weights = weights;
    this->size = size;
    this->depth = depth;

    // Create buffer on device
    cl_int result;
    clWeights = clCreateBuffer(Main::getContext(), CL_MEM_READ_WRITE, size*size*depth * sizeof(uint8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Add data to buffer
    result = clEnqueueWriteBuffer(Main::getCommandQueue(), clWeights, CL_TRUE, 0, size*size*depth * sizeof(uint8_t), this->weights, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
}

void Filter::apply(int8_t* input, int size, int depth, int stride, int8_t* output) {
    cl_context context = Main::getContext();
    cl_kernel kernel = Main::getApplyFilterKernel();
    cl_command_queue commandQueue = Main::getCommandQueue();

    size_t outputSize = (size - this->size) / stride + 1;

    // Set weights
    cl_int result = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& clWeights);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set filter size
    result = clSetKernelArg(kernel, 1, sizeof(int), (void*)& this->size);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set input
    cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, size*size*depth*sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, size*size*depth * sizeof(int8_t), input, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& clInput);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set input size
    result = clSetKernelArg(kernel, 3, sizeof(int), (void*)& size);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set depth
    result = clSetKernelArg(kernel, 4, sizeof(int), (void*)& depth);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set stride
    result = clSetKernelArg(kernel, 5, sizeof(int), (void*)& stride);
    Helper::assertResult(result, __FILE__, __LINE__);
    // Set output
    cl_mem clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * sizeof(int8_t), NULL, &result);
    Helper::assertResult(result, __FILE__, __LINE__);
    result = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)& clOutput);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Execute
    size_t globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, outputSize * outputSize * sizeof(int8_t), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);
    return;
}

int Filter::getSize() {
    return size;
}