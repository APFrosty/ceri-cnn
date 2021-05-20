#include "CNN.h"
#include "Helper.h"
#include "Main.h"

float* CNN::Tools::applyPadding(float* input, int inputSize, int inputDepth, int padding) {
    int size = inputSize * padding + 2;
    float* padded = new float[size*size*inputDepth];
    memset(padded, 0.0f, size*size*inputDepth*sizeof(float));

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

int* CNN::Tools::Convolution::createBiasIndices(int outputSize, int outputDepth) {
    int* array = new int[outputSize * outputSize * outputDepth];
    for(int i = 0; i < outputSize * outputSize * outputDepth; ++i) {
        array[i] = i / (outputSize * outputSize);
    }
    return array;
}

int* CNN::Tools::Convolution::createXArray(int size, int depth) {
    int* array = new int[size * size * depth];
    for(int i = 0; i < size*size*depth; ++i) {
        array[i] = i % size;
    }
    return array;
}

int* CNN::Tools::Convolution::createYArray(int size, int depth) {
    int* array = new int[size * size * depth];
    for(int i = 0; i < size*size*depth; ++i) {
        array[i] = i % (size*size) / size;
    }
    return array;
}

int* CNN::Tools::Convolution::createFilterIndices(int outputSize, int outputDepth, int filterSize, int inputDepth) {
    int* array = new int[outputSize * outputSize * outputDepth];
    for(int i = 0; i < outputSize * outputSize * outputDepth; ++i) {
        array[i] = i / (outputSize*outputSize) * filterSize * filterSize * inputDepth;
    }
    return array;
}

int* CNN::Tools::Pooling::createDepthsArray(int size, int depth) {
    int* array = new int[size * size * depth];
    for(int i = 0; i < size*size*depth; ++i) {
        array[i] = i / (size*size);
    }
    return array;
}

int* CNN::Tools::Pooling::createXArray(int outputSize, int outputDepth, int stride) {
    int* array = new int[outputSize * outputSize * outputDepth];
    for(int i = 0; i < outputSize * outputSize * outputDepth; ++i) {
        int position = i % (outputSize * outputSize);
        array[i] = (position % outputSize) * stride;
    }
    return array;
}

int* CNN::Tools::Pooling::createYArray(int outputSize, int outputDepth, int stride) {
    int* array = new int[outputSize * outputSize * outputDepth];
    for(int i = 0; i < outputSize * outputSize * outputDepth; ++i) {
        int position = i % (outputSize * outputSize);
        array[i] = (position / outputSize) * stride;
    }
    return array;
}

float* CNN::convolution(float* input, int inputSize, int inputDepth, float* filters, float* biases, int filterSize, int filterCount, int stride, int padding) {

    if(padding != 0) {
        input = Tools::applyPadding(input, inputSize, inputDepth, padding);
        inputSize = inputSize + padding * 2;
    }
    
    int outputSize = (inputSize - filterSize) / stride + 1;
    int outputDepth = filterCount;

    float* output = new float[outputSize*outputSize*outputDepth];

    // OpenCL implementation
    cl_context context = Main::getContext();
    cl_command_queue commandQueue = Main::getCommandQueue();
    cl_kernel convolutionKernel = Main::getConvolutionKernel();
    cl_int result;
    cl_mem clOutput;
    {
        // Set input
        cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputSize*inputSize*inputDepth*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, inputSize*inputSize*inputDepth*sizeof(float), input, 0, NULL, NULL);
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
        cl_mem clFilters = clCreateBuffer(context, CL_MEM_READ_WRITE, filterSize*filterSize*filterCount*inputDepth*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clFilters, CL_TRUE, 0, filterSize*filterSize*filterCount*inputDepth*sizeof(float), filters, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 3, sizeof(cl_mem), (void*)& clFilters);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set biases
        cl_mem clBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, filterCount*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clBiases, CL_TRUE, 0, filterCount*sizeof(float), biases, 0, NULL, NULL);
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
        clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize*outputSize*outputDepth*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 9, sizeof(cl_mem), (void*)& clOutput);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set Xs
        int* xs = Tools::Convolution::createXArray(outputSize, outputDepth);
        cl_mem clXs = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * outputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clXs, CL_TRUE, 0, outputSize*outputSize*outputDepth*sizeof(int), xs, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 10, sizeof(cl_mem), (void*)& clXs);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set Ys
        int* ys = Tools::Convolution::createYArray(outputSize, outputDepth);
        cl_mem clYs = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * outputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clYs, CL_TRUE, 0, outputSize*outputSize*outputDepth*sizeof(int), ys, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 11, sizeof(cl_mem), (void*)& clYs);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set biases indices
        int* indices = Tools::Convolution::createBiasIndices(outputSize, outputDepth);
        cl_mem clIndices = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * outputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clIndices, CL_TRUE, 0, outputSize*outputSize*outputDepth*sizeof(int), indices, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(convolutionKernel, 12, sizeof(cl_mem), (void*)& clIndices);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set output size
        result = clSetKernelArg(convolutionKernel, 13, sizeof(int), &outputSize);
        Helper::assertResult(result, __FILE__, __LINE__);
    }

    // Execute
    size_t globalItemSize = outputSize*outputSize*outputDepth;
    //globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, convolutionKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, outputSize * outputSize * outputDepth * sizeof(float), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    if(padding != 0) {
        delete[] input;
    }

    return output;
}

float* CNN::maxPooling(float* input, int inputSize, int inputDepth, int stride, int padding, int poolingSize) {
    
    int outputSize = (inputSize - poolingSize) / stride + 1;
    float* output = new float[inputDepth * outputSize * outputSize];

    if(padding != 0) {
        input = Tools::applyPadding(input, inputSize, inputDepth, padding);
        inputSize = inputSize + padding * 2;
    }

    // OpenCL implementation
    cl_context context = Main::getContext();
    cl_command_queue commandQueue = Main::getCommandQueue();
    cl_kernel maxPoolingKernel = Main::getMaxPoolingKernel();
    cl_int result;
    cl_mem clOutput;
    {
        // Set input
        cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputSize*inputSize*inputDepth*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, inputSize*inputSize*inputDepth*sizeof(float), input, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(maxPoolingKernel, 0, sizeof(cl_mem), (void*)& clInput);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set input size
        result = clSetKernelArg(maxPoolingKernel, 1, sizeof(int), &inputSize);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set input depth
        result = clSetKernelArg(maxPoolingKernel, 2, sizeof(int), &inputDepth);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set stride
        result = clSetKernelArg(maxPoolingKernel, 3, sizeof(int), &stride);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set padding
        result = clSetKernelArg(maxPoolingKernel, 4, sizeof(int), &padding);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set pooling size
        result = clSetKernelArg(maxPoolingKernel, 5, sizeof(int), &poolingSize);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set output
        clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize*outputSize*inputDepth*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(maxPoolingKernel, 6, sizeof(cl_mem), (void*)& clOutput);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set Xs
        int* xs = Tools::Pooling::createXArray(outputSize, inputDepth, stride);
        cl_mem clXs = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * inputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clXs, CL_TRUE, 0, outputSize*outputSize*inputDepth*sizeof(int), xs, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(maxPoolingKernel, 7, sizeof(cl_mem), (void*)& clXs);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set Ys
        int* ys = Tools::Pooling::createYArray(outputSize, inputDepth, stride);
        cl_mem clYs = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * inputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clYs, CL_TRUE, 0, outputSize*outputSize*inputDepth*sizeof(int), ys, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(maxPoolingKernel, 8, sizeof(cl_mem), (void*)& clYs);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set depths
        int* depths = Tools::Pooling::createDepthsArray(outputSize, inputDepth);
        cl_mem clDepths = clCreateBuffer(context, CL_MEM_READ_WRITE, outputSize * outputSize * inputDepth * sizeof(int), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clDepths, CL_TRUE, 0, outputSize*outputSize*inputDepth*sizeof(int), depths, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(maxPoolingKernel, 9, sizeof(cl_mem), (void*)& clDepths);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set output size
        result = clSetKernelArg(maxPoolingKernel, 10, sizeof(int), &outputSize);
        Helper::assertResult(result, __FILE__, __LINE__);

        // Clean
        delete[] xs;
        delete[] ys;
        delete[] depths;
    }

    // Execute
    size_t globalItemSize = outputSize*outputSize*inputDepth;
    //globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, maxPoolingKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, outputSize * outputSize * inputDepth * sizeof(float), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    if(padding != 0) {
        delete[] input;
    }

    return output;
}

float* CNN::fullyConnected(float* input, int inputLength, float* weights, float* biases, int neuronCount) {
    float* output = new float[neuronCount];

    // OpenCL implementation
    cl_context context = Main::getContext();
    cl_command_queue commandQueue = Main::getCommandQueue();
    cl_kernel fullyConnectedKernel = Main::getFullyConnectedKernel();
    cl_int result;
    cl_mem clOutput;
    {
        // Set input
        cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputLength*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, inputLength*sizeof(float), input, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(fullyConnectedKernel, 0, sizeof(cl_mem), (void*)& clInput);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set input length
        result = clSetKernelArg(fullyConnectedKernel, 1, sizeof(int), &inputLength);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set weights
        cl_mem clWeights = clCreateBuffer(context, CL_MEM_READ_WRITE, inputLength*neuronCount*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clWeights, CL_TRUE, 0, inputLength*neuronCount*sizeof(float), weights, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(fullyConnectedKernel, 2, sizeof(cl_mem), (void*)& clWeights);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set biases
        cl_mem clBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, neuronCount*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clEnqueueWriteBuffer(commandQueue, clBiases, CL_TRUE, 0, neuronCount*sizeof(float), biases, 0, NULL, NULL);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(fullyConnectedKernel, 3, sizeof(cl_mem), (void*)& clBiases);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set neuron count
        result = clSetKernelArg(fullyConnectedKernel, 4, sizeof(int), &neuronCount);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set output
        clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, neuronCount*sizeof(float), NULL, &result);
        Helper::assertResult(result, __FILE__, __LINE__);
        result = clSetKernelArg(fullyConnectedKernel, 5, sizeof(cl_mem), (void*)& clOutput);
        Helper::assertResult(result, __FILE__, __LINE__);
        // Set second
        int second = neuronCount == 10 ? 1 : 0;
        result = clSetKernelArg(fullyConnectedKernel, 6, sizeof(int), &second);
        Helper::assertResult(result, __FILE__, __LINE__);
    }

    // Execute
    size_t globalItemSize = neuronCount;
    //globalItemSize = 1;
    size_t localItemSize = 1;
    result = clEnqueueNDRangeKernel(commandQueue, fullyConnectedKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    // Read output
    result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, neuronCount * sizeof(float), output, 0, NULL, NULL);
    Helper::assertResult(result, __FILE__, __LINE__);

    return output;
}

// float* CNN::fullyConnected2(float* input, int inputLength, float* weights, float* biases, int neuronCount) {
//     float* output = new float[neuronCount];

//     // OpenCL implementation
//     cl_context context = Main::getContext();
//     cl_command_queue commandQueue = Main::getCommandQueue();
//     cl_kernel fullyConnectedKernel = Main::getFullyConnectedKernel();
//     cl_int result;
//     cl_mem clOutput;
//     {
//         // Set input
//         cl_mem clInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputLength*sizeof(float), NULL, &result);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clEnqueueWriteBuffer(commandQueue, clInput, CL_TRUE, 0, inputLength*sizeof(float), input, 0, NULL, NULL);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clSetKernelArg(fullyConnectedKernel, 0, sizeof(cl_mem), (void*)& clInput);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         // Set input length
//         result = clSetKernelArg(fullyConnectedKernel, 1, sizeof(int), &neuronCount);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         // Set weights
//         cl_mem clWeights = clCreateBuffer(context, CL_MEM_READ_WRITE, inputLength*neuronCount*sizeof(float), NULL, &result);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clEnqueueWriteBuffer(commandQueue, clWeights, CL_TRUE, 0, inputLength*neuronCount*sizeof(float), weights, 0, NULL, NULL);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clSetKernelArg(fullyConnectedKernel, 2, sizeof(cl_mem), (void*)& clWeights);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         // Set biases
//         cl_mem clBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, neuronCount*sizeof(float), NULL, &result);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clEnqueueWriteBuffer(commandQueue, clBiases, CL_TRUE, 0, neuronCount*sizeof(float), biases, 0, NULL, NULL);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clSetKernelArg(fullyConnectedKernel, 3, sizeof(cl_mem), (void*)& clBiases);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         // Set neuron count
//         result = clSetKernelArg(fullyConnectedKernel, 4, sizeof(int), &neuronCount);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         // Set output
//         clOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, neuronCount*sizeof(float), NULL, &result);
//         Helper::assertResult(result, __FILE__, __LINE__);
//         result = clSetKernelArg(fullyConnectedKernel, 5, sizeof(cl_mem), (void*)& clOutput);
//         Helper::assertResult(result, __FILE__, __LINE__);
//     }

//     // Execute
//     size_t globalItemSize = 1;
//     size_t localItemSize = 1;
//     result = clEnqueueNDRangeKernel(commandQueue, fullyConnectedKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
//     Helper::assertResult(result, __FILE__, __LINE__);

//     // Read output
//     result = clEnqueueReadBuffer(commandQueue, clOutput, CL_TRUE, 0, neuronCount * sizeof(float), output, 0, NULL, NULL);
//     Helper::assertResult(result, __FILE__, __LINE__);

//     return output;
// }