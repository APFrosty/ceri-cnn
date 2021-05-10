#include "Main.h"
#include "Helper.h"
#include "Testing.h"

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

void Main::run() {
    Testing::testingSequence();
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