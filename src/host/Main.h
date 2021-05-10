#include "Headers.h"

#ifndef MAIN
#define MAIN

int main(int argc, char const *argv[]);

class Main {
public:
    Main();

    void initOpenCL();
    void run();
    static cl_device_id getDevice();
    static cl_context getContext();
    static cl_command_queue getCommandQueue();
    static cl_kernel getConvolutionKernel();

private:
    static Main* singleton;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numDevices;
    cl_uint numPlatforms;
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel convolutionKernel;

};

#endif