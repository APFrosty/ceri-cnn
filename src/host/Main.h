#include "Headers.h"

int main(int argc, char const *argv[]);

class Main {
public:
    Main();

    int8_t* convolution(int8_t* input, int inputSize, int inputDepth, int8_t* filters, int8_t* biases, int filterSize, int filterCount, int stride, int padding);

    void initOpenCL();
    void run();
    static cl_device_id getDevice();
    static cl_context getContext();
    static cl_command_queue getCommandQueue();
    static cl_kernel getApplyFilterKernel();

private:
    static Main* singleton;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numDevices;
    cl_uint numPlatforms;
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel applyFilterKernel;

};