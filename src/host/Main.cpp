#include "Main.h"
#include "Helper.h"
#include "Layer.h"
#include "Filter.h"

Main* Main::singleton = NULL;

int main(int argc, char const *argv[])
{
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

    applyFilterKernel = clCreateKernel(program, "applyFilter", &result);
    Helper::assertResult(result, __FILE__, __LINE__);
}

void Main::run() {
    std::string IMAGE_FILE = "cat.png";
    int IMAGE_SIZE = Helper::getImageSize(IMAGE_FILE);
    auto input = Helper::imageToData(IMAGE_FILE);
    int FILTER_SIZE = 1;
    int FILTER_COUNT = 96;

    int8_t* weights = new int8_t[FILTER_SIZE*FILTER_SIZE*3];
    for(int i = 0; i < FILTER_SIZE*FILTER_SIZE*3; ++i) {
        weights[i] = 1;
    }
    std::vector<Filter> filters;
    filters.resize(FILTER_COUNT, Filter(weights, FILTER_SIZE, 3));
    Layer layer = Layer(filters, 1);
    std::cout << "output depth: " << layer.getOutputDepth() << std::endl;
    std::cout << "output size: " << layer.getOutputSize(IMAGE_SIZE) << std::endl;

    auto output = layer.apply(input, IMAGE_SIZE, 3);
    Helper::saveLayer("output", output, layer.getOutputSize(IMAGE_SIZE), layer.getOutputDepth());
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