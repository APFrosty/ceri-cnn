#include "Helper.h"
#include "Main.h"
#include "stb_image.h"
#include "stb_image_write.h"

void Helper::assertResult(cl_int result, std::string file, int line) {
    if(result != CL_SUCCESS) {
        std::string message = Helper::errorCodeToString(result);
        std::cout << "-------------------------------\nERROR: " << message << "\nFile: " << file << "\nLine: " << line << "\n-------------------------------" << std::endl;
        exit(0);
    };
}

int8_t* Helper::imageToData(std::string filename) {
    std::string path = "../images/" + filename;
    int width;
    int height;
    int nrChannels;
    stbi_uc* raw = stbi_load(path.c_str(), &width, &height, &nrChannels, 3);
    int depth = 3;

    int8_t* data = new int8_t[depth*width*height];

    for(int i = 0; i < width * height; ++i) {
        data[i] = raw[i*3];
        data[width*height + i] = raw[i*3 + 1];
        data[width*height*2 + i] = raw[i*3 + 2];
    }

    stbi_image_free(raw);
    return data;
}

void Helper::saveLayer(std::string filename, int8_t* data, int size, int depth) {
    int8_t* tmp = new int8_t[size*size*3];

    for(int i = 0; i < depth; ++i) {
        for(int j = 0; j < size*size; ++j) {
            tmp[j*3 + 0] = data[i * size * size + j];
            tmp[j*3 + 1] = data[i * size * size + j];
            tmp[j*3 + 2] = data[i * size * size + j];
        }

        std::string path = "../images/" + filename + "_" + std::to_string(i) + ".bmp";
        stbi_write_bmp(path.c_str(), size, size, 3, tmp);
    }
}

int Helper::getImageSize(std::string filename) {
    std::string path = "../images/" + filename;
    int width;
    int height;
    int nrChannels;
    stbi_uc* raw = stbi_load(path.c_str(), &width, &height, &nrChannels, 3);
    stbi_image_free(raw);
    return width;
}

cl_program Helper::createProgram(std::string name) {
    cl_int result;
    cl_program program;
    cl_device_id device = Main::getDevice();
    #ifdef FPGA
        std::string filename = name + ".aocx";
        std::string binaryStr = getFileContent(filename);
        const char* binary = binaryStr.c_str();
        size_t binarySize = binaryStr.length();
        cl_int binaryResult;
        program = clCreateProgramWithBinary(Main::getContext(), 1, &device, &binarySize, (const unsigned char **) &binary, &binaryResult, &result);
        assertResult(result, __FILE__, __LINE__);
        assertResult(binaryResult, __FILE__, __LINE__);
    #else
        std::string filename = "../src/device/" + name + ".cl";
        std::string sourceStr = Helper::getFileContent(filename);
        const char* source = sourceStr.c_str();
        const size_t size = sourceStr.length();
        program = clCreateProgramWithSource(Main::getContext(), 1, &source, &size, &result);
        assertResult(result, __FILE__, __LINE__);
    #endif

    result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(result != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        std::cout << "Build Program Error: " << log << std::endl;
    }
    Helper::assertResult(result, __FILE__, __LINE__);

    return program;
}

size_t Helper::getFileSize(std::string filename) {
    return getFileContent(filename).length();
}

std::string Helper::getFileContent(std::string filename) {
    using namespace std;
    ifstream file(filename.c_str());
    string content;
    content.assign( (istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()) );
    return content;
}

std::string Helper::errorCodeToString(cl_int errorCode) {
switch(errorCode){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
     case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    }
    return "Unknown OpenCL error";
}