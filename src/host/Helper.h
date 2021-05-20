#include "Headers.h"

namespace Helper {
    void assertResult(cl_int result, std::string file, int line);
    float* imageToInput(std::string filename);
    void saveAsImage(std::string filename, float* data, int size, int depth);
    int8_t* imageToData(std::string filename);
    void saveLayer(std::string filename, int8_t* data, int size, int depth);
    int getImageSize(std::string filename);
    cl_program createProgram(std::string name);
    size_t getFileSize(std::string filename);
    std::string getFileContent(std::string filename);
    std::string errorCodeToString(cl_int errorCode);
}