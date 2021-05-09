#include <Headers.h>
#include "Filter.h"

class Layer {
public:
    Layer(std::vector<Filter> filters, int stride);
    int8_t* apply(int8_t* input, int size, int depth);
    int getOutputDepth();
    int getOutputSize(int inputSize);

private:
    std::vector<Filter> filters;
    int stride;
};