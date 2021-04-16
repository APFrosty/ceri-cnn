#pragma once

#include <Headers.h>

class Filter {
public:
    Filter(int8_t* weights, int size, int depth);
    void apply(int8_t* input, int size, int depth, int stride, int8_t* output);
    int getSize();

private:
    int8_t* weights;
    cl_mem clWeights;
    int size;
    int depth;
};