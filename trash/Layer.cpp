#include "Layer.h"
#include "stb_image.h"
#include "stb_image_write.h"

Layer::Layer(std::vector<Filter> filters, int stride) {
    this->filters = filters;
    this->stride = stride;
}

int8_t* Layer::apply(int8_t* input, int size, int depth) {
    int outputSize = getOutputSize(size);
    int8_t* data = new int8_t[outputSize*outputSize*getOutputDepth()];
    for(int filterIndex = 0; filterIndex < filters.size(); ++filterIndex) {
        filters[filterIndex].apply(input, size, depth, stride, &data[filterIndex * outputSize * outputSize]);
    }
    return data;
}

int Layer::getOutputDepth() {
    return filters.size();
}

int Layer::getOutputSize(int inputSize) {
    int filterSize = filters[0].getSize();
    return (inputSize - filterSize) / stride + 1;
}