#include "Testing.h"
#include "CNN.h"

void Testing::testingSequence() {
    convolutionTest();
    poolingTest();
}

void Testing::convolutionTest() {

    std::cout << "Convolution Test: " << std::endl;
    std::string log = "";
    bool allGood = true;

    int8_t input[] = {
        1, 0, 1, 1, 2,
        2, 2, 1, 1, 0,
        2, 0, 0, 0, 0, 
        1, 2, 2, 1, 1, 
        1, 2, 2, 0, 2,

        1, 0, 1, 2, 1, 
        0, 2, 2, 1, 2, 
        1, 1, 1, 2, 2, 
        2, 2, 0, 2, 1, 
        2, 1, 1, 0, 1,

        1, 2, 2, 2, 2,
        2, 2, 1, 2, 0,
        1, 0, 2, 2, 1,
        0, 0, 1, 1, 0,
        1, 1, 0, 2, 2,
    };

    int8_t filters[] = {
        -1, -1, -1, 1, -1, -1, -1, -1, -1,
        -1, -1, 1, -1, -1, -1, -1, 1, 0,
        1, 0, 1, 1, 0, 0, -1, -1, 0,

        -1, 0, 0, 0, 0, 1, 1, -1, -1,
        1, -1, -1, -1, 0, 1, -1, 1, 0,
        0, 1, 1, 0, 1, 0, -1, -1, 0,
    };


    int8_t biases[] = {1, 0};

    int STRIDE = 2;
    int PADDING = 1;

    int8_t* output = CNN::convolution(input, 5, 3, filters, biases, 3, 2, STRIDE, PADDING);

    int8_t expected[] = {
        -7, -9, -3, -4, -14, -7, -8, -4, -4,
        -5, 2, 0, 3, -1, -5, 0, -1, 2,
    };

    for(int i = 0; i < 18; ++i) {
        if(expected[i] != output[i]) {
            log += "ERROR: expected[" + std::to_string(i)  +"] != output[" + std::to_string(i) + "] ----> " + std::to_string(expected[i]) + " != " + std::to_string(output[i]) + "\n";
            allGood = false;
        }
    }

    std::cout << (allGood ? "SUCCESS" : "FAILED") << std::endl;
    if(!allGood) {
        std::cout << log;
    }

    delete[] output;
}

void Testing::poolingTest() {

    std::cout << "Max Pooling Test: " << std::endl;
    std::string log = "";
    bool allGood = true;

    int8_t input[] = {
        1, 1, 2, 4,
        5, 6, 7, 8,
        3, 2, 1, 0,
        1, 2, 3, 4
    };

    int STRIDE = 2;
    int PADDING = 0;
    int POOLING_SIZE = 2;

    int8_t* output = CNN::maxPooling(input, 4, 1, STRIDE, PADDING, POOLING_SIZE);

    int8_t expected[] = {
        6, 8, 3, 4
    };

    for(int i = 0; i < 4; ++i) {
        if(expected[i] != output[i]) {
            log += "ERROR: expected[" + std::to_string(i)  +"] != output[" + std::to_string(i) + "] ----> " + std::to_string(expected[i]) + " != " + std::to_string(output[i]) + "\n";
            allGood = false;
        }
    }

    std::cout << (allGood ? "SUCCESS" : "FAILED") << std::endl;
    if(!allGood) {
        std::cout << log;
    }

    delete[] output;
}