#include "Testing.h"
#include "CNN.h"

void Testing::testingSequence() {
    convolutionTest();
    poolingTest();
    //fullyConnectedTest();
    MNISTTest();
}

void Testing::convolutionTest() {

    std::cout << "Convolution Test: ";
    std::string log = "";
    bool allGood = true;

    float input[] = {
        0, 1, 1, 0, 2,
        2, 1, 0, 1, 2, 
        2, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        2, 2, 1, 1, 2,

        1, 1, 2, 0, 0,
        2, 1, 0, 0, 0,
        0, 2, 0, 2, 1,
        2, 1, 2, 1, 0,
        0, 1, 2, 1, 0,

        0, 2, 0, 2, 0,
        0, 0, 1, 2, 2,
        0, 2, 1, 0, 0,
        2, 0, 0, 2, 0,
        0, 1, 2, 1, 1
    };

    float filters[] = {
        1, 1, -1, -1, -1, 0, -1, 0, 0,
        0, 0, -1, 1, 0, 0, -1, -1, 1,
        1, -1, 0, -1, 1, 1, 0, -1, 1,

        -1, 0, -1, 0, 1, 0, -1, 1, -1,
        1, 0, -1, 1, 0, 0, -1, 0, -1,
        -1, -1, 0, 1, 1, 1, 1, -1, 0
    };


    float biases[] = {1, 0};

    int STRIDE = 2;
    int PADDING = 1;

    float* output = CNN::convolution(input, 5, 3, filters, biases, 3, 2, STRIDE, PADDING);

    float expected[] = {
        2, -1, -6, -2, 1, 4, -3, 0, 3,
        2, 2, 5, -1, 1, -2, 0, 5, 3
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

    std::cout << "Max Pooling Test: ";
    std::string log = "";
    bool allGood = true;

    float input[] = {
        1, 1, 2, 4,
        5, 6, 7, 8,
        3, 2, 1, 0,
        1, 2, 3, 4,

        3, 8, 2, 1,
        9, 4, 1, 13,
        52, 12, 7, 1,
        7, 5, 56, 3
    };

    int STRIDE = 2;
    int PADDING = 0;
    int POOLING_SIZE = 2;

    float* output = CNN::maxPooling(input, 4, 2, STRIDE, PADDING, POOLING_SIZE);

    float expected[] = {
        6, 8, 3, 4,
        9, 13, 52, 56
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

void Testing::fullyConnectedTest() {
    /*
    std::cout << "Fully Connected Test: ";
    std::string log = "";
    bool allGood = true;

    float input[] = {
        3, -4, -7, 1, 21, 14
    };

    float weights[] = {
        4, 6, 9, 10, 12, 18, -2, 14
    };

    float expected[] = {
        112, 168, 252, 24, 80, 248, 0xC8, 136
    };

    float* output = CNN::fullyConnected(input, 6, weights, 8);

    for(int i = 0; i < 8; ++i) {
        if(memcmp(&output[i], &expected[i], sizeof(float)) != 0) {
            log += "ERROR: expected[" + std::to_string(i)  +"] != output[" + std::to_string(i) + "] ----> " + std::to_string((int8_t)expected[i]) + " != " + std::to_string(output[i]) + "\n";
            allGood = false;
        }
    }

    std::cout << (allGood ? "SUCCESS" : "FAILED") << std::endl;
    if(!allGood) {
        std::cout << log;
    }

    delete[] output;
    */
}

void Testing::MNISTTest() {

    std::cout << "MNIST Test: ";

    {
        int INPUT_SIZES[] = {28, 14};
        int INPUT_DEPTHS[] = {1, 32};

        int FILTER_SIZES[] = {3, 3};
        int FILTER_COUNTS[] = {32, 64};

        int STRIDES[] = {1, 1};
        int PADDINGS[] = {1, 1};

        for(int i = 0; i < 2; ++i) {
            float* input = new float[INPUT_SIZES[i]*INPUT_SIZES[i]*INPUT_DEPTHS[i]];
            float* filters = new float[FILTER_SIZES[i]*FILTER_SIZES[i]*INPUT_DEPTHS[i]];
            float* biases = new float[FILTER_COUNTS[i]];
            float* output = CNN::convolution(input, INPUT_SIZES[i], INPUT_DEPTHS[i], filters, biases, FILTER_SIZES[i], FILTER_COUNTS[i], STRIDES[i], PADDINGS[i]);
            delete[] output;
        }
    }

    {
        int INPUT_SIZES[] = {28, 14};
        int INPUT_DEPTHS[] = {32, 64};

        int POOLING_SIZES[] = {2, 2};

        int STRIDES[] = {1, 1};
        int PADDINGS[] = {1, 1};

        for(int i = 0; i < 2; ++i) {
            float* input = new float[INPUT_SIZES[i]*INPUT_SIZES[i]*INPUT_DEPTHS[i]];
            float* filters = new float[POOLING_SIZES[i]*POOLING_SIZES[i]*INPUT_DEPTHS[i]];
            float* output = CNN::maxPooling(input, INPUT_SIZES[i], INPUT_DEPTHS[i], STRIDES[i], PADDINGS[i], POOLING_SIZES[i]);
            delete[] output;
        }
    }

    {
        int INPUT_LENGHTS[] = {3136, 128};
        int WEIGHT_COUNTS[] = {128, 10};

        for(int i = 0; i < 2; ++i) {
            float* input = new float[INPUT_LENGHTS[i]];
            float* weights = new float[WEIGHT_COUNTS[i]];
            //float* output = CNN::fullyConnected(input, INPUT_LENGHTS[i], weights, WEIGHT_COUNTS[i]);
        }
    }

    std::cout << "ENDED" << std::endl;
}