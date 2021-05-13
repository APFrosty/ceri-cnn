#include "Testing.h"
#include "CNN.h"

void Testing::testingSequence() {
    //convolutionTest();
    //poolingTest();
    //fullyConnectedTest();
    MNISTTest();
}

void Testing::convolutionTest() {

    std::cout << "Convolution Test: ";
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

    std::cout << "Max Pooling Test: ";
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

void Testing::fullyConnectedTest() {
    std::cout << "Fully Connected Test: ";
    std::string log = "";
    bool allGood = true;

    int8_t input[] = {
        3, -4, -7, 1, 21, 14
    };

    int8_t weights[] = {
        4, 6, 9, 10, 12, 18, -2, 14
    };

    uint8_t expected[] = {
        112, 168, 252, 24, 80, 248, 0xC8, 136
    };

    int8_t* output = CNN::fullyConnected(input, 6, weights, 8);

    for(int i = 0; i < 8; ++i) {
        if(memcmp(&output[i], &expected[i], sizeof(int8_t)) != 0) {
            log += "ERROR: expected[" + std::to_string(i)  +"] != output[" + std::to_string(i) + "] ----> " + std::to_string((int8_t)expected[i]) + " != " + std::to_string(output[i]) + "\n";
            allGood = false;
        }
    }

    std::cout << (allGood ? "SUCCESS" : "FAILED") << std::endl;
    if(!allGood) {
        std::cout << log;
    }

    delete[] output;
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
            int8_t* input = new int8_t[INPUT_SIZES[i]*INPUT_SIZES[i]*INPUT_DEPTHS[i]];
            int8_t* filters = new int8_t[FILTER_SIZES[i]*FILTER_SIZES[i]*INPUT_DEPTHS[i]];
            int8_t* biases = new int8_t[FILTER_COUNTS[i]];
            int8_t* output = CNN::convolution(input, INPUT_SIZES[i], INPUT_DEPTHS[i], filters, biases, FILTER_SIZES[i], FILTER_COUNTS[i], STRIDES[i], PADDINGS[i]);
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
            int8_t* input = new int8_t[INPUT_SIZES[i]*INPUT_SIZES[i]*INPUT_DEPTHS[i]];
            int8_t* filters = new int8_t[POOLING_SIZES[i]*POOLING_SIZES[i]*INPUT_DEPTHS[i]];
            int8_t* output = CNN::maxPooling(input, INPUT_SIZES[i], INPUT_DEPTHS[i], STRIDES[i], PADDINGS[i], POOLING_SIZES[i]);
            delete[] output;
        }
    }

    {
        int INPUT_LENGHTS[] = {3136, 128};
        int WEIGHT_COUNTS[] = {128, 10};

        for(int i = 0; i < 2; ++i) {
            int8_t* input = new int8_t[INPUT_LENGHTS[i]];
            int8_t* weights = new int8_t[WEIGHT_COUNTS[i]];
            int8_t* output = CNN::fullyConnected(input, INPUT_LENGHTS[i], weights, WEIGHT_COUNTS[i]);
        }
    }

    std::cout << "ENDED" << std::endl;
}