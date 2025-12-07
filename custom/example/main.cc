#include <iostream>
#include <vector>
#include <numeric>
#include "custom_op.h" 

int main() {
    std::cout << "--- Starting ONNX Runtime Kernel Integration Test ---" << std::endl;
    
    // 1. Setup Data
    const size_t size = 8;
    
    // Input 1: Has negatives to prove ReLU works
    std::vector<float> input1_data = { -10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f, 100.0f };
    // Input 2: Values to add
    std::vector<float> input2_data = {   1.0f,  2.0f,  3.0f, 4.0f, 5.0f, 6.0f,  7.0f,   8.0f };
    
    // Expected Output Logic: max(0, input1) + input2
    // Expected: 1.0, 2.0, 3.0, 4.0, 6.0, 11.0, 17.0, 108.0

    std::vector<float> output_data(size);
    
    // 2. Call the dedicated test function
    // This function manages the ORT environment setup (Env, Allocator, Tensors) 
    // and calls SimpleReLUAddOpKernel::Compute with a Mock context.
    SimpleReLUAdd_ORT_Test(input1_data, input2_data, output_data, size);

    // 3. Print and verify results
    std::cout << "\nInput 1 (ReLU Applied): ";
    // Expected ReLU(I1): 0 0 0 0 1 5 10 100
    for(size_t i=0; i<size; ++i) {
        std::cout << std::max(0.0f, input1_data[i]) << " ";
    }
    std::cout << "\nInput 2 (Added):        1 2 3 4 5 6 7 8";
    
    std::cout << "\n\nResult (via ORT Kernel Test): \n";
    for (size_t i = 0; i < size; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nExpected Output:        1 2 3 4 6 11 17 108" << std::endl;
    
    return 0;
}