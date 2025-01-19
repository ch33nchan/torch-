#include <iostream>
#include "torchplusplus/tensor.hpp"
#include "torchplusplus/nn/linear.hpp"
#include "torchplusplus/ops/basic_ops.hpp"

using namespace torchplusplus;

int main() {
    // Create input tensor
    std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
    Tensor input(data, {2, 2});
    
    // Apply ReLU
    Tensor activated = ops::relu(input);
    
    // Print result
    std::cout << "Original data: ";
    for (const auto& val : input.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    
    std::cout << "After ReLU: ";
    for (const auto& val : activated.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    
    return 0;
}