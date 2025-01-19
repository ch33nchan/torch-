#include <iostream>
#include "torchplusplus/tensor.hpp"
#include "torchplusplus/nn/linear.hpp"
#include "torchplusplus/ops/basic_ops.hpp"

using namespace torchplusplus;

void print_tensor(const std::string& name, const Tensor& tensor) {
    std::cout << name << ": ";
    for (const auto& val : tensor.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}

int main() {
    try {
        // Create input tensors
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {0.5f, 1.0f, 1.5f, 2.0f};
        Tensor t1(data1, {2, 2});
        Tensor t2(data2, {2, 2});

        // Try different operations
        print_tensor("Tensor 1", t1);
        print_tensor("Tensor 2", t2);
        
        // Addition
        Tensor sum = t1 + t2;
        print_tensor("Addition (t1 + t2)", sum);
        
        // Matrix multiplication
        Tensor matmul = t1.matmul(t2);
        print_tensor("Matrix multiplication", matmul);
        
        // Try different activation functions
        print_tensor("ReLU", ops::relu(t1));
        print_tensor("Sigmoid", ops::sigmoid(t1));
        print_tensor("Tanh", ops::tanh(t1));
        
        // Create a neural network layer
        nn::Linear layer(2, 3); // Input size: 2, Output size: 3
        Tensor output = layer.forward(t1);
        print_tensor("Linear layer output", output);
        
        // Apply softmax
        Tensor softmax_output = ops::softmax(output);
        print_tensor("Softmax output", softmax_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}