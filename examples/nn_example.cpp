#include <iostream>
#include "torchplusplus/tensor.hpp"
#include "torchplusplus/nn/linear.hpp"
#include "torchplusplus/ops/basic_ops.hpp"

using namespace torchplusplus;

class SimpleNN {
public:
    SimpleNN() : 
        layer1(2, 4),   // Input size: 2, Hidden size: 4
        layer2(4, 1)    // Hidden size: 4, Output size: 1
    {}

    Tensor forward(const Tensor& input) {
        Tensor hidden = ops::relu(layer1.forward(input));
        return ops::sigmoid(layer2.forward(hidden));
    }

private:
    nn::Linear layer1;
    nn::Linear layer2;
};

int main() {
    try {
        // Create training data
        std::vector<float> input_data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
        std::vector<float> target_data = {0.0f, 1.0f, 1.0f, 1.0f};
        
        Tensor input(input_data, {4, 2});  // 4 samples, 2 features each
        Tensor target(target_data, {4, 1});

        // Create model
        SimpleNN model;

        // Training loop
        float learning_rate = 0.01f;
        for (int epoch = 0; epoch < 100; ++epoch) {
            // Forward pass
            Tensor output = model.forward(input);
            
            // Calculate loss
            Tensor loss = ops::mse_loss(output, target);

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: ";
                for (const auto& val : loss.data()) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}