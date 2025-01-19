#include "torchplusplus/nn/linear.hpp"
#include <random>

namespace torchplusplus {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(bias),
      weight_({in_features, out_features}, true),
      bias_(use_bias_ ? std::vector<size_t>{out_features} : std::vector<size_t>{}, true) {
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    auto& weight_data = weight_.data();
    for (auto& w : weight_data) {
        w = dis(gen);
    }
    
    if (use_bias_) {
        std::fill(bias_.data().begin(), bias_.data().end(), 0.0f);
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (input.get_shape().size() != 2 || input.get_shape()[1] != in_features_) {
        throw ShapeMismatchError("Invalid input shape for linear layer");
    }
    
    Tensor output = input.matmul(weight_);
    
    if (use_bias_) {
        auto& output_data = output.data();
        const auto& bias_data = bias_.data();
        for (size_t i = 0; i < output.get_shape()[0]; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                output_data[i * out_features_ + j] += bias_data[j];
            }
        }
    }
    
    return output;
}

void Linear::zero_grad() {
    weight_.zero_grad();
    if (use_bias_) {
        bias_.zero_grad();
    }
}

void Linear::update_parameters(float learning_rate) {
    auto& weight_data = weight_.data();
    auto& weight_grad = weight_.data();
    
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] -= learning_rate * weight_grad[i];
    }
    
    if (use_bias_) {
        auto& bias_data = bias_.data();
        auto& bias_grad = bias_.data();
        
        for (size_t i = 0; i < bias_data.size(); ++i) {
            bias_data[i] -= learning_rate * bias_grad[i];
        }
    }
}

} // namespace nn
} // namespace torchplusplus