#pragma once

#include "../tensor.hpp"

namespace torchplusplus {
namespace nn {

class Linear {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void zero_grad();
    void update_parameters(float learning_rate);
    
    // Getters
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }
    
private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Tensor weight_;
    Tensor bias_;
};

} // namespace nn
} // namespace torchplusplus