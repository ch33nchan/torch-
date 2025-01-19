#include "torchplusplus/ops/basic_ops.hpp"
#include <cmath>
#include <algorithm>

namespace torchplusplus {
namespace ops {

Tensor relu(const Tensor& input) {
    std::vector<float> result_data = input.data();
    for (auto& val : result_data) {
        val = std::max(0.0f, val);
    }
    return Tensor(result_data, input.get_shape());
}

Tensor sigmoid(const Tensor& input) {
    std::vector<float> result_data = input.data();
    for (auto& val : result_data) {
        val = 1.0f / (1.0f + std::exp(-val));
    }
    return Tensor(result_data, input.get_shape());
}

Tensor tanh(const Tensor& input) {
    std::vector<float> result_data = input.data();
    for (auto& val : result_data) {
        val = std::tanh(val);
    }
    return Tensor(result_data, input.get_shape());
}

Tensor softmax(const Tensor& input, int dim) {
    if (input.get_shape().size() != 2) {
        throw ShapeMismatchError("Softmax currently only supports 2D tensors");
    }
    
    std::vector<float> result_data = input.data();
    const auto& shape = input.get_shape();
    
    if (dim == -1) dim = shape.size() - 1;
    
    if (dim == 1) {
        for (size_t i = 0; i < shape[0]; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < shape[1]; ++j) {
                max_val = std::max(max_val, result_data[i * shape[1] + j]);
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < shape[1]; ++j) {
                size_t idx = i * shape[1] + j;
                result_data[idx] = std::exp(result_data[idx] - max_val);
                sum += result_data[idx];
            }
            
            for (size_t j = 0; j < shape[1]; ++j) {
                result_data[i * shape[1] + j] /= sum;
            }
        }
    }
    
    return Tensor(result_data, shape);
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.get_shape() != target.get_shape()) {
        throw ShapeMismatchError("Shapes must match for MSE loss");
    }
    
    float loss = 0.0f;
    const auto& pred_data = pred.data();
    const auto& target_data = target.data();
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        float diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    
    loss /= pred_data.size();
    return Tensor(std::vector<float>{loss}, std::vector<size_t>{1});
}

Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target) {
    if (pred.get_shape() != target.get_shape()) {
        throw ShapeMismatchError("Shapes must match for cross entropy loss");
    }
    
    float loss = 0.0f;
    const auto& pred_data = pred.data();
    const auto& target_data = target.data();
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        loss += -target_data[i] * std::log(std::max(pred_data[i], 1e-7f));
    }
    
    loss /= pred_data.size();
    return Tensor(std::vector<float>{loss}, std::vector<size_t>{1});
}

} // namespace ops
} // namespace torchplusplus