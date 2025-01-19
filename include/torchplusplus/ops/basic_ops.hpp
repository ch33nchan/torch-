#pragma once

#include "../tensor.hpp"

namespace torchplusplus {
namespace ops {

// Activation functions
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor softmax(const Tensor& input, int dim = -1);

// Loss functions
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target);

} // namespace ops
} // namespace torchplusplus