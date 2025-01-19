#include "torchplusplus/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace torchplusplus {

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape_(shape), 
      requires_grad_(requires_grad) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    data_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
    if (requires_grad) {
        grad_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
    }
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape)
    : shape_(shape),
      requires_grad_(false) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    if (data.size() != total_size) {
        throw ShapeMismatchError("Data size does not match specified shape");
    }
    data_ = std::make_shared<std::vector<float>>(data);
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw ShapeMismatchError("Tensor shapes must match for addition");
    }
    
    std::vector<float> result_data(data_->size());
    for (size_t i = 0; i < data_->size(); ++i) {
        result_data[i] = (*data_)[i] + (*other.data_)[i];
    }
    
    return Tensor(result_data, shape_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw ShapeMismatchError("Tensor shapes must match for subtraction");
    }
    
    std::vector<float> result_data(data_->size());
    for (size_t i = 0; i < data_->size(); ++i) {
        result_data[i] = (*data_)[i] - (*other.data_)[i];
    }
    
    return Tensor(result_data, shape_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw ShapeMismatchError("Tensor shapes must match for element-wise multiplication");
    }
    
    std::vector<float> result_data(data_->size());
    for (size_t i = 0; i < data_->size(); ++i) {
        result_data[i] = (*data_)[i] * (*other.data_)[i];
    }
    
    return Tensor(result_data, shape_);
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw ShapeMismatchError("Tensor shapes must match for division");
    }
    
    std::vector<float> result_data(data_->size());
    for (size_t i = 0; i < data_->size(); ++i) {
        if ((*other.data_)[i] == 0.0f) {
            throw std::runtime_error("Division by zero");
        }
        result_data[i] = (*data_)[i] / (*other.data_)[i];
    }
    
    return Tensor(result_data, shape_);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2 || shape_[1] != other.shape_[0]) {
        throw ShapeMismatchError("Invalid shapes for matrix multiplication");
    }
    
    std::vector<size_t> result_shape = {shape_[0], other.shape_[1]};
    std::vector<float> result_data(result_shape[0] * result_shape[1], 0.0f);
    
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < other.shape_[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < shape_[1]; ++k) {
                sum += (*data_)[i * shape_[1] + k] * (*other.data_)[k * other.shape_[1] + j];
            }
            result_data[i * result_shape[1] + j] = sum;
        }
    }
    
    return Tensor(result_data, result_shape);
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw ShapeMismatchError("Transpose operation currently only supports 2D tensors");
    }
    
    std::vector<size_t> result_shape = {shape_[1], shape_[0]};
    std::vector<float> result_data(data_->size());
    
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result_data[j * shape_[0] + i] = (*data_)[i * shape_[1] + j];
        }
    }
    
    return Tensor(result_data, result_shape);
}

void Tensor::backward() {
    if (!requires_grad_) {
        throw GradientError("Tensor does not require gradients");
    }
    
    for (auto it = grad_fn_.rbegin(); it != grad_fn_.rend(); ++it) {
        (*it)();
    }
}

void Tensor::zero_grad() {
    if (requires_grad_ && grad_) {
        std::fill(grad_->begin(), grad_->end(), 0.0f);
    }
}

} // namespace torchplusplus