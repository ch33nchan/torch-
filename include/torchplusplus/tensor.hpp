#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace torchplusplus {

class ShapeMismatchError : public std::runtime_error {
public:
    explicit ShapeMismatchError(const std::string& msg) : std::runtime_error(msg) {}
};

class GradientError : public std::runtime_error {
public:
    explicit GradientError(const std::string& msg) : std::runtime_error(msg) {}
};

class Tensor {
public:
    // Constructors
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape);
    
    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    
    // Gradient operations
    void backward();
    void zero_grad();
    
    // Utility functions
    std::vector<size_t> get_shape() const { return shape_; }
    std::vector<float>& data() { return *data_; }
    const std::vector<float>& data() const { return *data_; }
    bool requires_grad() const { return requires_grad_; }
    
private:
    std::shared_ptr<std::vector<float>> data_;
    std::vector<size_t> shape_;
    std::shared_ptr<std::vector<float>> grad_;
    bool requires_grad_;
    std::vector<std::function<void()>> grad_fn_;
};

} // namespace torchplusplus