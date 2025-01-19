#pragma once

#include <stdexcept>
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

} // namespace torchplusplus