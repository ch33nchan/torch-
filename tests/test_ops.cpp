#include <gtest/gtest.h>
#include "torchplusplus/nn/linear.hpp"

using namespace torchplusplus;
using namespace torchplusplus::nn;

TEST(LinearTest, Construction) {
    Linear layer(10, 5, true);
    // Check that the layer was created without throwing
    SUCCEED();
}

TEST(LinearTest, Forward) {
    Linear layer(2, 3, false);
    std::vector<float> input_data = {1.0f, 2.0f};
    Tensor input(input_data, {1, 2});
    
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.get_shape().size(), 2);
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 3);
}