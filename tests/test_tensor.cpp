#include <gtest/gtest.h>
#include "torchplusplus/tensor.hpp"

using namespace torchplusplus;

TEST(TensorTest, Construction) {
    // Test creation with shape
    Tensor t1({2, 3}, false);
    auto shape = t1.get_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Test creation with data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t2(data, {2, 2});
    EXPECT_EQ(t2.data().size(), 4);
    EXPECT_EQ(t2.data()[0], 1.0f);
}

TEST(TensorTest, Addition) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor t1(data1, {2, 2});
    Tensor t2(data2, {2, 2});
    
    Tensor result = t1 + t2;
    EXPECT_EQ(result.data()[0], 2.0f);
    EXPECT_EQ(result.data()[1], 3.0f);
}