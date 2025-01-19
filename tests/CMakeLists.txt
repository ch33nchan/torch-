# Add test executables
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor
    PRIVATE
        torchplusplus
        gtest
        gtest_main
)

add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear
    PRIVATE
        torchplusplus
        gtest
        gtest_main
)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops
    PRIVATE
        torchplusplus
        gtest
        gtest_main
)

# Register tests
add_test(NAME tensor_tests COMMAND test_tensor)
add_test(NAME linear_tests COMMAND test_linear)
add_test(NAME ops_tests COMMAND test_ops)