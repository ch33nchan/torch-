# Add example executables
add_executable(basic_example basic_examples.cpp)
target_link_libraries(basic_example PRIVATE torchplusplus)

add_executable(nn_example nn_example.cpp)
target_link_libraries(nn_example PRIVATE torchplusplus)

# Set properties
set_target_properties(basic_example nn_example
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)