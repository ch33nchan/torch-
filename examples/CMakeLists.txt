# Add example executable
add_executable(basic_example basic_examples.cpp)
target_link_libraries(basic_example
    PRIVATE
        torchplusplus
)