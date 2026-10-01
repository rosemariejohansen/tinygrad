#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinygrad/tensor.hpp"

namespace {

void check(bool condition, const std::string& message) {
    if(!condition){
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void test_one_dimensional_tensor() {
    tinygrad::Tensor tensor{
        std::vector<double>{1.0, 2.0, 3.0}, 
        std::vector<std::size_t>{3}
    };

    check(tensor.size() == 3, "1D tensor size");
    check(tensor.ndim() == 1, "1D tensor dimensions");
    check(tensor.shape() == std::vector<std::size_t>{3}, "1D tensor shape");
    check(tensor.at({0}) == 1.0, "1D tensor first element");
    check(tensor.at({2}) == 3.0, "1D tensor last element");
}

void test_two_dimensional_tensor() {
    tinygrad::Tensor tensor{
        std::vector<double> {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        },
        std::vector<std::size_t>{2, 3}
    };

    check(tensor.size() == 6, "2D tensor size");
    check(tensor.ndim() == 2, "2D tensor dimensions");
    check(tensor.shape() == std::vector<std::size_t>{2, 3}, "2D tensor shape");

    check(tensor.at({0, 0}) == 1.0, "tensor[0, 0]");
    check(tensor.at({0, 1}) == 2.0, "tensor[0, 1]");
    check(tensor.at({0, 2}) == 3.0, "tensor[0, 2]");
    check(tensor.at({1, 0}) == 4.0, "tensor[1, 0]");
    check(tensor.at({1, 1}) == 5.0, "tensor[1, 1]");
    check(tensor.at({1, 2}) == 6.0, "tensor[1, 2]");
}

void test_tensor_element_assignment() {
    tinygrad::Tensor tensor {
        std::vector<double> {
            1.0, 2.0,
            3.0, 4.0
        },
        std::vector<std::size_t>{2, 2}
    };

    tensor.at({1, 0}) = 10.0;

    check(tensor.at({1,0}) == 10.0, "Tensor element assignment");
}

void test_invalid_shape() {
    bool threw = false;

    try {
        tinygrad::Tensor tensor {
            std::vector<double>{1.0, 2.0, 3.0},
            std::vector<std::size_t>{2, 2}
        };

    } catch(const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "Invalid shape throws exception");
}

void test_invalid_number_of_indices() {
    tinygrad::Tensor tensor {
        std::vector<double>{
            1.0, 2.0,
            3.0, 4.0
        },
        std::vector<std::size_t>{2, 2}
    };

    bool threw = false;

    try {
        tensor.at({0});
    } catch(const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "Invalid number of indices throws exception");
}

void test_out_of_range_index() {
    tinygrad::Tensor tensor {
        std::vector<double> {
            1.0, 2.0,
            3.0, 4.0
        },
        std::vector<std::size_t>{2, 2}
    };

    bool threw = false;
    try {
        tensor.at({2, 0});
    } catch(const std::out_of_range&) {
        threw = true;
    }

    check(threw, "Out of range index throws exception");
}

void test_one_dimensional_initializer() {
    tinygrad::Tensor tensor{1.0, 2.0, 3.0};

    check(tensor.size() == 3, "Initializer list size");
    check(tensor.ndim() == 1, "Initializer list dimensions");
    check(tensor.shape() == std::vector<std::size_t>{3}, "Initializer list shape");
    check(tensor.at({0}) == 1.0, "Initializer list first element");
    check(tensor.at({2}) == 3.0, "Initializer list last element");
}

void test_elementwise_addition() {
    tinygrad::Tensor a{
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::size_t>{3}
    };

    tinygrad::Tensor b{
        std::vector<double>{4.0, 5.0, 6.0},
        std::vector<std::size_t>{3}
    };

    const tinygrad::Tensor c = a + b;

    check(c.data() == std::vector<double>{5.0, 7.0, 9.0}, "Element-wise addition");
}

void test_elementwise_subtraction() {
    tinygrad::Tensor a{
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::size_t>{3}
    };

    tinygrad::Tensor b{
        std::vector<double>{4.0, 5.0, 6.0},
        std::vector<std::size_t>{3}
    };

    const tinygrad::Tensor c = a - b;

    check(c.data() == std::vector<double>{-3.0, -3.0, -3.0}, "Element-wise subtraction");
}

void test_elementwise_multiplication() {
    tinygrad::Tensor a{
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::size_t>{3}
    };

    tinygrad::Tensor b{
        std::vector<double>{4.0, 5.0, 6.0},
        std::vector<std::size_t>{3}
    };

    const tinygrad::Tensor c = a * b;

    check(c.data() == std::vector<double>{4.0, 10.0, 18.0}, "Element-wise multiplication");
}

void test_elementwise_division() {
    tinygrad::Tensor a{
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::size_t>{3}
    };

    tinygrad::Tensor b{
        std::vector<double>{4.0, 5.0, 6.0},
        std::vector<std::size_t>{3}
    };

    const tinygrad::Tensor c = a / b;

    check(c.data() == std::vector<double>{0.25,0.4, 0.5}, "Element-wise division");
}

void test_elementwise_shape_mismatch() {
    tinygrad::Tensor a{
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::size_t>{3}
    };

    tinygrad::Tensor b{
        std::vector<double>{4.0, 5.0},
        std::vector<std::size_t>{2}
    };

    bool threw = false;
    
    try {
        const auto c = a + b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "Element-wise operation rejects different shapes");
}

}

int main() {
    test_one_dimensional_tensor();
    test_two_dimensional_tensor();
    test_tensor_element_assignment();
    test_invalid_shape();
    test_invalid_number_of_indices();
    test_out_of_range_index();
    test_one_dimensional_initializer();
    test_elementwise_addition();
    test_elementwise_division();
    test_elementwise_multiplication();
    test_elementwise_subtraction();
    test_elementwise_shape_mismatch();

    std::cout << "All Tensor tests passed.\n";
    return 0;  
}