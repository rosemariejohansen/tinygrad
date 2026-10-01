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

}

int main() {
    test_one_dimensional_tensor();
    test_two_dimensional_tensor();
    test_tensor_element_assignment();
    test_invalid_shape();
    test_invalid_number_of_indices();
    test_out_of_range_index();
    test_one_dimensional_initializer();
    std::cout << "All Tensor tests passed.\n";
    return 0;  
}