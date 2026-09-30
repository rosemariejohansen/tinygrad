#include <iostream>

#include "tinygrad/scalar.hpp"

int main() {
    tinygrad::Scalar a{6.0};
    tinygrad::Scalar b{2.0};

    tinygrad::Scalar c = a / b;
    tinygrad::Scalar d = -c;
    tinygrad::Scalar e = d - a;

    e.backward();

    std::cout << "e: " << e.value() << '\n';
    std::cout << "de/da: " << a.grad() << '\n';
    std::cout << "de/db: " << b.grad() << '\n';
}