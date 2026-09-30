#include <iostream>
#include "tinygrad/scalar.hpp"

int main(){
    tinygrad::Scalar x{3.0};

    std::cout << "value: " << x.value() << "\n";
    std::cout << "grad: " << x.grad() << "\n";

    return 0;
}