#include <iostream>
#include "tinygrad/scalar.hpp"

int main(){
    tinygrad::Scalar x{3.0};
    std::cout << x.value() << "\n";
    return 0;
}
