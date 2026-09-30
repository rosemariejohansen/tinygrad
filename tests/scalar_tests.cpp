#include <cmath>
#include <iostream>
#include <string>

#include "tinygrad/scalar.hpp"

namespace {

constexpr double EPSILON = 1e-9;

void check(bool condition, const std::string& message){
    if(!condition){
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void check_close(
    double actual,
    double expected,
    const std::string& message
){
    if(std::abs(actual - expected)>EPSILON){
        std::cerr << "FAIL: " << message
            << " | expected: " << expected
            << ", actual: " << actual << '\n';

        std::exit(1);
    }
}

void test_basic_values(){
    tinygrad::Scalar a{2.0};

    check_close(a.value(), 2.0, "Scalar stores its value");
    check_close(a.grad(), 0.0, "Initial gradient is zero");
}

void test_addition(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};
    
    tinygrad::Scalar c = a + b;

    check_close(c.value(), 5.0, "Addition");
}

void test_subtraction(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a - b;

    check_close(c.value(), -1.0, "Subtraction");
}

void test_multiplication(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a * b;

    check_close(c.value(), 6.0, "Multiplication");
}

void test_division(){
    tinygrad::Scalar a{6.0};
    tinygrad::Scalar b{2.0};

    tinygrad::Scalar c = a / b;
    
    check_close(c.value(), 3.0, "Division");
}

void test_negation(){
    tinygrad::Scalar a{2.0};

    tinygrad::Scalar b = -a;

    check_close(b.value(), -2.0, "Negation");
}

void test_multiplication_gradients(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a * b;

    c.backward();

    check_close(a.grad(), 3.0, "dc/da for multiplication");
    check_close(b.grad(), 2.0, "dc/db for multiplication");
}

void test_addition_gradients(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a + b;

    c.backward();
    check_close(a.grad(), 1.0, "dc/da for addition");
    check_close(b.grad(), 1.0, "dc/db for addition");
}

void test_subtraction_gradients(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a - b;

    c.backward();

    check_close(a.grad(), 1.0, "dc/da for subtraction");
    check_close(b.grad(), -1.0, "dc/b for subtraction");    
}

void test_division_gradients(){
    tinygrad::Scalar a{6.0};
    tinygrad::Scalar b{2.0};

    tinygrad::Scalar c = a / b;
    
    c.backward();

    check_close(a.grad(), 0.5, "dc/da for division");
    check_close(b.grad(), -1.5, "dc/db for division");
}

void test_negation_gradient(){
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b = -a;

    b.backward();

    check_close(a.grad(), -1.0, "db/da for negation");
}

void test_multiple_paths() {
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a * b;
    tinygrad::Scalar d = c + a;

    d.backward();

    check_close(a.grad(), 4.0, "Gradient accumulation through multiple paths");
    check_close(b.grad(), 2.0, "Gradient through multiplication path");
}

void test_complex_expression() {
    tinygrad::Scalar a{6.0};
    tinygrad::Scalar b{2.0};

    tinygrad::Scalar c = a / b;
    tinygrad::Scalar d = -c;
    tinygrad::Scalar e = d - a;

    e.backward();

    check_close(e.value(), -9.0, "Complex expression value");
    check_close(a.grad(), -1.5, "de/da for complex expression");
    check_close(b.grad(), 1.5, "de/db for complex expression");
}

void test_repeated_backward() {
    tinygrad::Scalar a{2.0};
    tinygrad::Scalar b{3.0};

    tinygrad::Scalar c = a * b;

    c.backward();

    const double first_a_grad = a.grad();
    const double first_b_grad = b.grad();

    c.backward();

    check_close(
        a.grad(),
        first_a_grad,
        "Repeated backward does not accumulate stale gradient in a"
    );

    check_close(
        b.grad(),
        first_b_grad,
        "Repeated backward does not accumulate stale gradient in b"
    );
}

} // namespace

int main() {
    test_basic_values();

    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_negation();

    test_addition_gradients();
    test_subtraction_gradients();
    test_multiplication_gradients();
    test_division_gradients();
    test_negation_gradient();

    test_multiple_paths();
    test_complex_expression();
    test_repeated_backward();

    std::cout << "All Scalar tests passed.\n";

    return 0;
}