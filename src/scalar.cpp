#include "tinygrad/scalar.hpp"

namespace tinygrad {

Scalar::Scalar(double value)
    : value_(value) {
}

double Scalar::value() const {
    return value_;
}

}