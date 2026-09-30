#include "tinygrad/scalar.hpp"  

namespace tinygrad {

Scalar::Scalar(double value)
    : node_(std::make_shared<Node>(Node{value, 0.0})){
}

double Scalar::value() const {
    return node_->value;
}

double Scalar::grad() const{
    return node_->grad;
}

}
