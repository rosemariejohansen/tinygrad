#pragma once

#include <memory>
#include <vector>

namespace tinygrad {

enum class Operation {
    None,
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate
};

struct Node {
    double value;
    double grad;
    Operation operation;

    std::vector<std::shared_ptr<Node>> parents;
};

class Scalar {
public:
    explicit Scalar(double value);

    double value() const;
    double grad() const;

    void backward();
private:
    std::shared_ptr<Node> node_;

    friend Scalar operator*(const Scalar& lhs, const Scalar& rhs);
    friend Scalar operator+(const Scalar& lhs, const Scalar& rhs);
    friend Scalar operator-(const Scalar& lhs, const Scalar& rhs);
    friend Scalar operator/(const Scalar& lhs, const Scalar& rhs);
    friend Scalar operator-(const Scalar& value);
};

Scalar operator*(const Scalar& lhs, const Scalar& rhs);
Scalar operator+(const Scalar& lhs, const Scalar& rhs);
Scalar operator-(const Scalar& lhs, const Scalar& rhs);
Scalar operator/(const Scalar& lhs, const Scalar& rhs);
Scalar operator-(const Scalar& value);
}
