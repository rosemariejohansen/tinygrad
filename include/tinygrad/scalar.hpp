#pragma once

#include <memory>
namespace tinygrad {

struct Node {
    double value;
    double grad;
};

class Scalar {
public:
    explicit Scalar(double value);

    double value() const;
    double grad() const;

private:
    std::shared_ptr<Node> node_;
};

}
