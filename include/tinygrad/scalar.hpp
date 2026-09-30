#pragma once

namespace tinygrad {

class Scalar {
public:
    explicit Scalar(double value);
    double value() const;

private:
    double value_;
};

}