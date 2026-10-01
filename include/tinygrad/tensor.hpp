#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace tinygrad {

class Tensor {
public:
    Tensor(
        std::vector<double> data,
        std::vector<size_t> shape
    );

    Tensor(std::initializer_list<double> values);

    std::size_t size() const;
    std::size_t ndim() const;

    const std::vector<std::size_t>& shape() const;

    double& at(const std::vector<std::size_t>& indices);
    const double& at(const std::vector<std::size_t>& indices) const;

    const std::vector<double>& data() const;

private:
    std::vector<double> data_;
    std::vector<std::size_t> shape_;

    std::size_t flatten_index(const std::vector<std::size_t>& indices) const;
};

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);

}
