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

    double sum() const;

    const std::vector<std::size_t>& shape() const;

    double& at(const std::vector<std::size_t>& indices);
    const double& at(const std::vector<std::size_t>& indices) const;

    const std::vector<double>& data() const;

    Tensor reshape(
        std::vector<std::size_t> new_shape
    ) const;
private:
    std::vector<double> data_;
    std::vector<std::size_t> shape_;

    std::size_t flatten_index(const std::vector<std::size_t>& indices) const;
};

// Tensor <-> Tensor    
Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);

// Unary
Tensor operator-(const Tensor& tensor);

// Tensor <-> scalar
Tensor operator+(const Tensor& tensor, double scalar);
Tensor operator+(double scalar, const Tensor& tensor);

Tensor operator-(const Tensor& tensor, double scalar);
Tensor operator-(double scalar, const Tensor& tensor);

Tensor operator*(const Tensor& tensor, double scalar);
Tensor operator*(double scalar, const Tensor& tensor);

Tensor operator/(const Tensor& tensor, double scalar);
Tensor operator/(double scalar, const Tensor& tensor);

Tensor matmul(const Tensor& lhs, const Tensor& rhs);

}
