#include "tinygrad/tensor.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace tinygrad {

namespace {

void check_same_shape(
    const Tensor& lhs,
    const Tensor& rhs
) {
    if(lhs.shape() != rhs.shape()) {
        throw std::invalid_argument("Tensor shapes must match");
    }
}

Tensor binary_operation(
    const Tensor& lhs,
    const Tensor& rhs,
    auto operation
) {
    check_same_shape(lhs, rhs);

    std::vector<double> result;

    result.reserve(lhs.size());

    for(std::size_t i = 0; i < lhs.size(); ++i) {
        result.push_back(operation(lhs.data()[i], rhs.data()[i]));
    }

    return Tensor{std::move(result), lhs.shape()};
}

Tensor scalar_operation(
    const Tensor& tensor,
    double scalar,
    auto operation
) {
    std::vector<double> result;
    result.reserve(tensor.size());

    for(const double value : tensor.data()) {
        result.push_back(operation(value, scalar));
    }

    return Tensor{std::move(result), tensor.shape()};
}

}

Tensor::Tensor(
    std::vector<double> data,
    std::vector<std::size_t> shape
) : data_(std::move(data)),
    shape_(std::move(shape)) {
    const std::size_t expected_size = std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},            
        std::multiplies<>()
    );

    if(expected_size != data_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape");
    }
}

Tensor::Tensor(std::initializer_list<double> values)
    : data_(values),
    shape_{values.size()} {
}

std::size_t Tensor::size() const {
    return data_.size();
}

std::size_t Tensor::ndim() const {
    return shape_.size();
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

double& Tensor::at(const std::vector<std::size_t>& indices){
    return data_.at(flatten_index(indices));
}

const double& Tensor::at(
    const std::vector<std::size_t>& indices
) const {
    return data_.at(flatten_index(indices));
}

const std::vector<double>& Tensor::data() const {
    return data_;
}

Tensor Tensor::reshape(
    std::vector<std::size_t> new_shape
) const {
    const std::size_t new_size = std::accumulate(
        new_shape.begin(),
        new_shape.end(),
        std::size_t{1},
        std::multiplies<>()
    );

    if(new_size != size()){
        throw std::invalid_argument("New shape must contain the same number of elements");
    }

    return Tensor{
        data_,
        std::move(new_shape)
    };
}

std::size_t Tensor::flatten_index(
    const std::vector<std::size_t>& indices
) const {
    if(indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Number of indices does not match tensor dimensions"
        );
    }

    std::size_t index = 0;
    std::size_t stride = 1;

    for(std::size_t dimension = shape_.size(); dimension > 0; --dimension) {
        const std::size_t i = dimension - 1;

        if(indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Tensor index out of range"
            );
        }

        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return index;
}

// Tensor <-> Tensor
Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs,
        rhs,
        [](double a, double b) {
            return a + b;
        }
    );
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs,
        rhs,
        [](double a, double b) {
            return a - b;
        }
    );
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs,
        rhs,
        [](double a, double b) {
            return a * b;
        }
    );
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs,
        rhs,
        [](double a, double b) {
            return a / b;
        }
    );
}

// Unary
Tensor operator-(const Tensor& tensor) {
    return scalar_operation(
        tensor,
        0.0,
        [](double value, double) {
            return -value;
        }
    );
}

// Tensor <-> scalar
Tensor operator+(const Tensor& tensor, double scalar) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return value + scalar_value;
        }
    );
}

Tensor operator+(double scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor operator-(const Tensor& tensor, double scalar) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return value - scalar_value;
        }
    );
}

Tensor operator-(double scalar, const Tensor& tensor) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return scalar_value - value;
        }
    );
}

Tensor operator*(const Tensor& tensor, double scalar) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return value * scalar_value;
        }
    );
}

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor operator/(const Tensor& tensor, double scalar) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return value / scalar_value;
        }
    );
}

Tensor operator/(double scalar, const Tensor& tensor) {
    return scalar_operation(
        tensor,
        scalar,
        [](double value, double scalar_value) {
            return scalar_value / value;
        }
    );
}

}
