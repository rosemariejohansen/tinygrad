#include "tinygrad/tensor.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace tinygrad {

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

}
