#include "tinygrad/scalar.hpp"  
#include <unordered_set>
#include <vector>

namespace {

void build_topological_order(
    const std::shared_ptr<tinygrad::Node>& node,
    std::vector<std::shared_ptr<tinygrad::Node>>& order,
    std::unordered_set<tinygrad::Node*>& visited
){
    if(!visited.insert(node.get()).second)
        return;

    for(const auto& parent: node->parents)
        build_topological_order(parent, order, visited);

    order.push_back(node);
}
}
namespace tinygrad {

Scalar::Scalar(double value)
    : node_(std::make_shared<Node>(Node{value, 0.0, Operation::None, {}})){
}

double Scalar::value() const {
    return node_->value;
}

double Scalar::grad() const{
    return node_->grad;
}

void Scalar::backward(){
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;

    build_topological_order(node_, order, visited);

    for(const auto& node : order)
        node->grad = 0.0;  

    node_->grad = 1.0;

    for(auto it = order.rbegin(); it != order.rend(); ++it){
        const auto& node = *it;
        switch(node->operation){
            case Operation::Multiply: {
                auto& lhs = node->parents[0];
                auto& rhs = node->parents[1];

                lhs->grad += node->grad * rhs->value;
                rhs->grad += node->grad * lhs->value;

                break;
            }

            case Operation::Add: {
                auto& lhs = node->parents[0];
                auto& rhs = node->parents[1];

                lhs->grad += node->grad;
                rhs->grad += node->grad;

                break;
            }

            case Operation::Subtract: {
                auto& lhs = node->parents[0];
                auto& rhs = node->parents[1];

                lhs->grad += node->grad;
                rhs->grad -= node->grad;

                break;
            }

            case Operation::Divide: {
                auto& lhs = node->parents[0];
                auto& rhs = node->parents[1];

                lhs->grad += node->grad / rhs->value;
                rhs->grad -= node->grad * lhs->value
                       / (rhs->value * rhs->value);

                break;
            }

            case Operation::Negate: {
                auto& parent = node->parents[0];

                parent->grad -= node->grad;

                break;
            }

            case Operation::None:
                break;
        }
    }
}

Scalar operator*(const Scalar& lhs, const Scalar& rhs){
    Scalar result{lhs.value() * rhs.value()};

    result.node_->operation = Operation::Multiply;

    result.node_->parents.push_back(lhs.node_);
    result.node_->parents.push_back(rhs.node_);

    return result;
}

Scalar operator+(const Scalar& lhs, const Scalar& rhs){
    Scalar result{lhs.value() + rhs.value()};

    result.node_->operation = Operation::Add;
    
    result.node_->parents.push_back(lhs.node_);
    result.node_->parents.push_back(rhs.node_);

    return result;
}

Scalar operator-(const Scalar& lhs, const Scalar& rhs){
    Scalar result{lhs.value() - rhs.value()};

    result.node_->operation = Operation::Subtract;
    
    result.node_->parents.push_back(lhs.node_);
    result.node_->parents.push_back(rhs.node_);

    return result;
}

Scalar operator/(const Scalar& lhs, const Scalar& rhs){
    Scalar result{lhs.value() / rhs.value()};

    result.node_->operation = Operation::Divide;

    result.node_->parents.push_back(lhs.node_);
    result.node_->parents.push_back(rhs.node_);

    return result;
}

Scalar operator-(const Scalar& value){
    Scalar result{-value.value()};
    
    result.node_->operation = Operation::Negate;
    result.node_->parents.push_back(value.node_);

    return result;
}

}
