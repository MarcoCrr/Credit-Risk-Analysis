#include <iostream>
#include "LogisticRegression.h"
#include <cmath>
#include <tuple>



LogisticRegression::LogisticRegression(int n_features) {
    weights = Eigen::VectorXd::Zero(n_features);
    bias = 0.0;
}


Eigen::VectorXd LogisticRegression::sigmoid(const Eigen::VectorXd& z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}


void LogisticRegression::train(const Eigen::MatrixXd& X,
                               const Eigen::VectorXd& y,
                               double lr, int epochs) {

    const int n = X.rows();
    const Eigen::MatrixXd X_T = X.transpose();

    for (int epoch = 0; epoch < epochs; ++epoch) {

        Eigen::VectorXd z = (X * weights).array() + bias;
        Eigen::VectorXd preds = sigmoid(z);

        Eigen::VectorXd error = preds - y;

        Eigen::VectorXd grad_w = (X_T * error) / n;
        double grad_b = error.mean();

        weights.noalias() -= lr * grad_w;
        bias -= lr * grad_b;
    }
}


Eigen::VectorXd LogisticRegression::predict_proba(const Eigen::MatrixXd& X) {
    Eigen::VectorXd z = (X * weights).array() + bias;
    return sigmoid(z);
}


Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) {
    Eigen::VectorXd probs = predict_proba(X);
    return (probs.array() > 0.5).cast<double>();
}