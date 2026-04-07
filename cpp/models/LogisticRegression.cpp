#include "LogisticRegression.h"
#include <cmath>

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

    int n = X.rows();

    for (int epoch = 0; epoch < epochs; ++epoch) {

        Eigen::VectorXd z = X * weights + Eigen::VectorXd::Ones(n) * bias;
        Eigen::VectorXd preds = sigmoid(z);

        Eigen::VectorXd error = preds - y;

        Eigen::VectorXd grad_w = (X.transpose() * error) / n;
        double grad_b = error.mean();

        weights -= lr * grad_w;
        bias -= lr * grad_b;
    }
}

Eigen::VectorXd LogisticRegression::predict_proba(const Eigen::MatrixXd& X) {
    int n = X.rows();
    Eigen::VectorXd z = X * weights + Eigen::VectorXd::Ones(n) * bias;
    return sigmoid(z);
}

Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) {
    Eigen::VectorXd probs = predict_proba(X);
    return (probs.array() > 0.5).cast<double>();
}