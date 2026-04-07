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


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,
           Eigen::VectorXd, Eigen::VectorXd>
train_test_split(const Eigen::MatrixXd& X,
                 const Eigen::VectorXd& y,
                 double test_ratio = 0.2) {

    int n = X.rows();
    int test_size = static_cast<int>(n * test_ratio);
    int train_size = n - test_size;

    Eigen::MatrixXd X_train = X.topRows(train_size);
    Eigen::MatrixXd X_test  = X.bottomRows(test_size);

    Eigen::VectorXd y_train = y.head(train_size);
    Eigen::VectorXd y_test  = y.tail(test_size);

    return {X_train, X_test, y_train, y_test};
}


double accuracy(const Eigen::VectorXd& y_true,
                const Eigen::VectorXd& y_pred) {

    int correct = 0;

    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) {
            correct++;
        }
    }

    return static_cast<double>(correct) / y_true.size();
}


void confusion_matrix(const Eigen::VectorXd& y_true,
                      const Eigen::VectorXd& y_pred) {

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == 1 && y_pred(i) == 1) tp++;
        else if (y_true(i) == 0 && y_pred(i) == 0) tn++;
        else if (y_true(i) == 0 && y_pred(i) == 1) fp++;
        else if (y_true(i) == 1 && y_pred(i) == 0) fn++;
    }

    std::cout << "\nConfusion Matrix:\n";
    std::cout << "TP: " << tp << "  FP: " << fp << "\n";
    std::cout << "FN: " << fn << "  TN: " << tn << "\n";
}