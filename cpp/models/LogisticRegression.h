#pragma once
#include <Eigen/Dense>

class LogisticRegression {
private:
    Eigen::VectorXd weights;
    double bias;

public:
    LogisticRegression(int n_features);

    Eigen::VectorXd sigmoid(const Eigen::VectorXd& z);

    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
               double lr, int epochs);

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X);
};