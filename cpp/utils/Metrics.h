#pragma once
#include <Eigen/Dense>
#include <tuple>

// Train/test split
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,
           Eigen::VectorXd, Eigen::VectorXd>
train_test_split(const Eigen::MatrixXd& X,
                 const Eigen::VectorXd& y,
                 double test_ratio = 0.2);

// Metrics
double accuracy(const Eigen::VectorXd& y_true,
                const Eigen::VectorXd& y_pred);

void confusion_matrix(const Eigen::VectorXd& y_true,
                      const Eigen::VectorXd& y_pred);

void normalize(Eigen::MatrixXd& X);