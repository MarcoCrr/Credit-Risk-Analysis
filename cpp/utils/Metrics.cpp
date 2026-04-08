#include "Metrics.h"
#include <iostream>
#include <random>
#include <algorithm>



std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,
           Eigen::VectorXd, Eigen::VectorXd>
train_test_split(const Eigen::MatrixXd& X,
                 const Eigen::VectorXd& y,
                 double test_ratio) {

    int n = X.rows();

    // Create shuffled indices
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    int test_size = static_cast<int>(n * test_ratio);
    int train_size = n - test_size;

    Eigen::MatrixXd X_train(train_size, X.cols());
    Eigen::MatrixXd X_test(test_size, X.cols());
    Eigen::VectorXd y_train(train_size);
    Eigen::VectorXd y_test(test_size);

    for (int i = 0; i < train_size; ++i) {
        X_train.row(i) = X.row(indices[i]);
        y_train(i) = y(indices[i]);
    }

    for (int i = 0; i < test_size; ++i) {
        X_test.row(i) = X.row(indices[train_size + i]);
        y_test(i) = y(indices[train_size + i]);
    }

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


void normalize(Eigen::MatrixXd& X) {
    for (int j = 0; j < X.cols(); ++j) {
        double mean = X.col(j).mean();

        double stddev = std::sqrt(
            (X.col(j).array() - mean).square().mean()
        );

        if (stddev > 1e-8) { // Avoid division by zero
            X.col(j) = (X.col(j).array() - mean) / stddev; // mean~0, stddev~1
        }
    }
}