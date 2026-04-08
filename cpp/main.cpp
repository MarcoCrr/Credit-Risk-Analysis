#include <iostream>
#include <fstream>
#include "io/DataLoader.h"
#include "models/LogisticRegression.h"
#include "utils/Metrics.h"

int main() {
    std::ifstream test("data/processed/lendingclub_X.csv");
    std::cout << "File exists: " << test.good() << std::endl;
    Eigen::MatrixXd X = DataLoader::loadFeatures("data/processed/lendingclub_X.csv");
    Eigen::VectorXd y = DataLoader::loadTarget("data/processed/lendingclub_y.csv");

    auto [X_train, X_test, y_train, y_test] = train_test_split(X, y);
    
    LogisticRegression model(X.cols());

    model.train(X_train, y_train, 0.01, 1000);

    Eigen::VectorXd preds = model.predict(X_test);
    std::cout << "Training complete!" << std::endl;

    double acc = accuracy(y_test, preds);
    std::cout << "Test Accuracy: " << acc << std::endl;

    confusion_matrix(y_test, preds);

    return 0;
}