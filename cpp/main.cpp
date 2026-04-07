#include <iostream>
#include <fstream>
#include "io/DataLoader.h"
#include "models/LogisticRegression.h"

int main() {
    std::ifstream test("data/processed/lendingclub_X.csv");
    std::cout << "File exists: " << test.good() << std::endl;
    Eigen::MatrixXd X = DataLoader::loadFeatures("data/processed/lendingclub_X.csv");
    Eigen::VectorXd y = DataLoader::loadTarget("data/processed/lendingclub_y.csv");

    LogisticRegression model(X.cols());

    model.train(X, y, 0.01, 1000);

    Eigen::VectorXd preds = model.predict(X);

    std::cout << "Training complete!" << std::endl;

    return 0;
}