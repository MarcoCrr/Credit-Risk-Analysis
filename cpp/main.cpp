#include <iostream>
#include <fstream>
#include "io/DataLoader.h"
#include "models/LogisticRegression.h"
#include "utils/Metrics.h"
#include "risk/Portfolio.h"
#include "risk/Simulator.h"

int main() {
    std::ifstream test("data/processed/lendingclub_X.csv");
    std::cout << "File exists: " << test.good() << std::endl;
    Eigen::MatrixXd X = DataLoader::loadFeatures("data/processed/lendingclub_X.csv");
    Eigen::VectorXd y = DataLoader::loadTarget("data/processed/lendingclub_y.csv");

    normalize(X);
    auto [X_train, X_test, y_train, y_test] = train_test_split(X, y);
    
    LogisticRegression model(X.cols());

    model.train(X_train, y_train, 0.01, 1000);

    Eigen::VectorXd probs = model.predict_proba(X_test);
    double threshold = 0.3;
    Eigen::VectorXd preds = (probs.array() > threshold).cast<double>();
    // Eigen::VectorXd preds = model.predict(X_test);   // OLD
    
    std::cout << "Training complete!" << std::endl;

    double acc = accuracy(y_test, preds);
    std::cout << "Test Accuracy: " << acc << std::endl;

    confusion_matrix(y_test, preds);

    std::cout << "\nPrecision: " << precision(y_test, preds) << std::endl;
    std::cout << "Recall: " << recall(y_test, preds) << std::endl;
    std::cout << "F1 Score: " << f1_score(y_test, preds) << std::endl;

    
    // risk-related
    // Build portfolio
    auto portfolio = Portfolio::build(X_test, probs);

    // Run simulation
    // Independendt defaults (old)
    // auto losses = Simulator::run(portfolio, 1000);
    
    // Correlated defaults
    double rho = 0.2;  // 0.1–0.3
    auto losses = Simulator::run_correlated(portfolio, 1000, rho);

    // Risk metrics
    std::cout << "Expected Loss: "
            << Simulator::expected_loss(losses) << std::endl;

    std::cout << "VaR 95%: "
            << Simulator::var(losses, 0.95) << std::endl;

    std::cout << "VaR 99%: "
            << Simulator::var(losses, 0.99) << std::endl;

    return 0;
}