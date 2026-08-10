#include <iostream>
#include <fstream>
#include "io/DataLoader.h"
#include "models/LogisticRegression.h"
#include "utils/Metrics.h"
#include "risk/Portfolio.h"
#include "risk/Simulator.h"
#include "risk/Scenario.h"
#include "risk/RiskReport.h"
#include "risk/RiskAnalyzer.h"

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
    

    // Hypothetical stress assumptions
    Scenario baseline{
        "Baseline",
        1.0, // pd_multiplier
        0.2, // rho
        1.0  // lgd_multiplier
    };

    Scenario mild_recession{
        "Mild recession",
        1.25,
        0.25,
        1.0
    };

    Scenario severe_recession{
        "Severe recession",
        1.75,
        0.35,
        1.0
    };

    RiskReport report = RiskAnalyzer::analyze(portfolio, baseline);
    RiskReport mild_report = RiskAnalyzer::analyze(portfolio, mild_recession);
    RiskReport severe_report = RiskAnalyzer::analyze(portfolio, severe_recession);


    std::cout << "\n=== Stress Testing ===\n";

    std::cout << "\nBaseline\n";
    std::cout << "Expected Loss: "
            << report.expected_loss << std::endl;
    std::cout << "VaR 95%: "
            << report.var95 << std::endl;
    std::cout << "VaR 99%: "
            << report.var99 << std::endl;

    std::cout << "\nMild Recession\n";
    std::cout << "Expected Loss: "
            << mild_report.expected_loss << std::endl;
    std::cout << "VaR 95%: "
            << mild_report.var95 << std::endl;
    std::cout << "VaR 99%: "
            << mild_report.var99 << std::endl;

    std::cout << "\nSevere Recession\n";
    std::cout << "Expected Loss: "
            << severe_report.expected_loss << std::endl;
    std::cout << "VaR 95%: "
            << severe_report.var95 << std::endl;
    std::cout << "VaR 99%: "
            << severe_report.var99 << std::endl;

    return 0;
}