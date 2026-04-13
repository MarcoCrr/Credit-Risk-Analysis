#include "Portfolio.h"

std::vector<Loan> Portfolio::build(const Eigen::MatrixXd& X,
                                   const Eigen::VectorXd& pd_preds) {

    std::vector<Loan> portfolio;

    for (int i = 0; i < X.rows(); ++i) {
        Loan loan;

        loan.pd = pd_preds(i);

        // IMPORTANT: adjust index if needed
        // assumes column 0 = loan amount
        loan.ead = X(i, 0);  

        loan.lgd = 0.6;

        portfolio.push_back(loan);
    }

    return portfolio;
}