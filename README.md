# Credit Risk Engine in C++

## Overview

This project implements a **credit risk analysis pipeline** combining Python for data preparation and C++ for statistical modeling, portfolio simulation, and risk analysis.

The project uses the **LendingClub Loan dataset** and is designed as a quantitative finance / software engineering project, with an emphasis on implementing the core components in C++ rather than relying entirely on external machine-learning libraries.

The current system:

1. Preprocesses LendingClub loan data using Python
2. Estimates **Probability of Default (PD)** using a logistic regression model implemented in C++
3. Evaluates the predictive model using classification metrics
4. Constructs a loan portfolio from model predictions
5. Simulates portfolio losses using **Monte Carlo methods**
6. Models correlated defaults using a one-factor Gaussian copula
7. Computes portfolio-level **Expected Loss (EL)** and **Value at Risk (VaR)**
8. Performs PD stress testing under hypothetical recession scenarios


## Project Structure
```
.
├── cpp/
│ ├── io/       # Data loading
│ ├── models/   # Logistic regression
│ ├── utils/    # Metrics (accuracy, precision, recall, etc.)
│ ├── risk/     # Portfolio, simulation and risk analysis
│ └── main.cpp
│
├── data/
│ ├── raw/            # Original LendingClub dataset
│ └── processed/      # Cleaned and model-ready
│
├── src/
│ └── prepare_data.py # Data preprocessing pipeline
```


---

## Pipeline Description

### 1. Data Preparation (Python)

- Loads the raw LendingClub dataset
- Cleans and filters relevant features
- Encodes variables and defines target (`default = 1`)
- Outputs:
  - `lendingclub_X.csv` (features)
  - `lendingclub_y.csv` (target)

---

### 2. Probability of Default Model (C++)

A logistic regression model is implemented from scratch in C++.
The model estimates:
```
PD = P(Default | X)
```
where `PD` is the probability that a loan defaults given its characteristics.

The implementation uses:
- Eigen
- Gradient descent
- Sigmoid activation
- Continuous probability output

The predicted probabilities are subsequently used as inputs to the portfolio risk engine.


---

### 3. Model Evaluation

The dataset is:
- Randomly shuffled
- Split into training (80%) and testing (20%)

Metrics computed:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Because the dataset is imbalanced, accuracy alone is not considered sufficient to evaluate the model. <br>
A classification threshold of *0.3* is currently used when converting predicted probabilities into binary default/no-default predictions. <br>
Importantly, this threshold is used only for classification evaluation. The portfolio simulation uses the continuous PD values directly.

---

### 4. Portfolio Construction

Each portfolio loss is modeled as:
```
Loss = Default × EAD × LGD
```
Where:
- **PD**: predicted Probability of Default
- **EAD**: (Exposure at Default)
- **LGD**: Loss Gived Default, assumed constant (0.6)

The predicted PDs come directly from the logistic regression model.

---

### 5. Monte Carlo Simulation

The risk engine simulates many possible future states of the portfolio.
For each simulation:

- A default state is generated for every loan
- Defaults are determined according to the loan's PD
- Individual losses are calculated
- Individual losses are aggregated into total portfolio loss

Repeating this process produces a simulated portfolio loss distribution. <br>
This allows the project to move beyond individual loan classification and quantify aggregate portfolio risk.

---

### 6. Correlated Defaults (Gaussian Copula)

Assuming completely independent defaults is unrealistic because economic conditions can cause defaults to cluster.
The simulator therefore supports a one-factor Gaussian copula.

#### Model Description

Each loan is associated with a latent variable:
```
Z_i = √ρ · M + √(1 - ρ) · ε_i
```
Where:
- `M` is a common systemic (market) factor
- `ε_i` is an idiosyncratic noise term
- `ρ` is the asset correlation parameter, default *0.2*

A default occurs if:
```
Z_i < Φ⁻¹(PD_i) ,
```
where:
- `PD_i` is the predicted probability of default from the logistic regression model
- `Φ⁻¹` is the inverse standard normal Cumulative Distribution Function (CDF)

Typically, this leads to an increase of VaR, meaning that correlated defaults usually lead to an increased portfolio risk.

---

### 7. Risk Metrics

The simulated loss distribution is used to calculate portfolio-level risk measures.

From simulated losses:

- **Expected Loss (EL)**  
  Average portfolio loss

- **Value at Risk (VaR)**  
  - VaR 95% → loss threshold not exceeded in 95% of cases  
  - VaR 99% → extreme risk estimate  

---

### 8. Stress Testing

The risk engine supports hypothetical PD stress scenarios.

A stress scenario applies a multiplier to each loan's estimated PD before running the portfolio simulation.

For example: <br>
*Baseline:*
PD multiplier = 1, ρ = 0.2

*Mild recession:*
PD multiplier = 1.25, ρ = 0.25

*Severe recession:*
PD multiplier = 1.75, ρ = 0.35

The stressed PD is bounded to remain strictly between 0 and 1 before being passed to the Gaussian copula.

The current scenarios are conceptually:

`Baseline` --->
`Mild recession` --->
`Severe recession`

with increasingly adverse assumptions.

The purpose is not to forecast an actual recession, but to study the sensitivity of portfolio losses and tail risk to deteriorating credit conditions.




---

## Possible Future Improvements

- Improve PD calibration
- Introduce more realistic LGD models
- Model stochastic EAD

---

## How to Run

Compile from project root:
```
g++ -I /usr/include/eigen3 \
cpp/main.cpp \
cpp/io/DataLoader.cpp \
cpp/models/LogisticRegression.cpp \
cpp/utils/Metrics.cpp \
cpp/risk/Portfolio.cpp \
cpp/risk/Simulator.cpp \
cpp/risk/RiskAnalyzer.cpp \
-o cpp/model
```

Run: `./cpp/model`
