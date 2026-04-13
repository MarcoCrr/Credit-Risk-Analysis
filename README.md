# Credit Risk Engine in C++

## Overview

This project implements a **credit risk analysis pipeline** combining Python for data preparation and C++ for modeling and risk simulation, using the Credit Lending Club Loan data. I will expand the project, if time allows.

The system:
1. Preprocesses loan data
2. Trains a logistic regression model to estimate default probabilities (PD)
3. Evaluates model performance
4. Simulates portfolio losses using Monte Carlo methods
5. Computes key risk metrics such as Expected Loss and Value at Risk (VaR)


## Project Structure
```
.
├── cpp/
│ ├── io/       # Data loading
│ ├── models/   # Logistic regression
│ ├── utils/    # Metrics (accuracy, precision, recall, etc.)
│ ├── risk/     # Portfolio and Monte Carlo simulation
│ └── main.cpp
│
├── data/
│ ├── raw/
│ └── processed/      # Cleaned and model-ready
│
├── src/
│ └── prepare_data.py # Data preprocessing pipeline
```


---

## Pipeline Description

### 1. Data Preparation (Python)

- Loads LendingClub dataset
- Cleans and filters relevant features
- Encodes variables and defines target (`default = 1`)
- Outputs:
  - `lendingclub_X.csv` (features)
  - `lendingclub_y.csv` (target)

---

### 2. Model (C++)

A **logistic regression model** is implemented from scratch using Eigen:

- Gradient descent optimization
- Sigmoid activation
- Probability output (PD)

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

---

### 4. Portfolio Construction

Each loan is represented as:
```
Loss = Default × EAD × LGD
```
Where:
- **PD**: predicted probability of default
- **EAD**: loan amount
- **LGD**: assumed constant (0.6)

---

### 5. Monte Carlo Simulation

A portfolio of loans is simulated multiple times:

- For each loan:
  - Draw a random number
  - Default occurs if `u < PD`
- Compute total portfolio loss
- Repeat over many simulations

This produces a loss distribution.

---

### 6. Risk Metrics

From simulated losses:

- **Expected Loss (EL)**  
  Average portfolio loss

- **Value at Risk (VaR)**  
  - VaR 95% → loss threshold not exceeded in 95% of cases  
  - VaR 99% → extreme risk estimate  



---

## Some comments

- Accuracy alone is misleading for imbalanced datasets like this one
- Threshold tuning is essential in this context

---

## Future Improvements

- Correlated defaults (Gaussian copula) ?
- ...?

---

## How to Run

Compile from project root:
```
g++ -I /usr/include/eigen3
cpp/main.cpp
cpp/io/DataLoader.cpp
cpp/models/LogisticRegression.cpp
cpp/utils/Metrics.cpp
cpp/risk/Portfolio.cpp
cpp/risk/Simulator.cpp
-o cpp/model
```

Run: `./cpp/model`
