# AML-Transaction-Monitoring-And-Screening-Model-Validation

This project validates Anti-Money Laundering (AML) transaction monitoring models by combining rule-based approaches (MANTAS, ACTIMIZE, NORKOM) with machine learning models (Logistic Regression, Decision Tree, and Random Forest). It also includes Below-the-Line (BTL) testing using Hypergeometric Sampling to evaluate model performance.

**Data Generation**

We create synthetic transaction data for AML validation. Each transaction has attributes like:

✅ Transaction Amount (random values between $100 - $50,000)

✅ Transaction Type (Wire Transfer, Cash Deposit, etc.)

✅ Customer Risk Score (Low, Medium, High)

✅ Jurisdiction Risk (Low, Medium, High)

✅ Unusual Frequency (number of repeated transactions)

✅ Flags for Suspicious Activity (Politically Exposed Persons (PEP), Watchlist match, Split transactions)

✅ Label for Suspicious Transactions (5% of transactions are randomly labeled as suspicious)

This dataset simulates real-world AML transaction data with risk-based factors.

**Data Preprocessing**

To prepare the data for machine learning:

✅ One-Hot Encoding converts categorical variables (Transaction Type, Risk Score) into numerical values.

✅ Standardization scales numeric features like Transaction Amount and Unusual Frequency for better model performance.

✅ Feature Selection keeps only relevant AML risk indicators.

✅ Splitting Data into training (70%) and testing (30%) to evaluate model performance.

**Machine Learning Models for AML Detection**

We train three different models to classify transactions as suspicious or non-suspicious:

1️⃣ Logistic Regression (Baseline Model)

A simple linear model that predicts suspicious transactions based on weighted risk factors.

Used as a benchmark to compare against more complex models.

2️⃣ Decision Tree (Rule-Based AML Model)

Mimics rule-based AML alerts by splitting transactions into different risk categories.

Can overfit if not tuned properly.

3️⃣ Random Forest (Best AML Model)

Uses multiple decision trees to improve accuracy.

Reduces false positives while detecting high-risk transactions better than rule-based systems.

Each model is evaluated using accuracy, precision, recall, and AUC-ROC score to measure how well it detects money laundering.

**Below-the-Line (BTL) Testing with Hypergeometric Sampling**

✅ Goal: Find false negatives (missed fraud cases) by testing unflagged transactions.

✅ Approach:

Use Hypergeometric Distribution to estimate the expected number of fraud cases in a random sample.

Compare the expected suspicious cases vs. actual suspicious cases.

If the actual suspicious count is lower than expected, it means the AML model is under-detecting fraud.

✅ Outcome:

If too many missed cases are found → Model needs threshold tuning or better detection rules.

If results match expectations → Model is performing well.

**Threshold Tuning for AML Optimization**

Rule-based AML models use fixed thresholds (e.g., flagging all transactions above $10,000).

Machine Learning models can dynamically adjust thresholds for better fraud detection.

The model is tested with different probability cutoffs to find the optimal threshold that reduces false positives while improving fraud detection.

✅ Final Optimization Step:

Finds the best decision boundary for classifying transactions as fraudulent or non-fraudulent.

Reduces false positives (unnecessary alerts) while still detecting high-risk transactions.
