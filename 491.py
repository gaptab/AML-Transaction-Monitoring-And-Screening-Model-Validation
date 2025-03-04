import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.stats import hypergeom

# Set random seed for reproducibility
np.random.seed(42)

# Define number of transactions
num_transactions = 100000

# Generate synthetic AML transaction data
data = {
    'Customer_ID': np.random.randint(1000, 11000, num_transactions),
    'Transaction_Amount': np.random.uniform(100, 50000, num_transactions),
    'Transaction_Type': np.random.choice(['Wire Transfer', 'Cash Deposit', 'Online Payment', 'International Transfer'], num_transactions),
    'Customer_Risk_Score': np.random.choice(['Low', 'Medium', 'High'], num_transactions, p=[0.7, 0.2, 0.1]),
    'Jurisdiction_Risk': np.random.choice(['Low', 'Medium', 'High'], num_transactions, p=[0.8, 0.15, 0.05]),
    'Unusual_Frequency': np.random.randint(0, 10, num_transactions),
    'Split_Transactions': np.random.choice([0, 1], num_transactions, p=[0.95, 0.05]),
    'PEP_Flag': np.random.choice([0, 1], num_transactions, p=[0.98, 0.02]),
    'Watchlist_Match': np.random.choice([0, 1], num_transactions, p=[0.99, 0.01]),
    'Suspicious': np.random.choice([0, 1], num_transactions, p=[0.95, 0.05])  # 5% fraud transactions
}

# Convert to DataFrame
df = pd.DataFrame(data)

# One-Hot Encoding for Categorical Features
encoder = OneHotEncoder(sparse_output=False)
categorical_features = ['Transaction_Type', 'Customer_Risk_Score', 'Jurisdiction_Risk']
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
encoded_features.columns = encoder.get_feature_names_out(categorical_features)

# Scale Numeric Features
scaler = StandardScaler()
numeric_features = ['Transaction_Amount', 'Unusual_Frequency']
scaled_features = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

# Merge Encoded and Scaled Data
X = pd.concat([scaled_features, encoded_features, df[['Split_Transactions', 'PEP_Flag', 'Watchlist_Match']]], axis=1)
y = df['Suspicious']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
print("Logistic Regression Results:")
print(classification_report(y_test, log_preds))
print("AUC-ROC Score:", roc_auc_score(y_test, log_preds))

tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
print("Decision Tree Results:")
print(classification_report(y_test, tree_preds))
print("AUC-ROC Score:", roc_auc_score(y_test, tree_preds))

tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
print("Decision Tree Results:")
print(classification_report(y_test, tree_preds))
print("AUC-ROC Score:", roc_auc_score(y_test, tree_preds))

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rf_preds))
print("AUC-ROC Score:", roc_auc_score(y_test, rf_preds))

# Define BTL Sampling Parameters
N = len(df)  # Total transactions
K = df['Suspicious'].sum()  # Total actual suspicious transactions
n = 500  # Sample size for testing

# Expected suspicious transactions in sample
expected_suspicious = hypergeom.mean(N, K, n)
print(f"Expected Suspicious Transactions in Sample: {expected_suspicious:.2f}")

# Perform BTL Testing
btl_sample = df.sample(n)
actual_suspicious = btl_sample['Suspicious'].sum()
print(f"Actual Suspicious Transactions Found: {actual_suspicious}")

# Compare Expected vs. Actual
if actual_suspicious < expected_suspicious:
    print("⚠️ AML Model is under-detecting fraud (Possible False Negatives)")
else:
    print("✅ AML Model detection is aligned with expectations")

# Checking impact of different threshold values
thresholds = np.arange(0.05, 1, 0.05)
best_threshold = 0
best_auc = 0

for threshold in thresholds:
    threshold_preds = (rf_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    auc = roc_auc_score(y_test, threshold_preds)
    
    if auc > best_auc:
        best_auc = auc
        best_threshold = threshold

print(f"Optimal Threshold for Fraud Detection: {best_threshold}")
print(f"Best AUC Score: {best_auc:.4f}")
