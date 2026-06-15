import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("enhanced_transactions.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Fraud counts
fraud_counts = df["isFraud"].value_counts()

print("\nFraud Counts:")
print(fraud_counts)

# Fraud percentages
fraud_percentages = df["isFraud"].value_counts(normalize=True) * 100

print("\nFraud Percentages:")
print(fraud_percentages.round(4))

# Summary
total_transactions = len(df)
fraud_transactions = (df["isFraud"] == 1).sum()
legitimate_transactions = (df["isFraud"] == 0).sum()

print("\nSummary:")
print(f"Total Transactions: {total_transactions:,}")
print(f"Fraud Transactions: {fraud_transactions:,}")
print(f"Legitimate Transactions: {legitimate_transactions:,}")

fraud_counts.plot(kind="bar")

plt.title("Fraud Distribution")
plt.xlabel("isFraud")
plt.ylabel("Number of Transactions")

plt.xticks([0, 1], ["Legitimate", "Fraud"])

plt.show()