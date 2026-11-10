
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")                    #Load datasets
test = pd.read_csv("test.csv")

df = train.copy()                                   # Make copies so the original datasets stay unchanged
test_df = test.copy()       


print("\n--- Dataset Overview ---")
print("Training Dataset Shape:", df.shape)
print("Testing Dataset Shape:", test_df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Duplicate Check ---")
print("Duplicate Rows:", df.duplicated().sum())
print("Duplicate IDs:", df["Id"].duplicated().sum())


print("\n--- Missing Value Report ---")

missing_values = df.isnull().sum()

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

missing_report = missing_report[missing_report["Missing Values"] > 0]

missing_report = missing_report.sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_report)


print("\n--- Column Classification ---")

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("Numerical Columns:")
print(list(numerical_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumber of Numerical Columns:", len(numerical_columns))
print("Number of Categorical Columns:", len(categorical_columns))


print("\n--- Sale Price Check ---")                                 #Check the targeted columns

print("Missing Sale Prices:", df["SalePrice"].isnull().sum())
print("\nSale Price Summary:")
print(df["SalePrice"].describe())


print("\n--- Train and Test Column Check ---")

train_columns = set(df.columns)
test_columns = set(test_df.columns)

print("Columns only in training data:")
print(train_columns - test_columns)

print("\nColumns only in testing data:")
print(test_columns - train_columns)

print("\nInitial inspection complete.")
