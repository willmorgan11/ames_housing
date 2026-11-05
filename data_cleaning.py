import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv")                                #Read the files
test = pd.read_csv("test.csv")

df = train.copy()                                                           #Copy the files so we aren't bothering the original datasets 
test_df = test.copy()

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nColumn Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nDuplicate Rows:", df.duplicated().sum())
