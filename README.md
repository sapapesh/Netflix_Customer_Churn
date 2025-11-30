# Analysis of Netflix Customer Churn

## Introduction
Streaming platforms have increasingly grown year over year in the number of new subscribers; however, they have also seen an increase in cancellations each year resulting in a declining amount of new additions and a lowering retention rate. There are several factors that can influence customer churn, including user experience, pricing and subscription models, changes in viewing habits, and lack of personalization.

This project will review Netflix customer data in an attempt to determine which factors are influencing customer churn.

## Dataset
The analysis uses the dataset found on Kaggle: Netflix Customer Churn and Engagement Dataset: www.kaggle.com/datasets/dddtra/netflix-customer-churn-and-engagement-dataset

The data has been saved at the following location: https://github.com/sapapesh/Netflix_Customer_Churn/blob/main/data/netflix_large_user_data.csv

## Notebook File
The details of the analysis can be found in the Jupyter notebook file: https://github.com/sapapesh/Netflix_Customer_Churn/blob/main/notebooks/netflix.ipynb

## Overleaf Report
The Predicting Churn for Netflix report can be found at Overleaf: https://www.overleaf.com/read/jngmgycgmqtc#2775b5


---
## Set Up Your Machine
Proper setup is critical.
Complete each step in the following guide and verify carefully.

- [SET UP MACHINE](./SET_UP_MACHINE.md) (https://github.com/sapapesh/Netflix_Customer_Churn/blob/main/SET_UP_MACHINE.md)

## Set Up Your Project
After verifying your machine is set up, set up a new Python project by copying this template. Complete each step in the following guide.

- [SET UP PROJECT](./SET_UP_PROJECT.md) (https://github.com/sapapesh/Netflix_Customer_Churn/blob/main/SET_UP_PROJECT.md)

It includes the critical commands to set up your local environment (and activate it):

```shell
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

**Windows (PowerShell):**

```shell
.\.venv\Scripts\activate
```

**macOS / Linux / WSL:**

```shell
source .venv/bin/activate
```

---

## Imports

```
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## Section 1. Import and Inspect Data
### Section 1.1 - Load the Netflix Customer Data, display the info, and display the first 10 rows of data
```
# Load the Netflix Customer Churn data from the data folder
df = pd.read_csv(r"C:\Repos\Netflix_Customer_Churn\data\netflix_large_user_data.csv", delimiter=",")

# Display info
df.info()

# Display the first 10 rows
df.head(10)
```

### Section 1.2 - Check for missing and/or duplicate values
```
# Check for missing statistics
df.isnull().values.any()
df.isnull().sum()/df.shape[0]

# Check for duplicates
df.drop_duplicates(inplace = True)
```

### Section 1.3 - Display summary statistics
```
# Display summary statistics
print(df.describe())
```

### Section 1.4 - Check for unique values for Payment History, Subscription Plan, and Churn Status
```
# Check for unique values for Payment History, Subscription Plan, and Churn Status
print("Unique Payment History values:", df["Payment History (On-Time/Delayed)"].unique())
print("Unique Subscription Plan values:", df["Subscription Plan"].unique())
print("Unique Churn Status values:", df["Churn Status (Yes/No)"].unique())
```

### Section 1.5 - Converting data from categorical to numeric
```
# Convert categorical to numeric using mapping
payment_mapping = {"On-Time": 1, "Delayed": 0}
subscription_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
churn_mapping = {"Yes": 1, "No": 0}
genre_mapping = {"Action": 0, "Comedy": 1, "Documentary": 2, "Drama": 3, "Romance": 4, "Sci-Fi": 5, "Thriller": 6}
device_mapping = {"Desktop": 0, "Laptop": 1, "Mobile": 2, "Smart TV": 3, "Tablet": 4}

# Apply mappings
df["Payment History (Numeric)"] = df["Payment History (On-Time/Delayed)"].map(payment_mapping)
df["Subscription Plan (Numeric)"] = df["Subscription Plan"].map(subscription_mapping)
df["Churn Status (Numeric)"] = df["Churn Status (Yes/No)"].map(churn_mapping)
df["Genre Preference (Numeric)"] = df["Genre Preference"].map(genre_mapping)
df["Device Used Most Often (Numeric)"] = df["Device Used Most Often"].map(device_mapping)
```

## Section 2 -Exploratory Data Analysis
### Section 2.1 Generating histograms
```
# Generate histograms for all numeric columns
numeric_cols = df.select_dtypes(include="number").columns

# Set up the figure size based on number of columns
n_cols = len(numeric_cols)
fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(12, 4 * ((n_cols + 1) // 2)))
axes = axes.flatten()  # Flatten in case of single row

for i, col in enumerate(numeric_cols):
    df[col].hist(
        bins=20,                  # More bins for detail
        ax=axes[i],
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )
    axes[i].set_title(f"{col} distribution", fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
```
![alt text](notebooks/histograms/hist_Customer Satisfaction Score (1-10).png)
![alt text](notebooks/histograms/hist_Subscription_Plan.png)
![alt text](notebooks/histograms/hist_Payment_History__On-Time_Delayed_.png)
![alt text](notebooks/histograms/hist_Support_Queries_Logged.png)
![alt text](notebooks/histograms/hist_Age.png)
![alt text](notebooks/histograms/hist_Monthly_Income____.png)
![alt text](notebooks/histograms/hist_Support_Queries_Logged.png)
![alt text](notebooks/histograms/hist_Health_Score.png)
![alt text](notebooks/histograms/hist_Genre_Preference.png)
![alt text](notebooks/histograms/hist_Device_Used_Most_Often.png)
