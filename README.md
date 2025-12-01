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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
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
![alt text](<notebooks/histograms/hist_Customer Satisfaction Score (1-10).png>)
![alt text](notebooks/histograms/hist_Subscription_Plan.png)
![alt text](notebooks/histograms/hist_Payment_History__On-Time_Delayed_.png)
![alt text](notebooks/histograms/hist_Support_Queries_Logged.png)
![alt text](notebooks/histograms/hist_Age.png)
![alt text](notebooks/histograms/hist_Monthly_Income____.png)
![alt text](notebooks/histograms/hist_Support_Queries_Logged.png)
![alt text](notebooks/histograms/hist_Health_Score.png)
![alt text](notebooks/histograms/hist_Genre_Preference.png)
![alt text](notebooks/histograms/hist_Device_Used_Most_Often.png)



### 2.2 Generating the heatmap
```
# Folder to save images
output_folder = Path("heatmaps")
output_folder.mkdir(parents=True, exist_ok=True)

# Compute correlation matrix
corr = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,          # show correlation values in cells
    fmt=".2f",           # number format
    cmap="coolwarm",     # color scheme
    square=True,
    cbar=True
)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()

# Save
heatmap_path = output_folder / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Heatmap saved to {heatmap_path}")
```
![alt text](notebooks/heatmaps/correlation_heatmap.png)

A review of the correlation heatmap shows almost no correlation between the attributes.  Therefore, there does not appear to be a singular attribute that will contribute to the customer churn rate.

### Generating scatterplots
```
# Adding scatterplots
# === SETUP ===
scatter_folder = Path("scatterplots")
scatter_folder.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name):
    """Replace invalid filename characters with underscores."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


# === DETECT CHURN COLUMN ===
# Priority: 'churn_status' → 'churn' → last column
if "churn_status" in df.columns:
    churn_col = "churn_status"
elif "churn" in df.columns:
    churn_col = "churn"
else:
    churn_col = df.columns[-1]  # fallback to last column

print(f"🎯 Using '{churn_col}' as the churn column (x-axis).")

# === SCATTERPLOTS ===
# Select numeric columns except churn column
numeric_cols = df.select_dtypes(include=["number"]).columns.drop(churn_col, errors="ignore")

if len(numeric_cols) == 0:
    print("⚠️ No numeric feature columns found — skipping scatterplots.")
else:
    for col in numeric_cols:
        safe_col = sanitize_filename(col)
        filename = scatter_folder / f"scatter_{sanitize_filename(churn_col)}_vs_{safe_col}.png"

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[churn_col], y=df[col], alpha=0.6)
        plt.title(f"{col} by {churn_col}")
        plt.xlabel(churn_col)
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"✅ Scatterplots saved to: {scatter_folder.resolve()}")
```
![alt text](notebooks/scatterplots/scatter_Customer_Satisfaction_Score__1-10__vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Subscription_Plan__Numeric__vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Payment_History__Numeric__vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Support_Queries_Logged_vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Age_vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Monthly_Income_____vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Health_Score_vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Genre_Preference__Numeric__vs_Churn_Status__Numeric_.png)
![alt text](notebooks/scatterplots/scatter_Device_Used_Most_Often__Numeric__vs_Churn_Status__Numeric_.png)

Scatterplots has been completed to compare each attribute to churn status. However, the scatterplots did not present any clear patterns and appeared to be fairly balanced.

### 2.4 - Generating violin plots
```
# Adding violin plots
# === SETUP ===
violin_folder = Path("violinplots")
violin_folder.mkdir(parents=True, exist_ok=True)

# === VIOLIN PLOTS ===
# Use the same churn column and numeric columns already detected
if len(numeric_cols) == 0:
    print("⚠️ No numeric feature columns found — skipping violin plots.")
else:
    for col in numeric_cols:
        safe_col = sanitize_filename(col)
        filename = violin_folder / f"violin_{sanitize_filename(churn_col)}_vs_{safe_col}.png"

        plt.figure(figsize=(6, 4))
        sns.violinplot(
            data=df,
            x=churn_col,
            y=col,
            cut=0,         # keeps violins within actual data range
            inner="quartile",  # adds lines for Q1, median, Q3
        )
        plt.title(f"{col} by {churn_col}")
        plt.xlabel(churn_col)
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"🎻 Violin plots saved to: {violin_folder.resolve()}")
```
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Customer_Satisfaction_Score__1-10_.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Subscription_Plan__Numeric_.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Payment_History__Numeric_.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Support_Queries_Logged.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Age.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Monthly_Income____.png)
![alt text](notebooks/violinplots/violin_Health_Score_vs_Churn_Status__Numeric_.png)
![alt text](notebooks/violinplots/violin_Device_Used_Most_Often__Numeric__vs_Churn_Status__Numeric_.png)
![alt text](notebooks/violinplots/violin_Churn_Status__Numeric__vs_Promotional_Offers_Used.png)

Of all the violin plots, Promotional Offers by Churn Status is the only result that appears to show a relationship between promotional offers used and whether a customer churn. It appears that the more promotional offers that a customer receives, the more likely that they are to churn.

### Section 2.5 Combining demographics variables of age and income
```
# Combining age and income to compare to churn

df["age_group"] = pd.cut(df["Age"], bins=[18, 30, 45, 60, 90],
                         labels=["18–30", "30–45", "45–60", "60+"])

df["income_group"] = pd.qcut(df["Monthly Income ($)"], q=4,
                             labels=["Low", "Mid-Low", "Mid-High", "High"])


crosstab = pd.crosstab(
    [df["age_group"], df["income_group"]],
    df["Churn Status (Numeric)"]
)

crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)

sns.heatmap(crosstab_norm, annot=True, cmap="Blues")
plt.title("Churn Rate by Age Group × Income Group")
plt.savefig("heatmaps/churn_age_income_heatmap.png",
            dpi=300, bbox_inches="tight")
plt.show()
```
The below heatmap generated to compare demographics to churn:
![alt text](notebooks/heatmaps/churn_age_income_heatmap.png)

## Section 3 - Feature Selection and Justifications

### 3.1 Choose features and targets
**Case 1 - Customer Satisfaction Score**
Input features: Customer Satisfaction Score (1-10)
Target: Churn Status (Yes/No)

**Case 2 - Subscription Plan**
Input features: Subscription Plan
Target: Churn Status (Yes/No)

**Case 3 - Subscription Plan and Payment History**
Input features: Subscription Plan and Payment History (On-Time/Delayed)
Target: Churn Status (Yes/No)

**Case 4 - Support Queries Logged**
Input features: Support Queries Logged
Target: Churn Status (Yes/No)

**Case 5 - Demographics**
Input features: Age and Monthly Income ($)
Target: Churn Status (Yes/No)

**Case 6 - Health Score**
Input features: Health Score
Target: Churn Status (Yes/No)

**Case 7 - Genre Preference**
Input features: Genre Preference (Numeric)
Target: Churn Status (Yes/No)

**Case 8 - Device Used Most Often**
Input features: Device Used Most Often (Numeric)
Target: Churn Status (Yes/No)

**Case 9 - Promotional Offers**
Input features: Promotional Offers Used
Target: Churn Status (Yes/No)

### 3.2 - Define x and y
```
# Case 1. Customer Satisfaction Score
x1 = df[["Customer Satisfaction Score (1-10)"]]
y1 = df["Churn Status (Numeric)"]

# Case 2. Subscription Plan
x2 = df[["Subscription Plan (Numeric)"]]
y2 = df["Churn Status (Numeric)"]

# Case 3. Subscription Plan and Payment History
x3 = df[["Subscription Plan (Numeric)", "Payment History (Numeric)"]]
y3 = df["Churn Status (Numeric)"]

# Case 4. Support Queries Logged
x4 = df[["Support Queries Logged"]]
y4 = df["Churn Status (Numeric)"]

# Case 5. Demographics
x5 = df[["Age", "Monthly Income ($)"]]
y5 = df["Churn Status (Numeric)"]

# Case 6. Health Score
x6 = df[["Health_Score"]]
y6 = df["Churn Status (Numeric)"]

# Case 7. Genre Preference
x7 = df[["Genre Preference (Numeric)"]]
y7 = df["Churn Status (Numeric)"]

# Case 8. Device Used Most Often
x8 = df[["Device Used Most Often (Numeric)"]]
y8 = df["Churn Status (Numeric)"]

# Case 9. Promotional Offers Used
x9 = df[["Promotional Offers Used"]]
y9 = df["Churn Status (Numeric)"]
```

## Section 4 - Train a Model (Linear Regression)

### 4.1 - Split the data into training and test sets using train_test_split
```
# Splitting the data into training and test sets
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.2, random_state=42)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.2, random_state=42)
x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y6, test_size=0.2, random_state=42)
x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y7, test_size=0.2, random_state=42)
x8_train, x8_test, y8_train, y8_test = train_test_split(x8, y8, test_size=0.2, random_state=42)
x9_train, x9_test, y9_train, y9_test = train_test_split(x9, y9, test_size=0.2, random_state=42)
```

### 4.2 - Train model using Scikit-Learn model.fit() method for Linear Regression
```
# Training the model
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()
model6 = LinearRegression()
model7 = LinearRegression()
model8 = LinearRegression()
model9 = LinearRegression()

model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)
model3.fit(x3_train, y3_train)
model4.fit(x4_train, y4_train)
model5.fit(x5_train, y5_train)
model6.fit(x6_train, y6_train)
model7.fit(x7_train, y7_train)
model8.fit(x8_train, y8_train)
model9.fit(x9_train, y9_train)
```

### Section 4.3 - Evaluate Performance

```python
def evaluate_model(name, model, x_test, y_test):
    """Evaluate a trained model on test data and return metrics."""
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model: {name}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  MAE     : {mae:.3f}")
    print(f"  RMSE    : {rmse:.3f}")
    print("-" * 30)

evaluate_model("Model 1", model1, x1_test, y1_test)
evaluate_model("Model 2", model2, x2_test, y2_test)
evaluate_model("Model 3", model3, x3_test, y3_test)
evaluate_model("Model 4", model4, x4_test, y4_test)
evaluate_model("Model 5", model5, x5_test, y5_test)
evaluate_model("Model 6", model6, x6_test, y6_test)
evaluate_model("Model 7", model7, x7_test, y7_test)
evaluate_model("Model 8", model8, x8_test, y8_test)
evaluate_model("Model 9", model9, x9_test, y9_test)
```

![alt text](LinearRegression.jpg)

The results of the linear regression are included in the table above. Upon reviewing these results, no specific use case is performing any better than the other use cases. The R-Squared Scores are all close to 0 indicating that the model does not match the data very well.  The MAE and RMSE all have similar scores and indicated that none of the cases predict customer churn better than the others.

## Section 5 - Train a Model (Logistic Regression)

### 5.1 - Split the data into training and test sets using train_test_split
```
# Splitting the data into training and test data sets

df["Churn Status (Numeric)"]
features1 = ["Customer Satisfaction Score (1-10)"]
features2 = ["Subscription Plan (Numeric)"]
features3 = ["Subscription Plan (Numeric)", "Payment History (Numeric)"]
features4 = ["Support Queries Logged"]
features5 = ["Age", "Monthly Income ($)"]
features6 = ["Health_Score"]
features7 = ["Genre Preference (Numeric)"]
features8 = ["Device Used Most Often (Numeric)"]
features9 = ["Promotional Offers Used"]

x1 = df[features1]
x2 = df[features2]
x3 = df[features3]
x4 = df[features4]
x5 = df[features5]
x6 = df[features6]
x7 = df[features7]
x8 = df[features8]
x9 = df[features9]

y = df["Churn Status (Numeric)"]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.2, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.2, random_state=42)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, test_size=0.2, random_state=42)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y, test_size=0.2, random_state=42)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y, test_size=0.2, random_state=42)
x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y, test_size=0.2, random_state=42)
x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y, test_size=0.2, random_state=42)
x8_train, x8_test, y8_train, y8_test = train_test_split(x8, y, test_size=0.2, random_state=42)
x9_train, x9_test, y9_train, y9_test = train_test_split(x9, y, test_size=0.2, random_state=42)
```

### 5.2 - Setting up the training and testing models
```
# Training the model

df["Churn Status (Numeric)"]
features1 = ["Customer Satisfaction Score (1-10)"]
features2 = ["Subscription Plan (Numeric)"]
features3 = ["Subscription Plan (Numeric)", "Payment History (Numeric)"]
features4 = ["Support Queries Logged"]
features5 = ["Age", "Monthly Income ($)"]
features6 = ["Health_Score"]
features7 = ["Genre Preference (Numeric)"]
features8 = ["Device Used Most Often (Numeric)"]
features9 = ["Promotional Offers Used"]

x1 = df[features1]
x2 = df[features2]
x3 = df[features3]
x4 = df[features4]
x5 = df[features5]
x6 = df[features6]
x7 = df[features7]
x8 = df[features8]
x9 = df[features9]

y = df["Churn Status (Numeric)"]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.2, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.2, random_state=42)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, test_size=0.2, random_state=42)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y, test_size=0.2, random_state=42)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y, test_size=0.2, random_state=42)
x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y, test_size=0.2, random_state=42)
x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y, test_size=0.2, random_state=42)
x8_train, x8_test, y8_train, y8_test = train_test_split(x8, y, test_size=0.2, random_state=42)
x9_train, x9_test, y9_train, y9_test = train_test_split(x9, y, test_size=0.2, random_state=42)

model1 = LogisticRegression(max_iter=1000)
model2 = LogisticRegression(max_iter=1000)
model3 = LogisticRegression(max_iter=1000)
model4 = LogisticRegression(max_iter=1000)
model5 = LogisticRegression(max_iter=1000)
model6 = LogisticRegression(max_iter=1000)
model7 = LogisticRegression(max_iter=1000)
model8 = LogisticRegression(max_iter=1000)
model9 = LogisticRegression(max_iter=1000)

model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)
model3.fit(x3_train, y3_train)
model4.fit(x4_train, y4_train)
model5.fit(x5_train, y5_train)
model6.fit(x6_train, y6_train)
model7.fit(x7_train, y7_train)
model8.fit(x8_train, y8_train)
model9.fit(x9_train, y9_train)
```

### 5.3 - Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
```
# Classification
def evaluate_train_test_with_cm(models, x_trains, x_tests, y_trains, y_tests):
    """Evaluate multiple models on their train and test datasets."""
    results = []

    for i, (model, x_train, x_test, y_train, y_test) in enumerate(zip(models, x_trains, x_tests, y_trains, y_tests, strict=False), 1):
        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Train metrics
        train_scores = {
            "Model": f"Model {i} (Train)",
            "Accuracy": accuracy_score(y_train, y_train_pred),
            "Precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_train, y_train_pred)
        }

        # Test metrics
        test_scores = {
            "Model": f"Model {i} (Test)",
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_test_pred)
        }

        results.extend([train_scores, test_scores])

    return results

models = [model1, model2, model3, model4, model5, model6, model7, model8]

# Grouped inputs
x_trains = [x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train]
x_tests = [x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test]
y_trains = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y7_train, y8_train]
y_tests = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test, y7_test, y8_test]

# Run evaluation
results = evaluate_train_test_with_cm(models, x_trains, x_tests, y_trains, y_tests)

# Extract just the main metrics for table view
summary_results = [
    {k: v for k, v in res.items() if k != "Confusion Matrix"}
    for res in results
]

results_df = pd.DataFrame(summary_results)
def print_results_with_confusion_matrix(results):
    """Print a confusion matrix."""
    for res in results:
        print(f"\n{res['Model']}")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall: {res['Recall']:.4f}")
        print(f"F1 Score: {res['F1']:.4f}")
        print("Confusion Matrix:")
        print(res["Confusion Matrix"])
        print("-" * 40)
print_results_with_confusion_matrix(results)
```
![alt text](LogisticRegressionTrainCases.jpg)
![alt text](LogisticRegressionTestCases.jpg)

Upon a comparison of the training data and the test data, the results are fairly close indicating there does not appear to be overfitting. The accuracy results between 51.5\% and 56\% indicate that the results are only slightly better than accurate half of the time.  Based on the F1 score, Demographics, Genre Preference, and Device Used Most Often offer the best indicator of Churn Status, but their scores are not strong predictors only being between 47.40\% and 49.98\%.

## Section 6 - Train a Model (Logistic Regression)

### 6.1 - Split the data into training and test sets using train_test_split
```
# Splitting the test data
df["Churn Status (Numeric)"]
features1 = ["Customer Satisfaction Score (1-10)"]
features2 = ["Subscription Plan (Numeric)"]
features3 = ["Subscription Plan (Numeric)", "Payment History (Numeric)"]
features4 = ["Support Queries Logged"]
features5 = ["Age", "Monthly Income ($)"]
features6 = ["Health_Score"]
features7 = ["Genre Preference (Numeric)"]
features8 = ["Device Used Most Often (Numeric)"]
features9 = ["Promotional Offers Used"]

x1 = df[features1]
x2 = df[features2]
x3 = df[features3]
x4 = df[features4]
x5 = df[features5]
x6 = df[features6]
x7 = df[features7]
x8 = df[features8]
x9 = df[features9]

y = df["Churn Status (Numeric)"]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.2, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.2, random_state=42)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, test_size=0.2, random_state=42)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y, test_size=0.2, random_state=42)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y, test_size=0.2, random_state=42)
x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y, test_size=0.2, random_state=42)
x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y, test_size=0.2, random_state=42)
x8_train, x8_test, y8_train, y8_test = train_test_split(x8, y, test_size=0.2, random_state=42)
x9_train, x9_test, y9_train, y9_test = train_test_split(x9, y, test_size=0.2, random_state=42)
```

### 6.2 - Setting up the training and testing models
```
# Setting up the models

model1 = LogisticRegression(max_iter=1000)
model2 = LogisticRegression(max_iter=1000)
model3 = LogisticRegression(max_iter=1000)
model4 = LogisticRegression(max_iter=1000)
model5 = LogisticRegression(max_iter=1000)
model6 = LogisticRegression(max_iter=1000)
model7 = LogisticRegression(max_iter=1000)
model8 = LogisticRegression(max_iter=1000)
model9 = LogisticRegression(max_iter=1000)

model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)
model3.fit(x3_train, y3_train)
model4.fit(x4_train, y4_train)
model5.fit(x5_train, y5_train)
model6.fit(x6_train, y6_train)
model7.fit(x7_train, y7_train)
model8.fit(x8_train, y8_train)
model9.fit(x9_train, y9_train)

def evaluate_train_test_with_cm(models, x_trains, x_tests, y_trains, y_tests):
    """Evaluate multiple models on their train and test datasets."""
    results = []

    for i, (model, x_train, x_test, y_train, y_test) in enumerate(zip(models, x_trains, x_tests, y_trains, y_tests, strict=False), 1):
        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Train metrics
        train_scores = {
            "Model": f"Model {i} (Train)",
            "Accuracy": accuracy_score(y_train, y_train_pred),
            "Precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_train, y_train_pred)
        }

        # Test metrics
        test_scores = {
            "Model": f"Model {i} (Test)",
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_test_pred)
        }

        results.extend([train_scores, test_scores])

    return results

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]

# Grouped inputs
x_trains = [x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train]
x_tests = [x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test, x9_test]
y_trains = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y7_train, y8_train, y9_train]
y_tests = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test, y7_test, y8_test, y9_test]

# Run evaluation
results = evaluate_train_test_with_cm(models, x_trains, x_tests, y_trains, y_tests)

# Extract just the main metrics for table view
summary_results = [
    {k: v for k, v in res.items() if k != "Confusion Matrix"}
    for res in results
]

results_df = pd.DataFrame(summary_results)
def print_results_with_confusion_matrix(results):
    """Print a confusion matrix."""
    for res in results:
        print(f"\n{res['Model']}")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall: {res['Recall']:.4f}")
        print(f"F1 Score: {res['F1']:.4f}")
        print("Confusion Matrix:")
        print(res["Confusion Matrix"])
        print("-" * 40)
print_results_with_confusion_matrix(results)
```

### 6.3 - Clustering: Inertia, Silhoutte Score
```
# List of feature sets
x_list = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
case_names = ["Customer Satisfaction Score", "Subscription Plan", "Subscription Plan and Payment History", "Support Queries Logged", "Demographics", "Health Score", "Genre Preference", "Device Used Most Often", "Promotional Offers Used"]

# Store results
clustering_results = []

# Loop over all cases
for i, x in enumerate(x_list):
    x_clean = x.dropna()  # drop NaNs if present
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_clean)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(x_clean, kmeans.labels_)

    clustering_results.append({
        "Case": case_names[i],
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

# Display results
for res in clustering_results:
    print(f"\nCase: {res['Case']}")
    print(f"Inertia: {res['Inertia']:.2f}")
    print(f"Silhouette Score: {res['Silhouette Score']:.4f}")
    print("-" * 40)
```
### Section 6.4 - Visualization of the Clustering Results
![alt text](notebooks/clustering/x1_Customer_Satisfaction.png)
![alt text](notebooks/clustering/x2_Subscription_Plan.png)
![alt text](notebooks/clustering/x3_Plan_Payment.png)
![alt text](notebooks/clustering/x4_Support_Queries.png)
![alt text](notebooks/clustering/x5_Age_Income.png)
![alt text](notebooks/clustering/x6_Health_Score.png)
![alt text](notebooks/clustering/x7_Genre_Preference.png)
![alt text](notebooks/clustering/x8_Device_Used.png)
![alt text](notebooks/clustering/x9_Promo_Offers.png)

Given that the other testing methods are showing a poor ability to predict customer churn, one consideration may be to use the clusters in order to try predicting customer churn.

## Section 7 - Random Forest

### Section 7.1 - Splitting the training and testing data
```
df["Churn Status (Numeric)"]
features1 = ["Customer Satisfaction Score (1-10)"]
features2 = ["Subscription Plan (Numeric)"]
features3 = ["Subscription Plan (Numeric)", "Payment History (Numeric)"]
features4 = ["Support Queries Logged"]
features5 = ["Age", "Monthly Income ($)"]
features6 = ["Health_Score"]
features7 = ["Genre Preference (Numeric)"]
features8 = ["Device Used Most Often (Numeric)"]
features9 = ["Promotional Offers Used"]

x1 = df[features1]
x2 = df[features2]
x3 = df[features3]
x4 = df[features4]
x5 = df[features5]
x6 = df[features6]
x7 = df[features7]
x8 = df[features8]
x9 = df[features9]

y = df["Churn Status (Numeric)"]

results = {}

#train/test split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.2, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.2, random_state=42)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, test_size=0.2, random_state=42)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y, test_size=0.2, random_state=42)
x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y, test_size=0.2, random_state=42)
x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y, test_size=0.2, random_state=42)
x7_train, x7_test, y7_train, y7_test = train_test_split(x7, y, test_size=0.2, random_state=42)
x8_train, x8_test, y8_train, y8_test = train_test_split(x8, y, test_size=0.2, random_state=42)
x9_train, x9_test, y9_train, y9_test = train_test_split(x9, y, test_size=0.2, random_state=42)
```

### 7.2 -  Setting up the training and testing models
```
model1 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model2 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model3 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model4 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model5 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model6 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model7 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model8 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model9 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)
model3.fit(x3_train, y3_train)
model4.fit(x4_train, y4_train)
model5.fit(x5_train, y5_train)
model6.fit(x6_train, y6_train)
model7.fit(x7_train, y7_train)
model8.fit(x8_train, y8_train)
model9.fit(x9_train, y9_train)
```

### 7.3 - Random Forest: Accuracy, Precision, Recall, F1 Score
```
def evaluate_train_test_with_rf(models, x_trains, x_tests, y_trains, y_tests):
    """Evaluate multiple models using Random Forest on train and test datasets."""
    results = []

    for i, (model, x_train, x_test, y_train, y_test) in enumerate(zip(models, x_trains, x_tests, y_trains, y_tests, strict=False), 1):
        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Train metrics
        train_scores = {
            "Model": f"Model {i} (Train)",
            "Accuracy": accuracy_score(y_train, y_train_pred),
            "Precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
        }

        # Test metrics
        test_scores = {
            "Model": f"Model {i} (Test)",
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        }

        results.extend([train_scores, test_scores])

    return results

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]

# Grouped inputs
x_trains = [x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train]
x_tests = [x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test, x9_test]
y_trains = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y7_train, y8_train, y9_train]
y_tests = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test, y7_test, y8_test, y9_test]

# Run evaluation
results = evaluate_train_test_with_rf(models, x_trains, x_tests, y_trains, y_tests)

results_df = pd.DataFrame(summary_results)
def print_results(results):
    """Print a confusion matrix."""
    for res in results:
        print(f"\n{res['Model']}")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall: {res['Recall']:.4f}")
        print(f"F1 Score: {res['F1']:.4f}")
print_results(results)
```

