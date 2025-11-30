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

## 
